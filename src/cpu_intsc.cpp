#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <unordered_map>
#include <vector>

void raiseError(const char* errorMessage) {
  fprintf(stderr, "Error: %s\n",
          errorMessage); // Print the error message to stderr
  exit(EXIT_FAILURE);    // Exit the program with a failure status
}

typedef struct Index {
  // Stores the starting offset of rows, sorted stably from
  // least significant to most significant column
  // Index is always the most significant column
  std::vector<int> sorted_arr;
  // stores the key(index col val) and value is index in sorted_array
  std::unordered_map<std::vector<int>, int, boost::hash<std::vector<int>>> map;
} Index;

typedef struct Relation {
  char* name;
  int num_rows;
  int num_cols;
  std::vector<int> index_cols;
  Index index;
  std::vector<int> data_arr;
} Relation;

void printRow(Relation* rel, int offset) {
  for (int i = offset; i < offset + rel->num_cols; i++)
    printf("%d\t", rel->data_arr[i]);
  printf("\n");
}

void initializeIndex(Relation& rel) {
  Index& index = rel.index;
  for (int i = 0; i < rel.num_rows * rel.num_cols; i = i + rel.num_cols)
    index.sorted_arr.push_back(i);

  // at this point sorted_arry holds all the offsets for rows
  // but is not sorted yet, to sort based on each col
  for (int col = rel.num_cols - 1; col >= 0; col--)
    std::stable_sort(index.sorted_arr.begin(), index.sorted_arr.end(),
                     [rel, col](int offset1, int offset2) {
                       // printf("offset1 %d, offset2: %d, tot_size = %zu\n",
                       //        offset1, offset2, rel.data_arr.size());
                       return (rel.data_arr[offset1 + col]) <
                              (rel.data_arr[offset2 + col]);
                     });

  std::vector<int> prev_row_idx_vals;
  for (int i = 0; i < rel.num_rows; i++) {
    std::vector<int> cur_row_idx_vals;
    for (int col : rel.index_cols) {
      cur_row_idx_vals.push_back(
          rel.data_arr[rel.index.sorted_arr[i] +
                       col]); // this assumes first values is
                              // the index and only first value
    }
    if (cur_row_idx_vals !=
        prev_row_idx_vals) { // skipping over rows whose index is already in
      index.map[cur_row_idx_vals] = i;
      prev_row_idx_vals = cur_row_idx_vals;
    }
  }
}

void initializeRelation(Relation& rel, char* name, int num_cols, int num_rows,
                        std::vector<int> index_cols,
                        const std::vector<int>& graph) {
  rel.name = (char*)malloc(strlen(name) + 1);
  strcpy(rel.name, name);
  rel.num_cols = num_cols;
  rel.num_rows = num_rows;
  rel.index_cols = index_cols;
  rel.data_arr = graph; // this does a deep copy.
}

void printIndex(Relation& rel) {
  printf("Sorted offsets are:\n");
  for (int offset : rel.index.sorted_arr)
    printf("%d\t", offset);
  printf("\n");
  printf("Keys indexed are\n");
  for (std::pair<std::vector<int>, int> pair : rel.index.map){
    for (int col : pair.first)
      printf("%d\t", rel.data_arr[rel.index.sorted_arr[pair.second] + col]);
    printf("\n");
  }
  printf("\n");
}

void printRelation(Relation& rel, bool print_data, bool print_index) {
  printf("The relation name is: %s\n", rel.name);
  printf("Number of columns: %d\n", rel.num_cols);
  printf("Number of rows: %d\n", rel.num_rows);
  for (int col : rel.index_cols)
    printf("The index col is: %d\t", col);
  printf("\n");
  if (print_data) {
    for (int i = 0; i < rel.num_rows; i++) {
      for (int j = 0; j < rel.num_cols; j++)
        printf("%d \t", rel.data_arr[i * rel.num_cols + j]);
      printf("\n");
    }
  }
  if (print_index)
    printIndex(rel);
}

void deleteRelation(Relation* rel) {
  free(rel->name);
  delete rel;
}

bool compareVec(std::vector<int>* data_array, std::vector<int>* keys,
                int begin) {
  for (int i = 0; i < keys->size(); i++) {
    if (data_array->at(begin + i) != keys->at(i))
      return false;
  }
  return true;
}

/*
 * Takes two relations, assumes they are being joined on their indexed column.
 * Create a new relation, iterates through the hash table, and when there are
 * matching hashes then iter through both tables and make new rows, and stop
 * once index column values change Do we have to think about inner relation and
 * outer relation, I don't think so, right, coz column that we are matching on
 * is indexed in both tables, or doesn't matter which one is which...
 */
std::vector<int>* joinRelation(Relation* graph1, Relation* graph2,
                               Relation* graph3) {
  // make a new relation to store the tuples from the join
  std::vector<int>* intsc_new_data_array = new std::vector<int>;
  for (auto& pair : graph1->index.map) {
    std::vector<int> keys = pair.first; // this is the key value
    // data_array offset to the first row with this key on sorted array
    int graph1_srtd_index = graph1->index.map[keys];
    int graph1_data_idx = graph1->index.sorted_arr[graph1_srtd_index];
    // printf("The key is: %d\n", keys);
    printf("The outer key is: %d\n", graph1->data_arr[graph1_data_idx]);
    // while (graph1->data_arr[graph1_data_idx] == keys) {
    while (compareVec(&graph1->data_arr, &keys, graph1_data_idx)) {
      // not using hash here
      int graph2_srtd_idx = graph2->index.map[keys];
      int graph2_data_idx = graph2->index.sorted_arr[graph2_srtd_idx];
      printf("Outer row\n");
      printRow(graph1, graph1_data_idx);
      // while (graph2->data_arr[graph2_edge_idx] == keys) {
      while (compareVec(&graph2->data_arr, &keys, graph2_data_idx)) {
        int graph3_srtd_idx = graph3->index.map[keys];
        int graph3_data_idx = graph3->index.sorted_arr[graph3_srtd_idx];
        printf("inner row\n");
        printRow(graph2, graph2_data_idx);
        while (compareVec(&graph3->data_arr, &keys, graph3_data_idx)) {
          // there should be a better way to decide which way tuples go in
          // they don't clearly follow what datalog rule says, coz index columns
          // are put first, and then when inserting into path, we reverse the
          // initial input data.
          intsc_new_data_array->push_back(graph1->data_arr[graph1_data_idx]);
          intsc_new_data_array->push_back(
              graph1->data_arr[graph1_data_idx + 1]);
          graph3_srtd_idx++;
          graph3_data_idx = graph3->index.sorted_arr[graph3_srtd_idx];
          if (graph3_data_idx >= graph3->data_arr.size())
            break;
        }
        graph2_srtd_idx++;
        graph2_data_idx = graph2->index.sorted_arr[graph2_srtd_idx];
        if (graph2_data_idx >= graph2->data_arr.size())
          break;
      }
      graph1_srtd_index++;
      graph1_data_idx = graph1->index.sorted_arr[graph1_srtd_index];
      if (graph1_data_idx >= graph1->data_arr.size())
        break;
    }
  }
  printf("The joined data array is:\n");
  for (int i = 0; i < intsc_new_data_array->size(); i += 2) {
    std::cout << intsc_new_data_array->at(i) << " "
              << intsc_new_data_array->at(i + 1) << std::endl;
  }
  return intsc_new_data_array;
}

// Takes the path and path_new created from the join operation
// and merges them, I think there is another step of deduplication or
// creation of delta before this
void mergeRelation(Relation* path, Relation* path_new) {
  // I basically loop through the path_new as outer and path as inner
  // use the index, and look for same tuples, and insert the ones from delta
  // that don't math into another vector;
  // Now build a new relation with that vector and make index
  // and then append the delta and full vectors together, merge the sorted
  // arrays. merge the hash maps One thing we don't want is a new relation, we
  // want to actually merge
  // for (auto& pair : path_new->index.map) {
  //   int key = pair.first;
  // }
}

Relation* make_rel(const std::vector<int>& graph, char* name, int num_cols,
                   int num_rows, std::vector<int> index_cols, bool do_index) {
  Relation* rel = new Relation();
  initializeRelation(*rel, name, num_cols, num_rows, index_cols, graph);
  if (do_index)
    initializeIndex(*rel);
  return rel;
}

int main() {
  int num_cols = 2;
  int num_rows = 6;
  std::vector<int> graph_1_data{1, 2, 2, 3, 2, 4, 3, 5, 4, 5, 5, 6};
  std::vector<int> graph_2_data{1, 2, 2, 3, 3, 4, 3, 5, 5, 6};
  std::vector<int> graph_3_data{1, 2, 2, 4, 2, 5, 3, 4, 4, 5, 5, 6};
  std::vector<int> index_cols{0, 1};

  char graph1_name[] = "graph1";
  Relation* graph1 =
      make_rel(graph_1_data, graph1_name, num_cols, 6, index_cols, 1);
  printRelation(*graph1, 1, 1);

  char graph2_name[] = "graph2";
  Relation* graph2 =
      make_rel(graph_2_data, graph2_name, num_cols, 5, index_cols, 1);
  printRelation(*graph2, 1, 1);

  char graph3_name[] = "graph3";
  Relation* graph3 =
      make_rel(graph_3_data, graph3_name, num_cols, 6, index_cols, 1);
  printRelation(*graph3, 1, 1);

  std::vector<int>* join_ret = joinRelation(graph1, graph2, graph3);
  char join_name[] = "graph_new";
  Relation* join_temp = make_rel(*join_ret, join_name, num_cols,
                                 join_ret->size() / num_cols, index_cols, 1);

  printRelation(*join_temp, 1, 1);
  delete join_ret;
  deleteRelation(graph1);
  deleteRelation(graph2);
  deleteRelation(graph3);
  deleteRelation(join_temp);
  return 0;
}
