#include <algorithm>
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
// hardcode the input for now
typedef struct Index {
  // Stores the starting offset of rows, sorted stably from
  // least significant to most significant column
  // Index is always the most significant column
  std::vector<int> sorted_arr;
  // stores the key(index col val) and value is index in sorted_array
  std::unordered_map<int, int> map;
} Index;

typedef struct Relation {
  char* name;
  int num_rows;
  int num_cols;
  int index_col;
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
  printf("the sorted array is:\n");

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

  int prev_hash = -1;
  for (int i = 0; i < rel.num_rows; i++) {
    int cur_hash =
        rel.data_arr[index.sorted_arr[i]]; // this assumes first values is
                                           // the index and only first value
    if (cur_hash != prev_hash) { // skipping over rows whose index is already in
      index.map[cur_hash] = i;
      prev_hash = cur_hash;
    }
  }
}

void initializeRelation(Relation& rel, char* name, int num_cols, int num_rows,
                        int index_col, const std::vector<int>& graph) {
  rel.name = (char*)malloc(strlen(name) + 1);
  strcpy(rel.name, name);
  rel.num_cols = num_cols;
  rel.num_rows = num_rows;
  rel.index_col = index_col;
  rel.data_arr = graph; // this does a deep copy.
}

void printIndex(Relation& rel) {
  printf("Sorted offsets are:\n");
  for (int offset : rel.index.sorted_arr)
    printf("%d\t", offset);
  printf("\n");
  printf("Keys indexed are\n");
  for (std::pair<int, int> pair : rel.index.map)
    std::cout << rel.data_arr[rel.index.sorted_arr[pair.second]] << std::endl;
  printf("\n");
}

void printRelation(Relation& rel, bool print_data, bool print_index) {
  printf("The relation name is: %s\n", rel.name);
  printf("Number of columns: %d\n", rel.num_cols);
  printf("Number of rows: %d\n", rel.num_rows);
  printf("The index col is: %d\n", rel.index_col);
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

/*
 * Takes two relations, assumes they are being joined on their indexed column.
 * Create a new relation, iterates through the hash table, and when there are
 * matching hashes then iter through both tables and make new rows, and stop
 * once index column values change Do we have to think about inner relation and
 * outer relation, I don't think so, right, coz column that we are matching on
 * is indexed in both tables, or doesn't matter which one is which...
 */
std::vector<int>* joinRelation(Relation* path, Relation* edge) {
  // make a new relation to store the tuples from the join
  std::vector<int>* path_new_data_array = new std::vector<int>;
  for (auto& pair : path->index.map) {
    int key = pair.first; // this is the key value
    // data_array offset to the first row with this key on sorted array
    int path_srtd_idx = path->index.map[key];
    int path_data_idx = path->index.sorted_arr[path_srtd_idx];
    printf("The key is: %d\n", key);
    printf("The outer key is: %d\n", path->data_arr[path_data_idx]);
    while (path->data_arr[path_data_idx] == key) {
      // not using hash here
      int edge_srtd_idx = edge->index.map[key];
      int edge_data_idx = edge->index.sorted_arr[edge_srtd_idx];
      printf("Outer row\n");
      printRow(path, path_data_idx);
      printf("inner row\n");
      while (edge->data_arr[edge_data_idx] == key) {
        printRow(edge, edge_data_idx);
        // there should be a better way to decide which way tuples go in
        // they don't clearly follow what datalog rule says, coz index columns
        // are put first, and then when inserting into path, we reverse the
        // initial input data.
        path_new_data_array->push_back(edge->data_arr[edge_data_idx + 1]);
        path_new_data_array->push_back(path->data_arr[path_data_idx + 1]);
        edge_srtd_idx++;
        edge_data_idx = edge->index.sorted_arr[edge_srtd_idx];
        if (edge_data_idx >= edge->data_arr.size())
          break;
      }
      path_srtd_idx++;
      path_data_idx = path->index.sorted_arr[path_srtd_idx];
      if (path_data_idx >= path->data_arr.size())
        break;
    }
  }
  printf("The joined data array is:\n");
  for (int i = 0; i < path_new_data_array->size(); i += 2) {
    std::cout << path_new_data_array->at(i) << " "
              << path_new_data_array->at(i + 1) << std::endl;
  }
  return path_new_data_array;
}

// Takes the path and path_new created from the join operation
void mergeRelation(Relation* path, Relation* path_new) {}
Relation* make_rel(const std::vector<int>& graph, char* name, int num_cols,
                   int num_rows, int index_col, bool do_index) {
  Relation* rel = new Relation();
  initializeRelation(*rel, name, num_cols, num_rows, index_col, graph);
  if (do_index)
    initializeIndex(*rel);
  return rel;
}

int main() {
  int num_cols = 2;
  int num_rows = 6;
  std::vector<int> graph_edge{1, 2, 2, 3, 2, 4, 3, 5, 4, 5, 5, 6};
  std::vector<int> graph_path{2, 1, 4, 2, 3, 2, 5, 3, 5, 4, 6, 5};

  char path_name[] = "path";
  Relation* path = make_rel(graph_path, path_name, num_cols, num_rows, 0, 1);
  printRelation(*path, 1, 1);

  char edge_name[] = "edge";
  Relation* edge = make_rel(graph_edge, edge_name, num_cols, num_rows, 0, 1);
  printRelation(*edge, 1, 1);

  std::vector<int>* join_ret = joinRelation(path, edge);
  char join_name[] = "path_new";
  Relation* join_temp = make_rel(*join_ret, join_name, num_cols,
                                 join_ret->size() / num_cols, 0, 1);
  printRelation(*join_temp, 1, 1);
  delete join_ret;
  deleteRelation(path);
  deleteRelation(edge);
  deleteRelation(join_temp);
  return 0;
}
