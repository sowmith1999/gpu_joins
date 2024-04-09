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
  for (std::pair<std::vector<int>, int> pair : rel.index.map) {
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

std::vector<int> joinRelations(const std::vector<Relation*>& relations) {
  if (relations.empty())
    return {};

  std::vector<int> result;

  for (int idx : relations[0]->index.sorted_arr) {
    bool matchFound = true;
    std::vector<int> combinedRow;

    std::vector<int> keys;
    for (int col : relations[0]->index_cols) {
      keys.push_back(relations[0]->data_arr[idx + col]);
    }

    for (size_t relIdx = 1; relIdx < relations.size(); ++relIdx) {
      auto& rel = relations[relIdx];
      auto it = rel->index.map.find(keys);
      if (it == rel->index.map.end()) {
        matchFound = false;
        break;
      }

      int matchingRowStart = rel->index.sorted_arr[it->second];
      for (int i = 0; i < rel->num_cols; ++i) {
        if (std::find(rel->index_cols.begin(), rel->index_cols.end(), i) ==
                rel->index_cols.end() ||
            relIdx == 1) {
          combinedRow.push_back(rel->data_arr[matchingRowStart + i]);
        }
      }
    }

    if (matchFound) {
      result.insert(result.end(), combinedRow.begin(), combinedRow.end());
    }
  }

  return result;
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
  std::vector<int> graph_4_data{1, 2, 10, 11, 12, 13, 14, 2, 3};
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

  // std::vector<int>* join_ret = joinRelation(graph1, graph2, graph3);
  // char join_name[] = "graph_new";
  // Relation* join_temp = make_rel(*join_ret, join_name, num_cols,
  //                                join_ret->size() / num_cols, index_cols, 1);

  std::vector<Relation*> relations = {graph1, graph2, graph3};
  std::vector<int> join_ret = joinRelations(relations);
  std::cout << "The joined data array on three graphs is: " << std::endl;

  int result_columns = 2;

  for (size_t i = 0; i < join_ret.size(); i += result_columns) {
    for (int j = 0; j < result_columns; ++j) {
      std::cout << join_ret[i + j] << " ";
    }
    std::cout << "\n";
  }

  char graph4_name[] = "graph4";
  Relation* graph4 =
      make_rel(graph_4_data, graph4_name, num_cols, 6, index_cols, 1);
  // printRelation(*graph4, 1, 1);

  relations = {graph1, graph2, graph3, graph4};
  join_ret = joinRelations(relations);
  std::cout << "\nThe joined data array on four graphs is: " << std::endl;

  // This could probably be a function or 
  for (size_t i = 0; i < join_ret.size(); i += result_columns) {
    for (int j = 0; j < result_columns; ++j) {
      std::cout << join_ret[i + j] << " ";
    }
    std::cout << "\n";
  }

  // printRelation(*join_temp, 1, 1);
  // delete join_ret;
  deleteRelation(graph1);
  deleteRelation(graph2);
  deleteRelation(graph3);
  deleteRelation(graph4);
  // deleteRelation(join_temp);
  return 0;
}
