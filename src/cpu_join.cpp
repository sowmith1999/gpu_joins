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
  std::vector<int> sorted_array;
  // stores the key and the location in sorted_array
  std::unordered_map<int, int> map;
} Index;

typedef struct Relation {
  char* name;
  int num_rows;
  int num_cols;
  int index_col;
  Index index;
  std::vector<int> data_array;
} Relation;

void printRow(Relation* rel, int* row) {
  for (int i = 0; i < rel->num_cols; i++)
    printf("%d\t", row[i]);
}

void initializeIndex(Relation& rel) {
  Index& index = rel.index;
  for (int i = 0; i < rel.num_rows * rel.num_cols; i = i + rel.num_cols) {
    index.sorted_array.push_back(i);
  }
  // at this point sorted_arry holds all the offsets for rows
  // but is not sorted yet, to sort based on each col
  for (int col = 0; col < rel.num_cols; col++) {
    std::stable_sort(index.sorted_array.begin(), index.sorted_array.end(),
                     [rel, col](int offset1, int offset2) {
                       return (rel.data_array[offset1 + col]) <
                              (rel.data_array[offset2 + col]);
                     });
  }

  int prev_hash = -1;
  for (int i = 0; i < rel.num_rows; i++) {
    int cur_hash = rel.data_array[index.sorted_array[i]];
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
  rel.data_array = graph; // this does a deep copy.
}

void printIndex(Relation& rel) {
  printf("Keys indexed are\n");
  for (auto& pair : rel.index.map)
    std::cout << rel.data_array[rel.index.sorted_array[pair.second]]
              << std::endl;
  printf("Sorted offsets are:\n");
  for (int offset : rel.index.sorted_array)
      printf("%d\t",offset);
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
        printf("%d \t", rel.data_array[i * rel.num_cols + j]);
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
  auto join_ret = new std::vector<int>;
  for (auto& pair : path->index.map) {
    int key = pair.first; // this would be a hash, and the value is index in the
                          // sorted array
    // pointer to the first row with this key on sorted array
    int row_idx = path->index.map[key];
    while (path->data_array[row_idx] == key) {
      int inner_row_idx = edge->index.map[key];
      while (edge->data_array[inner_row_idx] == key) {
        join_ret->push_back(path->data_array[row_idx]);
        join_ret->push_back(edge->data_array[inner_row_idx + 1]);
        inner_row_idx += edge->num_cols;
        if (inner_row_idx >= edge->data_array.size())
          break;
      }
      row_idx += path->num_cols;
      if (row_idx >= path->num_rows)
        break;
    }
  }
  for (int i = 0; i < join_ret->size(); i += 2) {
    std::cout << join_ret->at(i) << " " << join_ret->at(i + 1) << std::endl;
  }
  return join_ret;
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
  int path_index_col = 1;
  int edge_index_col = 0;
  std::vector<int> graph_edge{1, 2, 2, 3, 2, 4, 3, 5, 4, 5, 5, 6};
  std::vector<int> graph_path{2, 1, 4, 2, 3, 2, 5, 3, 5, 4, 6, 5};

  char path_name[] = "path";
  Relation* path = make_rel(graph_path, path_name, num_cols, num_rows, 0, 1);
  printRelation(*path, 1, 1);

  char edge_name[] = "edge";
  Relation* edge = make_rel(graph_edge, edge_name, num_cols, num_rows, 0, 1);
  printRelation(*edge, 1, 1);

  // std::vector<int>* join_ret = joinRelation(path, edge);
  // char join_name[] = "path_new";
  // Relation* join_temp = make_rel(*join_ret, join_name, num_cols,
  //                                join_ret->size(), path_index_col, 1);
  // delete join_ret;
  deleteRelation(path);
  deleteRelation(edge);
  // deleteRelation(join_temp);
  return 0;
}
