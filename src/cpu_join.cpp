#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <unordered_map>

void raiseError(const char *errorMessage) {
    fprintf(stderr, "Error: %s\n", errorMessage); // Print the error message to stderr
    exit(EXIT_FAILURE); // Exit the program with a failure status
}
// hardcode the input for now
typedef struct Index {
  int **sorted_array;
  std::unordered_map<int, int> map;
} Index;

typedef struct Relation {
  char *name;
  int num_rows;
  int num_cols;
  int index_col;
  Index index;
  int **rows; // data_array
} Relation;
/*
 * Takes a relation, and row number in sorted array,
 * give the pointer to next row as per sorted array
 * */
int *nextRow(Relation *rel, int row_idx) {
  return rel->index.sorted_array[row_idx + 1];
}

/*
 * Takes a Relation and sorted_array row number and return the pointer to row
 * */
int *getRow(Relation *rel, int row_idx) {
  // printf("In getRow, for relation %s and row_idx %d\n", rel->name, row_idx);
  if(row_idx < rel->num_rows)
    return rel->index.sorted_array[row_idx];
  raiseError("In getRow, row_idx greater than num_row");
  return nullptr;
}

void printRow(Relation *rel, int *row) {
  for (int i = 0; i < rel->num_cols; i++)
    printf("%d\t", row[i]);
}

void initializeIndex(Relation &rel) {
  // do indexing
  // take data_array and do radix sort, and store the pointers to the rows in
  // that order for now, we can directly use the data_array as it is already
  // sorted
  Index &index = rel.index;
  index.sorted_array = (int **)malloc(rel.num_rows * (sizeof(int *)));
  for (int i = 0; i < rel.num_rows; i++) {
    index.sorted_array[i] =
        rel.rows[i]; // storing the pointer to the row in data array
  }

  int prev_hash = -1;
  for (int i = 0; i < rel.num_rows; i++) {
    // hash the index_cols and use that as key
    // hash all the index cols together, if there is more than and use that hash
    // for now directly using the values of the index_column
    int cur_hash = index.sorted_array[i][rel.index_col];
    if (cur_hash != prev_hash) { // skipping over rows whose index is already in
                                 // hash table, as we iter through sorted array
                                 // only need to check against previous hash
      index.map[cur_hash] = i;   // storing the first occurrence of new hash as
                                 // the index value in sorted array
      prev_hash = cur_hash;
    }
  }
}

void initializeRelation(Relation &rel, char *name, int num_cols, int num_rows,
                        int index_col, int *data) {
  rel.name = (char *)malloc(strlen(name) + 1);
  strcpy(rel.name, name);
  rel.num_cols = num_cols;
  rel.num_rows = num_rows;
  rel.index_col = index_col;

  rel.rows = (int **)malloc(rel.num_rows * sizeof(int *));
  for (int i = 0; i < rel.num_rows; i++) {
    rel.rows[i] = (int *)malloc(rel.num_cols * sizeof(int));
    for (int j = 0; j < rel.num_cols; j++) {
      rel.rows[i][j] = data[i * rel.num_cols + j];
    }
  }
  initializeIndex(rel);
}

void printIndex(Relation &rel) {
  printf("Keys indexed are\n");
  for (auto &pair : rel.index.map)
    std::cout << pair.first << std::endl;
}

void printRelation(Relation &rel, bool print_data, bool print_index) {
  printf("The relation name is: %s\n", rel.name);
  printf("Number of columns: %d\n", rel.num_cols);
  printf("Number of rows: %d\n", rel.num_rows);
  printf("The index col is: %d\n", rel.index_col);
  if (print_data) {
    for (int i = 0; i < rel.num_rows; i++) {
      for (int j = 0; j < rel.num_cols; j++)
        printf("%d \t", rel.rows[i][j]);
      printf("\n");
    }
  }
  if (print_index)
    printIndex(rel);
}

void deleteRelation(Relation *rel) {
  free(rel->name);
  for (int i = 0; i < rel->num_rows; i++)
    free(rel->rows[i]);
  free(rel->rows);
  free(rel->index.sorted_array);
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
void joinRelation(Relation *rel1, Relation *rel2) {
  // rel1 hash loop
  for (auto &pair : rel1->index.map) {
    int key = pair.first; // this would be a hash, and the value is index in the
                          // sorted array
    // pointer to the first row with this key on sorted array
    printf("The key is : %d\n", key);
    printf("The rel one values for key:\n");
    int row_idx = rel1->index.map[key];
    int *row = getRow(rel1, row_idx);
    while (row[rel1->index_col] == key) {
      printRow(rel1, row);
      printf("\n");
      row_idx++;
      if(row_idx >= rel1->num_rows) break;
      row = getRow(rel1, row_idx);
    }
    printf("The rel two values for key:\n");
    row_idx = rel2->index.map[key];
    row = getRow(rel2, row_idx);
    while (row[rel2->index_col] == key) {
      printRow(rel2, row);
      printf("\n");
      row_idx++;
      if(row_idx >= rel2->num_rows) break;
      row = getRow(rel2, row_idx);
    }
  }
}

Relation *make_rel(int *init_values, char *name, int num_cols, int num_rows,
                   int index_col) {
  Relation *rel = new Relation();
  int *data = (int *)calloc(num_rows * num_cols, sizeof(int));
  for (int i = 0; i < num_rows * num_cols; i++)
    data[i] = init_values[i];
  initializeRelation(*rel, name, num_cols, num_rows, index_col, data);
  free(data);
  return rel;
}

int main() {
  int num_cols = 2;
  int num_rows = 6;
  // int edge_index= 0;
  // int index_col2 = 1;
  // relation 1
  char rel1_name[] = "path";
  int init_values_1[] = {1, 2, 2, 3, 2, 4, 3, 5, 4, 5, 5, 6};
  Relation *path = make_rel(init_values_1, rel1_name, num_cols, num_rows, 1);
  printRelation(*path, 1, 1);

  // relation 2
  char rel2_name[] = "edge";
  int init_values_2[] = {1, 2, 2, 3, 2, 4, 3, 5, 4, 5, 5, 6};
  Relation *edge = make_rel(init_values_2, rel2_name, num_cols, num_rows, 0);
  printRelation(*edge, 1, 1);

  //    printf("The index val 3 points to: %d\n",
  //    (*rel1).index.sorted_array[(*rel1).index.map[6]][2]);
  joinRelation(path, edge);
  deleteRelation(path);
  deleteRelation(edge);
  return 0;
}
