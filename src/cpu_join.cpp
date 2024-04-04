#include <stdio.h>
#include <unordered_map>
#include <cstdlib>
#include <string.h>
#include <iostream>

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

void initializeIndex(Relation &rel) {
    // do indexing
    // take data_array and do radix sort, and store the pointers to the rows in that order
    // for now, we can directly use the data_array as it is already sorted
    Index &index = rel.index;
    index.sorted_array = (int **) malloc(rel.num_rows * (sizeof(int *)));
    for (int i = 0; i < rel.num_rows; i++) {
        index.sorted_array[i] = rel.rows[i]; // storing the pointer to the row in data array
    }

    int prev_hash = -1;
    for (int i = 0; i < rel.num_rows; i++) {
        // hash the index_cols and use that as key
        // hash all the index cols together, if there is more than and use that hash
        // for now directly using the values of the index_column
        int cur_hash = index.sorted_array[i][rel.index_col];
        if (cur_hash !=
            prev_hash) { // skipping over rows whose index is already in hash table, as we iter through sorted array only need to check against previous hash
            index.map[cur_hash] = i; // storing the first occurrence of new hash as the index value in sorted array
            prev_hash = cur_hash;
        }
    }
}

void initializeRelation(Relation &rel, char *name, int num_cols, int num_rows, int index_col, int *data) {
    rel.name = (char *) malloc(strlen(name) + 1);
    strcpy(rel.name, name);
    rel.num_cols = num_cols;
    rel.num_rows = num_rows;
    rel.index_col = index_col;

    rel.rows = (int **) malloc(rel.num_rows * sizeof(int *));
    for (int i = 0; i < rel.num_rows; i++) {
        rel.rows[i] = (int *) malloc(rel.num_cols * sizeof(int));
        for (int j = 0; j < rel.num_cols; j++) {
            rel.rows[i][j] = data[i * rel.num_cols + j];
        }
    }
    initializeIndex(rel);
}

void printIndex(Relation &rel) {
    printf("Keys indexed are\n");
    for (auto &pair: rel.index.map)
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
    if (print_index) printIndex(rel);
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
* Create a new relation, iterates through the
*/
//Relation* joinRelation(Relation* rel1, Relation* rel1){
//
//}
//
Relation *make_rel(int *init_values, char *name, int num_cols, int num_rows, int index_col) {
    Relation *rel = new Relation();
    int *data = (int *) calloc(num_rows * num_cols, sizeof(int));
    for (int i = 0; i < num_rows * num_cols; i++)
        data[i] = init_values[i];
    initializeRelation(*rel, name, num_cols, num_rows, index_col, data);
    free(data);
    return rel;
}

int main() {
    int num_cols = 3;
    int num_rows = 4;
    int index_col = 0;
    // relation 1
    char rel1_name[] = "relation_1";
    int init_values_1[] = {0, 1, 2, 3, 4, 5, 3, 4, 6, 6, 7, 8};
    Relation *rel1 = make_rel(init_values_1, rel1_name, num_cols, num_rows, index_col);
    printRelation(*rel1, 1, 1);

    // relation 2
    char rel2_name[] = "relation_2";
    int init_values_2[] = {0, 1, 2, 3, 4, 7, 3, 4, 8, 6, 7, 8};
    Relation *rel2 = make_rel(init_values_2, rel2_name, num_cols, num_rows, index_col);
    printRelation(*rel2, 1, 1);
//    printf("The index val 3 points to: %d\n", (*rel1).index.sorted_array[(*rel1).index.map[6]][2]);
    deleteRelation(rel1);
    deleteRelation(rel2);
    return 0;
}