#include "exception.cuh"
#include "hisa.cu"
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>
#define BLOCK_SIZE 256
#define DEBUG 1
using u32 = uint32_t;
struct KernelTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  KernelTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~KernelTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void start_timer() { cudaEventRecord(start, 0); }

  void stop_timer() { cudaEventRecord(stop, 0); }

  float get_spent_time() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    elapsed /= 1000.0;
    return elapsed;
  }
};

// Here we take two relation and create a hash based index on them
// and do a join on them, but instead of doing a directly we split the join into
// batches, it shouldn't be too different if at all.
std::vector<u32>* readInput(const std::string& filename) {
  std::vector<u32>* data = new std::vector<u32>();
  std::ifstream file(filename);
  std::string line;
  int number;
  if (!file.is_open()) {
    std::cerr << "Error opening the file" << std::endl;
    return data;
  }

  while (getline(file, line)) {
    std::istringstream iss(line);
    while (iss >> number) {
      data->push_back(number);
    }
  }
  file.close();
  return data;
}

/*
Takes an integer and generates a linear path graph with that many edges
and edgeSize+1 nodes
*/
std::vector<u32>* linearGraph(u32 edgeSize) {
  std::vector<u32>* data = new std::vector<u32>();
  for (u32 i = 0; i < edgeSize; i++) {
    data->push_back(i);
    data->push_back(i + 1);
  }
  return data;
}

typedef struct Relation {
  char* name = nullptr;
  u32 num_rows;
  KeyValue* map = nullptr;
  u32* sorted_arr;
  u32* data_arr;
} Relation;

__global__ void testKernel(Relation* rel) {
  printf("The Relation is %s\n", rel->name);
  printf("\t rel.numrows = %d\n", rel->num_rows);
  printf("\n");
  printf("The data array is: \n");
  for (int i = 0; i < rel->num_rows; i++) {
    printf("%d\t%d", rel->data_arr[rel->sorted_arr[i]],
           rel->data_arr[rel->sorted_arr[i] + 1]);
    printf("\n");
  }
  printf("The sorted arrray is:\n");
  for (int i = 0; i < rel->num_rows; i++) {
    printf("%d ", rel->sorted_arr[i]);
  }
  printf("\n");
  printf("The map is:\n");
  for (int i = 0; i < kHashTableCapacity; i++) {
    if (rel->map[i].key != kEmpty)
      printf("%d\t%d\n", rel->map[i].key, rel->map[i].value);
  }
}

void printRelation(Relation* rel) {
  if (DEBUG) {
    testKernel<<<1,1>>>(rel);
    checkCuda(cudaDeviceSynchronize());
  }
}

__global__ void initSortedArr(Relation* rel) {
  u32 idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < rel->num_rows)
    rel->sorted_arr[idx] = idx * 2;
}

__global__ void initMap(Relation* rel) {
  u32 cur_idx = threadIdx.x + blockDim.x * blockIdx.x;
  u32 prev_idx = cur_idx - 1;
  if (cur_idx < rel->num_rows) {
    if (cur_idx == 0 || (rel->data_arr[rel->sorted_arr[cur_idx]] !=
                         rel->data_arr[rel->sorted_arr[prev_idx]])) {
      u32 key = rel->data_arr[rel->sorted_arr[cur_idx]];
      u32 value = cur_idx;
      insertKey(rel->map, key, value);
    }
  }
}

Relation* makeRelation(std::vector<u32>* data, char* name) {
  if (data->size() % 2 != 0) {
    fprintf(stderr, "Error: number of elements: %d for data:%s isn't even",
            data->size(), name);
    return 0;
  }
  u32 num_rows = data->size() / 2;

  Relation* h_rel = (Relation*)malloc(sizeof(Relation));
  Relation* d_rel;
  checkCuda(cudaMalloc((void**)&d_rel, sizeof(Relation)));

  char* d_name;
  checkCuda(cudaMalloc((void**)&d_name, sizeof(strlen(name) + 1)));
  checkCuda(cudaMemcpy(d_name, name, strlen(name) + 1, cudaMemcpyHostToDevice));

  u32* d_data_arr;
  checkCuda(cudaMalloc((void**)&d_data_arr, sizeof(u32) * data->size()));
  checkCuda(cudaMemcpy(d_data_arr, data->data(), sizeof(u32) * data->size(),
                       cudaMemcpyHostToDevice));

  u32* d_sorted_array;
  checkCuda(cudaMalloc((void**)&d_sorted_array, sizeof(u32) * num_rows));

  KeyValue* d_map = create_hashtable();

  h_rel->name = d_name;
  h_rel->num_rows = num_rows;
  h_rel->sorted_arr = d_sorted_array;
  h_rel->map = d_map;
  h_rel->data_arr = d_data_arr;
  checkCuda(
      cudaMemcpy(d_rel, &h_rel, sizeof(Relation), cudaMemcpyHostToDevice));

  // make sorted array
  int numBlocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  initSortedArr<<<numBlocks, BLOCK_SIZE>>>(d_rel);
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());
  // it has to be sorted if the data_array is not sorted

  // make the map
  initMap<<<numBlocks, BLOCK_SIZE>>>(d_rel);
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());
  return d_rel;
}

int main(int argc, char* argv[]) {
  u32 edgeSize = 5;

  char path_name[] = "path";
  std::vector<u32>* path_data = linearGraph(edgeSize);
  Relation* graph_path = makeRelation(path_data, path_name);
  printRelation(graph_path);

  char edge_name[] = "edge";
  std::vector<u32>* edge_data = linearGraph(edgeSize);
  std::reverse(edge_data->begin(), edge_data->end());
  Relation* graph_edge = makeRelation(edge_data, edge_name);
  printRelation(graph_edge);

  return 0;
}