#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <thrust/device_ptr.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <vector>

/********************************/
const uint32_t kHashTableCapacity = 32;
const uint32_t kEmpty = 0xffffffff;

__device__ uint32_t hash(uint32_t key) {
  key ^= key >> 16;
  key *= 0x85ebca6b;
  key ^= key >> 13;
  key *= 0xc2b2ae35;
  key ^= key >> 16;
  return key & (kHashTableCapacity - 1);
}

typedef struct KeyValue {
  uint32_t key;
  uint32_t value;
} KeyValue;

void checkCUDAError(const char* msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__device__ void insertKey(KeyValue* hashtable, uint32_t key, uint32_t value) {
  uint32_t slot = hash(key);
  while (true) {
    uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
    if (prev == kEmpty || prev == key) {
      hashtable[slot].value = value;
      return;
    }
    slot = (slot + 1) & (kHashTableCapacity - 1);
  }
}

__global__ void insertMulKeys(KeyValue* hashtable, const KeyValue* kvs,
                              uint32_t numkvs) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numkvs) {
    uint32_t key = kvs[tid].key;
    uint32_t value = kvs[tid].value;
    uint32_t slot = hash(key);
    while (true) {
      uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
      if (prev == kEmpty || prev == key) {
        hashtable[slot].value = value;
        printf("Inserted Key: %u, value: %u\n", key, value);
        return;
      }
      slot = (slot + 1) & (kHashTableCapacity - 1);
    }
  }
}

__device__ uint32_t lookupKey(KeyValue* hashtable, uint32_t key) {
  uint32_t slot = hash(key);
  while (true) {
    if (hashtable[slot].key == key) {
      return hashtable[slot].value;
    }
    if (hashtable[slot].key == kEmpty) {
      return kEmpty;
    }
    slot = (slot + 1) & (kHashTableCapacity + 1);
  }
}

__global__ void lookupMulKeys(KeyValue* hashtable, KeyValue* kvs,
                              unsigned int numkvs) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numkvs) {
    uint32_t key = kvs[tid].key;
    uint32_t slot = hash(key);
    while (true) {
      if (hashtable[slot].key == key) {
        kvs[tid].value = hashtable[slot].value;
        return;
      }
      if (hashtable[slot].key == kEmpty) {
        kvs[tid].value = kEmpty;
        return;
      }
      slot = (slot + 1) & (kHashTableCapacity + 1);
    }
  }
}

KeyValue* create_hashtable() {
  KeyValue* hashtable;
  cudaMalloc(&hashtable, sizeof(KeyValue) * kHashTableCapacity);
  checkCUDAError("HashTable initialization malloc");

  static_assert(kEmpty == 0xffffffff, "kEmpty has to be 0xffffffff");
  cudaMemset(hashtable, 0xff, sizeof(KeyValue) * kHashTableCapacity);
  return hashtable;
}

std::vector<KeyValue> generate_random_KVs(std::mt19937& rnd, uint32_t numkvs) {
  std::uniform_int_distribution<uint32_t> dis(0, kEmpty - 1);
  std::vector<KeyValue> kvs;
  kvs.reserve(numkvs);

  for (uint32_t i = 0; i < numkvs; i++) {
    uint32_t key = dis(rnd);
    uint32_t value = dis(rnd);
    kvs.push_back(KeyValue{key, value});
  }
  return kvs;
}

/********************************/
typedef struct Index {
  int* sorted_arr;
  KeyValue* map = nullptr;
} Index;

typedef struct Relation {
  char* name = nullptr;
  int num_rows;
  int num_cols;
  int num_indx_cols;
  int* index_col;
  Index index;
  int* data_arr;
} Relation;

struct TupleLessCol {
  int col;
  const int* data_arr;

  TupleLessCol(int col, const int* data_arr) : col(col), data_arr(data_arr) {}

  __device__ bool operator()(const int& offset1, const int& offset2) const {
    return data_arr[offset1 + col] < data_arr[offset2 + col];
  }
};

struct TupleEqual {
  const int* data_arr;

  TupleEqual(const int* data_arr) : data_arr(data_arr) {}

  __device__ bool operator()(const int& x, const int& y) const {
    return (data_arr[x] == data_arr[y] && data_arr[x + 1] == data_arr[y + 1]);
  }
};

struct TupleLess {
  const int* data_arr1;
  const int* data_arr2;

  TupleLess(const int* data_arr1, const int* data_arr2)
      : data_arr1(data_arr1), data_arr2(data_arr2) {}

  __device__ bool operator()(const int& offset1, const int& offset2) {
//    printf("1st ele:%d and 2nd ele: %d\n", data_arr1[offset1],
//           data_arr2[offset2]);
    if (data_arr1[offset1] == data_arr2[offset2])
      return data_arr1[offset1 + 1] > data_arr2[offset2 + 1];
    else
      return data_arr1[offset1] > data_arr2[offset2];
  }
};

void printArray(int* arr, int count) {
  for (int i = 0; i < count; i++) {
    printf("%d\t", arr[i]);
  }
  printf("\n");
}

void printDeviceArray(int* d_arr, int size) {
  int* h_arr = (int*)malloc(sizeof(int) * size);
  cudaMemcpy(h_arr, d_arr, sizeof(int) * size, cudaMemcpyDeviceToHost);
  checkCUDAError("Inside printDeviceArray after cudaMempy\n");
  for (int i = 0; i < size; i++) {
    printf("%d\t", h_arr[i]);
  }
  printf("\n");
  free(h_arr);
}

__global__ void d_printArray(int* arr, int size) {
  printf("printing array inside the device:\n");
  for (int i = 0; i < size; i++) {
    printf("%d\t", arr[i]);
  }
  printf("\n");
}
__global__ void initSortedArr(Relation* rel) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < rel->num_rows)
    rel->index.sorted_arr[idx] = idx * rel->num_cols;
}

__global__ void initMap(Relation* rel) {
  int cur_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int prev_idx = cur_idx - 1;
  if (cur_idx < rel->num_rows) {
    if ((rel->data_arr[rel->index.sorted_arr[cur_idx]] !=
         rel->data_arr[rel->index.sorted_arr[prev_idx]]) ||
        cur_idx == 0) {
      uint32_t key = rel->data_arr[rel->index.sorted_arr[cur_idx]];
      uint32_t value = cur_idx;
      insertKey(rel->index.map, key, value);
    }
  }
}

__global__ void testKernel(Relation* rel) {
  printf("The Relation is %s\n", rel->name);
  printf("\t rel.numrows = %d\n", rel->num_rows);
  printf("\t rel.numcols= %d\n", rel->num_cols);
  printf("\t rel.num_indx_cols= %d\n", rel->num_indx_cols);
  for (int i = 0; i < rel->num_indx_cols; i++) {
    printf("%d\t", rel->index_col[i]);
  }
  printf("\n");
  printf("The data array is: \n");
  for (int i = 0; i < rel->num_rows; i++) {
    printf("%d\t%d", rel->data_arr[rel->index.sorted_arr[i]],
           rel->data_arr[rel->index.sorted_arr[i] + 1]);
    printf("\n");
  }
  printf("The sorted arrray is:\n");
  for (int i = 0; i < rel->num_rows; i++) {
    printf("%d ", rel->index.sorted_arr[i]);
  }
  printf("\n");
  //  printf("The map is:\n");
  //  for (int i = 0; i < kHashTableCapacity; i++) {
  //    printf("%d\t%d\n", rel->index.map[i].key, rel->index.map[i].value);
  //  }
}

__global__ void joinRelationCount(Relation* outer, Relation* inner,
                                  int* count_arr) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  int count = 0;
  if (idx < outer->num_rows) {
    int key = outer->data_arr[outer->index.sorted_arr[idx]];
    int inner_idx = lookupKey(inner->index.map, key);
    if (inner_idx != kEmpty) {
      for (int inner_srtd_indx = inner_idx; inner_srtd_indx < inner->num_rows;
           inner_srtd_indx++) {
        int inner_key =
            inner->data_arr[inner->index.sorted_arr[inner_srtd_indx]];
        if (inner_key == key)
          count++;
        else
          break;
      }
    }
  }
  count_arr[idx] = count;
}

__global__ void joinRelationData(Relation* outer, Relation* inner,
                                 int* result_idx_arr, int* join_data_arr) {
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < outer->num_rows) {
    uint32_t result_idx = 2 * result_idx_arr[idx];
    int key = outer->data_arr[outer->index.sorted_arr[idx]];
    int outer_value = outer->data_arr[outer->index.sorted_arr[idx] + 1];
    int inner_idx = lookupKey(inner->index.map, key);
    if (inner_idx != kEmpty) {
      for (int inner_srtd_indx = inner_idx; inner_srtd_indx < inner->num_rows;
           inner_srtd_indx++) {
        int inner_key =
            inner->data_arr[inner->index.sorted_arr[inner_srtd_indx]];
        int inner_value =
            inner->data_arr[inner->index.sorted_arr[inner_srtd_indx] + 1];
        if (inner_key == key) {
          join_data_arr[result_idx] = inner_value;
          join_data_arr[result_idx + 1] = outer_value;
          //                    printf("Row inserted: Key:%d, Value:%d\n",
          //                    outer_value, inner_value);
          result_idx += 2;
        } else
          break;
      }
    }
  }
}

// Not Used, we are using set difference now
__global__ void makeDeltaData(Relation* new_rel, int* del_sorted_arr,
                              int del_num_rows, int* del_data_arr) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < del_num_rows) {
    int data_idx = del_sorted_arr[idx];
    del_data_arr[idx * 2] = new_rel->data_arr[data_idx];
    del_data_arr[idx * 2 + 1] = new_rel->data_arr[data_idx + 1];
  }
}

Relation* make_Relation(std::vector<int>* host_data_vec, int* d_data_inp,
                        int num_rows, int num_cols,
                        std::vector<int>* index_cols, char* name) {
  Relation* d_rel;
  Relation h_rel;
  cudaMalloc((void**)&d_rel, sizeof(Relation));

  char* d_name;
  cudaMalloc((void**)&d_name, strlen(name) + 1);
  cudaMemcpy(d_name, name, strlen(name) + 1, cudaMemcpyHostToDevice);

  int* d_data_arr;
  if (d_data_inp == nullptr) {
    cudaMalloc((void**)&d_data_arr, sizeof(int) * host_data_vec->size());
    cudaMemcpy(d_data_arr, host_data_vec->data(),
               host_data_vec->size() * sizeof(int), cudaMemcpyHostToDevice);
  } else
    d_data_arr = d_data_inp;

  int* d_index_cols;
  cudaMalloc((void**)&d_index_cols, sizeof(int) * index_cols->size());
  cudaMemcpy(d_index_cols, index_cols->data(), index_cols->size() * sizeof(int),
             cudaMemcpyHostToDevice);

  Index* d_index;
  cudaMalloc((void**)&d_index, sizeof(Index));

  int* d_sorted_array;
  cudaMalloc((void**)&d_sorted_array, sizeof(int) * num_rows);

  KeyValue* d_map = create_hashtable();

  h_rel.name = d_name;
  h_rel.num_rows = num_rows;
  h_rel.num_cols = num_cols;
  h_rel.num_indx_cols = index_cols->size();
  h_rel.index_col = d_index_cols;
  h_rel.index.map = d_map;
  h_rel.index.sorted_arr = d_sorted_array;
  h_rel.data_arr = d_data_arr;
  cudaMemcpy(d_rel, &h_rel, sizeof(Relation), cudaMemcpyHostToDevice);

  // make sorted array
  int blockSize = 256;
  int numBlocks = (num_rows + blockSize - 1) / blockSize;
  initSortedArr<<<numBlocks, blockSize>>>(d_rel);
  cudaDeviceSynchronize();
//  printf("the initialized sorted array, before sorting is:\n");
//  printDeviceArray(d_sorted_array, num_rows);
  // sort sorted array
  thrust::device_ptr<int> t_data_arr(d_data_arr);
  thrust::device_ptr<int> t_sorted_arr(d_sorted_array);

  for (int col = num_cols - 1; col >= 0; col--) {
    TupleLessCol comp(col, d_data_arr);
    thrust::stable_sort(thrust::device, t_sorted_arr, t_sorted_arr + num_rows,
                        comp);
  }

  // make the map
  initMap<<<numBlocks, blockSize>>>(d_rel);
  cudaDeviceSynchronize();
  return d_rel;
}

void removeDuplicates(Relation* rel) {
  Relation* h_rel = (Relation*)malloc(sizeof(Relation));
  cudaMemcpy(h_rel, rel, sizeof(Relation), cudaMemcpyDeviceToHost);

  thrust::device_ptr<int> t_new_srtd_arr(h_rel->index.sorted_arr);
  TupleEqual comp(h_rel->data_arr);
  auto new_end = thrust::unique(thrust::device, t_new_srtd_arr,
                                t_new_srtd_arr + h_rel->num_rows, comp);

  int new_size = new_end - t_new_srtd_arr;
  cudaMemcpy(&(rel->num_rows), &new_size, sizeof(int), cudaMemcpyHostToDevice);
}

Relation* joinRelations_host(Relation* outer, Relation* inner, int outer_rows) {
  int* h_count_arr = (int*)malloc(sizeof(int) * outer_rows);
  int* d_count_arr;
  cudaMalloc((void**)&d_count_arr, sizeof(int) * outer_rows);

  int blockSize = 256;
  int numBlocks = (outer_rows + blockSize - 1) / blockSize;
  joinRelationCount<<<numBlocks, blockSize>>>(outer, inner, d_count_arr);
  cudaDeviceSynchronize();

  thrust::device_ptr<int> t_count_arr(d_count_arr);
  thrust::exclusive_scan(t_count_arr, t_count_arr + outer_rows, t_count_arr);
  int totalJoinRowCount;
  cudaMemcpy(&totalJoinRowCount, d_count_arr + outer_rows - 1, sizeof(int),
             cudaMemcpyDeviceToHost);
  checkCUDAError("CudaMecpy after exclusive scan");
//  printf("Total number of Rows is: %d\n", totalJoinRowCount);

  int* d_join_data_arr;
  cudaMalloc((void**)&d_join_data_arr, sizeof(int) * 2 * totalJoinRowCount);
  joinRelationData<<<numBlocks, blockSize>>>(outer, inner, d_count_arr,
                                             d_join_data_arr);
  cudaDeviceSynchronize();

//  printDeviceArray(d_join_data_arr, totalJoinRowCount * 2);
  std::vector<int> index_cols{0};
  char path_new_name[] = "path_new";
  Relation* path_new =
      make_Relation(nullptr, d_join_data_arr, totalJoinRowCount, 2, &index_cols,
                    path_new_name);
  removeDuplicates(path_new);
  return path_new;
}

Relation* makeDelta(Relation* full_rel, Relation* new_rel) {
  Relation* h_new_rel = (Relation*)malloc(sizeof(Relation));
  cudaMemcpy(h_new_rel, new_rel, sizeof(Relation), cudaMemcpyDeviceToHost);
  Relation* h_full_rel = (Relation*)malloc(sizeof(Relation));
  cudaMemcpy(h_full_rel, full_rel, sizeof(Relation), cudaMemcpyDeviceToHost);

  int* d_del_srtd_arr;
  cudaMalloc((void**)&d_del_srtd_arr, sizeof(int) * h_new_rel->num_rows);

  thrust::device_ptr<int> t_new_rel_sorted(h_new_rel->index.sorted_arr);
  thrust::device_ptr<int> t_full_rel_sorted(h_full_rel->index.sorted_arr);
  thrust::device_ptr<int> t_del_srtd_arr(d_del_srtd_arr);
  TupleLess comp(h_new_rel->data_arr, h_full_rel->data_arr);
  auto del_end = thrust::set_difference(
      thrust::device, t_new_rel_sorted, t_new_rel_sorted + h_new_rel->num_rows,
      t_full_rel_sorted, t_full_rel_sorted + h_full_rel->num_rows,
      t_del_srtd_arr, comp);
  int delta_size = del_end - t_del_srtd_arr;
//  printf("the t_del_srtd_arr is:\n");
//  printDeviceArray(d_del_srtd_arr, delta_size);
  int* d_del_data_arr;
  cudaMalloc((void**)&d_del_data_arr, sizeof(int) * 2 * delta_size);
  int blockSize = 256;
  int numBlocks = (delta_size + blockSize - 1) / blockSize;
//  printf("the numBlocks: %d, blockSize: %d\n", numBlocks, blockSize);
  makeDeltaData<<<numBlocks, blockSize>>>(new_rel, d_del_srtd_arr, delta_size,
                                          d_del_data_arr);
  cudaDeviceSynchronize();
  checkCUDAError("Make Delta Data");
//  printf("the delta data is:\n");
//  printDeviceArray(d_del_data_arr, delta_size * 2);
  std::vector<int> index_cols{0};
  char del_name[] = "path_delta";
  Relation* path_delta =
      make_Relation(nullptr, d_del_data_arr, delta_size, h_new_rel->num_cols,
                    &index_cols, del_name);
  return path_delta;
}

Relation* updateFull(Relation* full_rel, Relation* del_rel) {
//  printf("Inside update full");
  Relation* h_del_rel = (Relation*)malloc(sizeof(Relation));
  cudaMemcpy(h_del_rel, del_rel, sizeof(Relation), cudaMemcpyDeviceToHost);
  checkCUDAError("After the del_rel copy into h_full_rel\n");
  Relation* h_full_rel = (Relation*)malloc(sizeof(Relation));
  cudaMemcpy(h_full_rel, full_rel, sizeof(Relation), cudaMemcpyDeviceToHost);
  checkCUDAError("After the full_rel copy into h_full_rel\n");
  // I append the data arrays together, that should'nt be a problem.
  // I merge the sorted arrays
  // And do a linear scan and create the map.
  int* d_full_rel_merge;
  cudaMalloc((void**)&d_full_rel_merge,
             sizeof(int) * 2 * (h_full_rel->num_rows + h_del_rel->num_rows));
  cudaMemcpy(d_full_rel_merge, h_full_rel->data_arr,
             sizeof(int) * 2 * h_full_rel->num_rows, cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_full_rel_merge + 2 * h_full_rel->num_rows, h_del_rel->data_arr,
             sizeof(int) * 2 * h_del_rel->num_rows, cudaMemcpyDeviceToDevice);

//  int total_size = (h_full_rel->num_rows + h_del_rel->num_rows) * 2;
//  printf("the d_full_rel_merge array is:\n");
//  printDeviceArray(d_full_rel_merge, total_size);
  // should be improved to not do all the work again
  std::vector<int> index_col{0};
  char path_full_name[] = "path_full";
  Relation* path_full = make_Relation(
      nullptr, d_full_rel_merge, (h_full_rel->num_rows + h_del_rel->num_rows),
      2, &index_col, path_full_name);
  return path_full;
}

int getRowCount(Relation* d_rel) {
  Relation* h_rel = (Relation*)malloc(sizeof(Relation));
  cudaMemcpy(h_rel, d_rel, sizeof(Relation), cudaMemcpyDeviceToHost);
  return h_rel->num_rows;
}

int main() {
  std::vector<int> index_cols{0};
  std::vector<int> graph_path{2, 1, 3, 2, 4, 2, 5, 3, 5, 4, 6, 5};
  std::vector<int> graph_edge{1, 2, 2, 3, 2, 4, 3, 5, 4, 5, 5, 6};
  int num_rows = 6;
  int num_cols = 2;

  char edge_name[] = "edge";
  Relation* d_edge = make_Relation(&graph_edge, nullptr, num_rows, num_cols,
                                   &index_cols, edge_name);
  testKernel<<<1, 1>>>(d_edge);
  cudaDeviceSynchronize();

  char rel_name[] = "path";
  Relation* d_path = make_Relation(&graph_path, nullptr, num_rows, num_cols,
                                   &index_cols, rel_name);
  testKernel<<<1, 1>>>(d_path);
  cudaDeviceSynchronize();

  Relation *d_path_new, *d_path_delta, *d_path_full;
  d_path_delta = d_path;
  d_path_full = d_path;
  int count = -1;
  do {
    count++;
    printf("---------------- %d Iteration --------------------\n",count);
    d_path_new =
        joinRelations_host(d_path_delta, d_edge, getRowCount(d_path_delta));
    testKernel<<<1, 1>>>(d_path_new);
    cudaDeviceSynchronize();

    d_path_delta = makeDelta(d_path_full, d_path_new);
    testKernel<<<1, 1>>>(d_path_delta);
    cudaDeviceSynchronize();

    d_path_full = updateFull(d_path_full, d_path_delta);
    testKernel<<<1, 1>>>(d_path_full);
    cudaDeviceSynchronize();
  } while (getRowCount(d_path_delta) != 0);
  return 0;
}