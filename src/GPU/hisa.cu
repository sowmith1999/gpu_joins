#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>


/*
This is a real simple implementation of a hash table on GPU
Only  supports 32 bit unsigned integers
https://github.com/nosferalatu/SimpleGPUHashTable
doesn't support delete operations
*/
// hashtable capacity has to be power of Two to avoid using,
// modulo operation and replace it with BitWise And with number - 1, here
// number has to be power of 2, and that number is hashtable capacity for us
// 32 bit murmur hash

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

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__device__ void insertKey(KeyValue *hashtable, uint32_t key, uint32_t value) {
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

__global__ void insertMulKeys(KeyValue *hashtable, const KeyValue *kvs, uint32_t numkvs) {
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

__device__ uint32_t lookupKey(KeyValue *hashtable, uint32_t key) {
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

__global__ void lookupMulKeys(KeyValue *hashtable, KeyValue *kvs, unsigned int numkvs) {
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

KeyValue *create_hashtable() {
    KeyValue *hashtable;
    cudaMalloc(&hashtable, sizeof(KeyValue) * kHashTableCapacity);
    checkCUDAError("HashTable initialization malloc");

    static_assert(kEmpty == 0xffffffff, "kEmpty has to be 0xffffffff");
    cudaMemset(hashtable, 0xff, sizeof(KeyValue) * kHashTableCapacity);
    return hashtable;
}

std::vector<KeyValue> generate_random_KVs(std::mt19937 &rnd, uint32_t numkvs) {
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
//    std::random_device rd;
//    uint32_t seed = rd();
//    uint32_t numkvs = 16;
//    std::mt19937 rnd(seed); // this is passed around as a reference, so that sequences don't repeat.
//    // create a random set of KeyValue elements to test for insertion, deletion and resizing
//    std::vector<KeyValue> kvs = generate_random_KVs(rnd, numkvs);
//    KeyValue *hashTable = create_hashtable();
//    KeyValue *device_kvs;
//    cudaMalloc((void**)&device_kvs, sizeof(KeyValue) * numkvs);
//    cudaMemcpy(device_kvs, kvs.data(), (sizeof(KeyValue) * numkvs), cudaMemcpyHostToDevice);
//    insertMulKeys<<<1, 32>>>(hashTable, device_kvs, (uint32_t)numkvs);
//    for(auto kv: kvs){
//        printf("Key is: %u, Value is %u\n", kv.key, kv.value);
//    }
//    cudaDeviceSynchronize();
//    cudaFree(device_kvs);
//    cudaFree(hashTable);
typedef struct Index {
    int* sorted_arr;
    KeyValue* map = nullptr;
} Index;

typedef struct Relation {
    char* name = nullptr;
    int num_rows;
    int num_cols;
    int index_col;
    Index index;
    int* data_arr;
} Relation;

int main() {

}
