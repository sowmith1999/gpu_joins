#include "../gdlog/include/relation.cuh"
#include "../gdlog/include/lie.cuh"
// #include "../gdlog/include/relational_algebra.cuh"
#include "../gdlog/include/tuple.cuh"
#include "../gdlog/include/timer.cuh"


__global__ void KaryJoinKernel(GHashRelContainer *table1, GHashRelContainer *table2, column_type *output, int *outputCounter) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < table1->index_map_size) {
        MEntity entry1 = table1->index_map[idx];
        MEntity entry2 = table2->index_map[idx]; 

        if(entry1.key == entry2.key && entry1.key != EMPTY_HASH_ENTRY) {
            int localOutputIndex = atomicAdd(outputCounter, 1);
            output[localOutputIndex] = entry1.key;
        }
    }
}

__global__ void GHashRelJoinKernel(GHashRelContainer *table1, GHashRelContainer *table2, int *output, int *matchCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < table1->index_map_size) {
        MEntity entry1 = table1->index_map[idx];
        
        for (size_t i = 0; i < table2->index_map_size; i++) {
            MEntity entry2 = table2->index_map[i];
            if (entry1.key == entry2.key) {
                int loc = atomicAdd(matchCount, 1);
                output[loc] = idx;
                break; 
            }
        }
    }
}

__global__ void k_aryJoinKernel(GHashRelContainer *tables, int numTables, tuple_size_t *matching_counts) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    bool isMatch = true;
    tuple_size_t local_count = 0;

    if (idx < tables[0].index_map_size) { // Iterate through all tuples of the first table
        MEntity entity1 = tables[0].index_map[idx];

        for (int tblIdx = 1; tblIdx < numTables; ++tblIdx) { // Compare with tuples in other tables
            bool foundMatch = false;
            for (int j = 0; j < tables[tblIdx].index_map_size; ++j) {
                if (entity1.key == tables[tblIdx].index_map[j].key) {
                    foundMatch = true;
                    break;
                }
            }
            if (!foundMatch) {
                isMatch = false;
                break;
            }
        }

        if (isMatch) {
            local_count = 1;
        }
    }

    atomicAdd(matching_counts, local_count);
}