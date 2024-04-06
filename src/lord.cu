#include "lord.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

void copyGHashRelContainerToDevice(GHashRelContainer* &d_container, const GHashRelContainer &h_container) {
    cudaMalloc(&d_container, sizeof(GHashRelContainer));

    MEntity* d_index_map = nullptr;
    cudaMalloc(&d_index_map, sizeof(MEntity) * h_container.index_map_size);
    cudaMemcpy(d_index_map, h_container.index_map, sizeof(MEntity) * h_container.index_map_size, cudaMemcpyHostToDevice);

    GHashRelContainer temp_container = h_container;
    temp_container.index_map = d_index_map;

    cudaMemcpy(d_container, &temp_container, sizeof(GHashRelContainer), cudaMemcpyHostToDevice);
}

int main() {
    const int numTables = 2; // Example number of tables
    std::vector<GHashRelContainer*> h_tables(numTables);
    std::vector<GHashRelContainer*> d_tables(numTables); // Pointers to device tables

    for(int i = 0; i < numTables; ++i) {
        cudaMalloc(&d_tables[i], sizeof(GHashRelContainer));
        cudaMalloc(&d_tables[i]->index_map, sizeof(MEntity) * h_tables[i]->index_map_size);
        cudaMemcpy(d_tables[i]->index_map, h_tables[i]->index_map, sizeof(MEntity) * h_tables[i]->index_map_size, cudaMemcpyHostToDevice);
    }

    GHashRelContainer** d_tablesArray;
    cudaMalloc(&d_tablesArray, numTables * sizeof(GHashRelContainer*));
    cudaMemcpy(d_tablesArray, d_tables.data(), numTables * sizeof(GHashRelContainer*), cudaMemcpyHostToDevice);

    unsigned int* d_matchingCounts;
    cudaMalloc(&d_matchingCounts, sizeof(unsigned int));
    cudaMemset(d_matchingCounts, 0, sizeof(unsigned int));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (h_tables[0]->index_map_size + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < numTables; ++i) {
        GHashRelContainer* d_table = d_tables[i];
        k_aryJoinKernel<<<blocksPerGrid, threadsPerBlock>>>(d_table, 1, reinterpret_cast<tuple_size_t*>(d_matchingCounts));
    }


    unsigned int h_matchingCounts;
    cudaMemcpy(&h_matchingCounts, d_matchingCounts, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::cout << "Total matching tuples across all tables: " << h_matchingCounts << std::endl;

    cudaFree(d_matchingCounts);
    for(auto t : d_tables) {
        cudaFree(t);
    }
    cudaFree(d_tablesArray);

    return 0;
}

// void executeGHashRelJoinKernel(GHashRelContainer &table1, GHashRelContainer &table2) {
//     GHashRelContainer *d_table1, *d_table2;
//     int *d_output, *d_matchCount;
//     int matchCount = 0;
//     int *output = new int[table1.index_map_size]; // Maximum possible matches

//     cudaMalloc(&d_output, table1.index_map_size * sizeof(int));
//     cudaMalloc(&d_matchCount, sizeof(int));
//     cudaMemset(d_matchCount, 0, sizeof(int));

//     copyGHashRelContainerToDevice(d_table1, table1);
//     copyGHashRelContainerToDevice(d_table2, table2);

//     int threadsPerBlock = 256;
//     int blocksPerGrid = (table1.index_map_size + threadsPerBlock - 1) / threadsPerBlock;

//     GHashRelJoinKernel<<<blocksPerGrid, threadsPerBlock>>>(d_table1, d_table2, d_output, d_matchCount);

//     cudaMemcpy(output, d_output, table1.index_map_size * sizeof(int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(&matchCount, d_matchCount, sizeof(int), cudaMemcpyDeviceToHost);

//     std::cout << "Match count: " << matchCount << std::endl;
//     for(int i = 0; i < matchCount; ++i) {
//         std::cout << "Match found at index " << output[i] << " in Table 1\n";
//     }

//     cudaFree(d_table1->index_map);
//     cudaFree(d_table2->index_map);
//     cudaFree(d_table1);
//     cudaFree(d_table2);
//     cudaFree(d_output);
//     cudaFree(d_matchCount);
//     delete[] output;
// }

// int main() {
//     const size_t table1Size = 3, table2Size = 3;
//     MEntity table1Data[table1Size] = {{1, 0}, {2, 0}, {3, 0}};
//     MEntity table2Data[table2Size] = {{2, 0}, {3, 0}, {4, 0}};

//     GHashRelContainer table1(0, 0, 0), table2(0, 0, 0);
//     table1.index_map = table1Data;
//     table1.index_map_size = table1Size;
//     table2.index_map = table2Data;
//     table2.index_map_size = table2Size;

//     executeGHashRelJoinKernel(table1, table2);

//     return 0;
// }