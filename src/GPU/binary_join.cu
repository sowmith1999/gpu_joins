// Authors: Michael G & Sowmith K
#include <chrono>
#include <cstdio>
#include <iostream>

#include "kernels.cuh"
#include "exception.cuh"

void binaryJoin(const Row* table1, size_t table1Size, const Row* table2,
                size_t table2Size) {
    Row* d_table1;
    Row* d_table2;
    Row* d_resultTable;
    size_t* d_resultSize;

    cudaMalloc(&d_table1, table1Size * sizeof(Row));
    cudaMalloc(&d_table2, table2Size * sizeof(Row));
    cudaMalloc(&d_resultTable, (table1Size + table2Size) * sizeof(Row));
    cudaMalloc(&d_resultSize, sizeof(size_t));

    size_t initialResultSize = 0;
    cudaMemcpy(d_resultSize, &initialResultSize, sizeof(size_t),
               cudaMemcpyHostToDevice);

    checkCuda(cudaMemcpy(d_table1, table1, table1Size * sizeof(Row),
               cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_table2, table2, table2Size * sizeof(Row),
               cudaMemcpyHostToDevice));

    int blockSize = 1;
    int numBlocks = (table1Size + blockSize - 1) / blockSize;

    binaryJoinKernel<<<numBlocks, blockSize>>>(d_table1, table1Size, d_table2,
                                               table2Size, d_resultTable,
                                               d_resultSize);

    size_t resultSize;
    cudaMemcpy(&resultSize, d_resultSize, sizeof(size_t),
               cudaMemcpyDeviceToHost);

    Row* resultTable = new Row[resultSize];
    cudaMemcpy(resultTable, d_resultTable, resultSize * sizeof(Row),
               cudaMemcpyDeviceToHost);

    cudaFree(d_table1);
    cudaFree(d_table2);
    cudaFree(d_resultTable);
    cudaFree(d_resultSize);

    std::cout << "Number of joined rows: " << resultSize << std::endl;
    for (size_t i = 0; i < resultSize; ++i) {
        std::cout << resultTable[i].key << " " << resultTable[i].value
                  << std::endl;
    }
    delete[] resultTable;
}

void binaryJoinWithUnifiedMemory(const Row* h_table1, size_t table1Size,
                                 const Row* h_table2, size_t table2Size) {
    Row *table1, *table2, *resultTable;
    size_t* resultSize;

    cudaMallocManaged(&table1, table1Size * sizeof(Row));
    cudaMallocManaged(&table2, table2Size * sizeof(Row));
    cudaMallocManaged(&resultTable, (table1Size + table2Size) * sizeof(Row));
    cudaMallocManaged(&resultSize, sizeof(size_t));

    *resultSize = 0;

    memcpy(table1, h_table1, table1Size * sizeof(Row));
    memcpy(table2, h_table2, table2Size * sizeof(Row));

    int blockSize = 1;
    int numBlocks = (table1Size + blockSize - 1) / blockSize;

    binaryJoinKernel<<<numBlocks, blockSize>>>(
        table1, table1Size, table2, table2Size, resultTable, resultSize);

    cudaDeviceSynchronize();

    std::cout << std::endl;
    std::cout << "Number of joined rows: " << *resultSize << std::endl;
    for (size_t i = 0; i < *resultSize; ++i) {
        std::cout << resultTable[i].key << " " << resultTable[i].value
                  << std::endl;
    }

    cudaFree(table1);
    cudaFree(table2);
    cudaFree(resultTable);
    cudaFree(resultSize);
}

int main() {
    Row table1[] = {{1, 1.5f},   {2, 2.5f},   {3, 3.5f},   {4, 4.5f},
                    {5, 5.5f},   {6, 6.5f},   {7, 7.5f},   {8, 8.5f},
                    {9, 9.5f},   {10, 10.5f}, {11, 11.5f}, {12, 12.5f},
                    {13, 13.5f}, {14, 14.5f}, {15, 15.5f}, {16, 16.5f},
                    {17, 17.5f}, {18, 18.5f}, {19, 19.5f}, {20, 20.5f}};

    Row table2[] = {{5, 0.5f},  {6, 1.0f},  {7, 1.5f},  {8, 2.0f},
                    {9, 2.5f},  {10, 3.0f}, {11, 3.5f}, {12, 4.0f},
                    {13, 4.5f}, {14, 5.0f}, {15, 5.5f}, {16, 6.0f},
                    {17, 6.5f}, {18, 7.0f}, {19, 7.5f}, {20, 8.0f},
                    {21, 8.5f}, {22, 9.0f}, {23, 9.5f}, {24, 10.0f}};

    auto non_unified_start = std::chrono::steady_clock::now();

    binaryJoin(table1, sizeof(table1) / sizeof(Row), table2,
               sizeof(table2) / sizeof(Row));

    auto non_unified_end = std::chrono::steady_clock::now();
    auto non_unified_time = non_unified_end - non_unified_start;

    auto unified_start = std::chrono::steady_clock::now();

    binaryJoinWithUnifiedMemory(table1, sizeof(table1) / sizeof(Row), table2,
                                sizeof(table2) / sizeof(Row));

    auto unified_end = std::chrono::steady_clock::now();
    auto unified_time = unified_end - unified_start;

    std::cout << "\n" << std::endl;
    std::cout
        << "Non Unified Memory Time (nanoseconds): "
        << std::chrono::duration<double, std::nano>(non_unified_time).count()
        << " ns" << std::endl;
    std::cout
        << "Non Unified Memory Time (milliseconds): "
        << std::chrono::duration<double, std::milli>(non_unified_time).count()
        << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Unified Memory Time (nanoseconds): "
              << std::chrono::duration<double, std::nano>(unified_time).count()
              << " ns" << std::endl;
    std::cout << "Unified Memory Time (milliseconds): "
              << std::chrono::duration<double, std::milli>(unified_time).count()
              << " ms" << std::endl;

    return 0;
}