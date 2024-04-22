// Authors: Michael G & Sowmith K
#include <iostream>
#include <chrono>
// #include <vector> // should come from graph.cuh
#include "kernels.cuh"
#include "graph.cuh"
// #include "graph.h.gch"

int main() {
    // These are graphs where we have source nodes and dest nodes. So source(1) and dest(2) = edge(1, 2)
    // in this current setup, there is edge(1, 2) in the first index.
    // std::vector<std::vector<int>> srcNodesVec = {
    //         {1, 2, 4},
    //         {1, 11, 2},
    //         {1, 10, 3}
    // };

    // std::vector<std::vector<int>> destNodesVec = {
    //         {2, 3, 2},
    //         {2, 3, 2},
    //         {2, 3, 4}
    // };

    int numEdges = srcNodesVec[0].size();

    Graph* graphsOnGPU;
    cudaMallocManaged(&graphsOnGPU, 3 * sizeof(Graph));

    for (int i = 0; i < 3; ++i) {
        new (&graphsOnGPU[i]) Graph(numEdges);
        fillGraphData(graphsOnGPU[i], srcNodesVec[i], destNodesVec[i]);
    }

    Graph* outputGraph;
    cudaMallocManaged(&outputGraph, sizeof(Graph));
    cudaMallocManaged(&outputGraph->srcNodes, numEdges * sizeof(int));
    cudaMallocManaged(&outputGraph->destNodes, numEdges * sizeof(int));
    outputGraph->numEdges = numEdges;

    int blockSize = 1;
    int numBlocks = (numEdges + blockSize - 1) / blockSize;

    int* offsetter;
    cudaMalloc(&offsetter, sizeof(int));
    cudaMemset(offsetter, 0, sizeof(int));

    auto combinedStart = std::chrono::steady_clock::now();

    intersectAndMergeGraphsKernel<<<numBlocks, blockSize>>>(graphsOnGPU, 3, outputGraph, numEdges, offsetter);
    cudaDeviceSynchronize();

    auto combinedEnd = std::chrono::steady_clock::now();

    std::cout << "Output Graph: " << std::endl;
    for (int i = 0; i < outputGraph->numEdges; ++i) {
        if (outputGraph->srcNodes[i] != -1 && outputGraph->destNodes[i] != -1) {
            std::cout << "(" << outputGraph->srcNodes[i] << ", " << outputGraph->destNodes[i] << ")" << std::endl;
        }
    }

    std::cout << "Time (nanoseconds): " << std::chrono::duration<double, std::milli>(combinedEnd - combinedStart).count() << " ms" << std::endl;

    // Cleanup
    cudaFree(graphsOnGPU->srcNodes);
    cudaFree(graphsOnGPU->destNodes);
    cudaFree(outputGraph->srcNodes);
    cudaFree(outputGraph->destNodes);
    cudaFree(outputGraph);
    cudaFree(graphsOnGPU);
    cudaFree(offsetter);

    return 0;
}