#pragma once
#include <cuda_runtime.h>
#include <vector>

struct Graph {
    int* srcNodes; 
    int* destNodes; 
    int numEdges;

    __host__ Graph(int numEdges) : numEdges(numEdges) {
        cudaMalloc(&srcNodes, numEdges * sizeof(int));
        cudaMalloc(&destNodes, numEdges * sizeof(int));
    }

    __host__ ~Graph() {
        cudaFree(srcNodes);
        cudaFree(destNodes);
    }
};

__host__ void fillGraphData(Graph& graph, const std::vector<int>& srcNodes, const std::vector<int>& destNodes) {
    if (srcNodes.size() != graph.numEdges || destNodes.size() != graph.numEdges) {
        return;
    }

    cudaMemcpy(graph.srcNodes, srcNodes.data(), graph.numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph.destNodes, destNodes.data(), graph.numEdges * sizeof(int), cudaMemcpyHostToDevice);
}
