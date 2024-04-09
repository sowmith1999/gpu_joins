// Authors: Michael G & Sowmith K
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "hisa.cuh"
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

struct Row {
    int key;
    float value;
};


__global__ void binaryJoinKernel(const Row* table1, size_t table1Size,
                                 const Row* table2, size_t table2Size,
                                 Row* resultTable, size_t* resultSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= table1Size) return;

    for (size_t j = 0; j < table2Size; ++j) {
        if (table1[idx].key == table2[j].key) {
            int resultIdx = atomicAdd(
                reinterpret_cast<unsigned long long*>(resultSize), 1ULL);
            resultTable[resultIdx] = {table1[idx].key,
                                      table1[idx].value + table2[j].value};
            break;
        }
    }
}

__global__ void intersectGraphsKernel(Graph* graphs, int numGraphs, int* outputBuffer, int numEdges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numEdges) {
      return; 
    }

    bool intersects = true;
    for (int i = 1; i < numGraphs && intersects; i++) {
        intersects &= (graphs[0].srcNodes[idx] == graphs[i].srcNodes[idx]) && 
                       (graphs[0].destNodes[idx] == graphs[i].destNodes[idx]);
    }

    outputBuffer[idx] = intersects ? 1 : 0;
}

__global__ void mergeGraphsKernel(Graph* graphs, int numGraphs, Graph* outputGraph) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= graphs[0].numEdges) {
        return;
    }

    bool intersects = true;
    for (int i = 1; i < numGraphs && intersects; i++) {
        intersects &= (graphs[0].srcNodes[idx] == graphs[i].srcNodes[idx]) && 
                       (graphs[0].destNodes[idx] == graphs[i].destNodes[idx]);
    }

    if (intersects) {
        outputGraph->srcNodes[idx] = graphs[0].srcNodes[idx];
        outputGraph->destNodes[idx] = graphs[0].destNodes[idx];
    }
}