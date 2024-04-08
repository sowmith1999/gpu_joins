#include "graph_utilities.cuh"
#include "gpu_join_kernels.cuh"
#include <iostream>
#include <vector>

int main() {
    // These are graphs where we have source nodes and dest nodes. So source(1) and dest(2) = edge(1, 2)
    // in this current setup, there is edge(1, 2) in the first index. 
    std::vector<std::vector<int>> srcNodesVec = {
        {1, 2, 4}, 
        {1, 11, 2}, 
        {1, 10, 3}  
    };

    std::vector<std::vector<int>> destNodesVec = {
        {2, 3, 2}, 
        {2, 3, 2}, 
        {2, 3, 4}  
    };

    int numEdges = srcNodesVec[0].size();

    Graph* graphsOnGPU;
    cudaMallocManaged(&graphsOnGPU, 3 * sizeof(Graph)); 

    for (int i = 0; i < 3; ++i) {
        new (&graphsOnGPU[i]) Graph(numEdges); 
        fillGraphData(graphsOnGPU[i], srcNodesVec[i], destNodesVec[i]);
    }

    int* outputBuffer;
    cudaMalloc(&outputBuffer, numEdges * sizeof(int));

    int blockSize = 256; 
    int numBlocks = (numEdges + blockSize - 1) / blockSize;

    intersectGraphsKernel<<<numBlocks, blockSize>>>(graphsOnGPU, 3, outputBuffer, numEdges);
    cudaDeviceSynchronize();

    std::vector<int> intersectionResults(numEdges);
    cudaMemcpy(intersectionResults.data(), outputBuffer, numEdges * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Intersection Results:" << std::endl;
    for (int i = 0; i < numEdges; ++i) {
        std::cout << intersectionResults[i] << (i < numEdges - 1 ? ", " : "\n");
    }

    // Cleanup
    for (int i = 0; i < 3; ++i) {
        cudaFree(graphsOnGPU[i].srcNodes);
        cudaFree(graphsOnGPU[i].destNodes);
    }
    cudaFree(graphsOnGPU);
    cudaFree(outputBuffer);

    return 0;
}