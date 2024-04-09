// Authors: Michael G & Sowmith K
#include "kernels.cuh"
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

    int blockSize = 1;
    int numBlocks = (numEdges + blockSize - 1) / blockSize;

    int* intersectionCount;
    cudaMalloc(&intersectionCount, sizeof(int));
    cudaMemset(intersectionCount, 0, sizeof(int));

    intersectGraphsKernel<<<numBlocks, blockSize>>>(graphsOnGPU, 3, outputBuffer, numEdges, intersectionCount);
    cudaDeviceSynchronize();

    int h_intersectionCount;
    cudaMemcpy(&h_intersectionCount, intersectionCount, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Total Intersections: " << h_intersectionCount << std::endl;
    std::vector<int> intersectionResults(numEdges);
    cudaMemcpy(intersectionResults.data(), outputBuffer, numEdges * sizeof(int), cudaMemcpyDeviceToHost);

    // Merging area
    Graph* outputGraph;
    cudaMallocManaged(&outputGraph, sizeof(Graph));
    cudaMallocManaged(&outputGraph->srcNodes, h_intersectionCount * sizeof(int));
    cudaMallocManaged(&outputGraph->destNodes, h_intersectionCount * sizeof(int));
    outputGraph->numEdges = h_intersectionCount;

    numBlocks = (outputGraph->numEdges + blockSize - 1) / blockSize;

    mergeGraphsKernel<<<numBlocks, blockSize>>>(graphsOnGPU, 3, outputGraph);
    cudaDeviceSynchronize();

    std::cout << "Output Graph: " << std::endl;
    for (int i = 0; i < outputGraph->numEdges; ++i) {
        // Ensure you are checking against a valid sentinel value or ensure edges are initialized correctly
        if (outputGraph->srcNodes[i] != -1 && outputGraph->destNodes[i] != -1) {
            std::cout << "(" << outputGraph->srcNodes[i] << ", " << outputGraph->destNodes[i] << ")" << std::endl;
        }
    }

//    int intersectionCount = /* result from counting intersections */;
//    cudaMallocManaged(&outputGraph->srcNodes, intersectionCount * sizeof(int));
//    cudaMallocManaged(&outputGraph->destNodes, intersectionCount * sizeof(int));

    // std::cout << "Intersection Results:" << std::endl;
    // for (int i = 0; i < numEdges; ++i) {
    //     std::cout << intersectionResults[i] << (i < numEdges - 1 ? ", " : "\n");
    // }

    // Cleanup
    cudaFree(graphsOnGPU->srcNodes);
    cudaFree(graphsOnGPU->destNodes);
    cudaFree(outputGraph->srcNodes);
    cudaFree(outputGraph->destNodes);
    cudaFree(outputGraph);
    cudaFree(graphsOnGPU);
    cudaFree(outputBuffer);
    cudaFree(intersectionCount);

    return 0;
}