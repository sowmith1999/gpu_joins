#include <vector>
#include <iostream>
#include <chrono>

#include "graph.cuh"
// #include "graph.h.gch"

struct Graph {
    std::vector<int> srcNodes;
    std::vector<int> destNodes;
    Graph () {}
    Graph(const std::vector<int>& src, const std::vector<int>& dest) : srcNodes(src), destNodes(dest) {}
};

std::vector<Graph> intersectGraphsCPU(const std::vector<Graph>& graphs) {
    Graph result;
    for (int i = 0; i < graphs[0].srcNodes.size(); ++i) {
        bool intersects = true;
        for (int j = 1; j < graphs.size() && intersects; ++j) {
            if (graphs[0].srcNodes[i] != graphs[j].srcNodes[i] ||
                graphs[0].destNodes[i] != graphs[j].destNodes[i]) {
                intersects = false;
            }
        }
        if (intersects) {
            result.srcNodes.push_back(graphs[0].srcNodes[i]);
            result.destNodes.push_back(graphs[0].destNodes[i]);
        }
    }
    return std::vector<Graph>{result};
}

int main() {
    std::vector<Graph> graphs;
    for (size_t i = 0; i < srcNodesVec.size(); ++i) {
        graphs.push_back(Graph(srcNodesVec[i], destNodesVec[i]));
    }
    auto start = std::chrono::steady_clock::now();

    std::vector<Graph> intersectedGraphs = intersectGraphsCPU(graphs);

    auto end = std::chrono::steady_clock::now();

    std::cout << "Intersected Graph Edges:" << std::endl;
    for (size_t i = 0; i < intersectedGraphs[0].srcNodes.size(); ++i) {
        std::cout << "(" << intersectedGraphs[0].srcNodes[i] << ", " << intersectedGraphs[0].destNodes[i] << ")" << std::endl;
    }

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "CPU Time (nanoseconds): " << elapsed.count() << " ms" << std::endl;

    return 0;
}
