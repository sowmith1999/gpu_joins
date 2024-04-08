#pragma once

#include <chrono>
#include <cstdio>
#include <iostream>

#include "exception.cuh"
#include "graph_utilities.cuh"

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

/*
std::vector<int> joinRelationsCpu(const std::vector<Relation*>& relations) {
  if (relations.empty())
    return {};

  std::vector<int> result;

  for (int idx : relations[0]->index.sorted_arr) {
    bool matchFound = true;
    std::vector<int> combinedRow;

    std::vector<int> keys;
    for (int col : relations[0]->index_cols) {
      keys.push_back(relations[0]->data_arr[idx + col]);
    }

    for (size_t relIdx = 1; relIdx < relations.size(); ++relIdx) {
      auto& rel = relations[relIdx];
      auto it = rel->index.map.find(keys);
      if (it == rel->index.map.end()) {
        matchFound = false;
        break;
      }

      int matchingRowStart = rel->index.sorted_arr[it->second];
      for (int i = 0; i < rel->num_cols; ++i) {
        if (std::find(rel->index_cols.begin(), rel->index_cols.end(), i) ==
                rel->index_cols.end() ||
            relIdx == 1) {
          combinedRow.push_back(rel->data_arr[matchingRowStart + i]);
        }
      }
    }

    if (matchFound) {
      result.insert(result.end(), combinedRow.begin(), combinedRow.end());
    }
  }

  return result;
}

*/