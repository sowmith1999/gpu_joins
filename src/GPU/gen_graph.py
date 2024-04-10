import numpy as np
import sys

num_edges = int(sys.argv[1])  # Number of edges in each graph
max_node_value = 5000  # Maximum node value
np.random.seed(42)  

header = "#pragma once\n#include <vector>\n"
src_nodes_cpp = header + "std::vector<std::vector<int>> srcNodesVec = {\n"
dest_nodes_cpp = "std::vector<std::vector<int>> destNodesVec = {\n"

for _ in range(3):  
    src_nodes = np.random.randint(1, max_node_value, num_edges)
    dest_nodes = np.random.randint(1, max_node_value, num_edges)
    
    src_nodes_cpp += "    { " + ", ".join(map(str, src_nodes.tolist())) + " },\n"
    dest_nodes_cpp += "    { " + ", ".join(map(str, dest_nodes.tolist())) + " },\n"

src_nodes_cpp += "};\n"
dest_nodes_cpp += "};\n"

cpp_code = src_nodes_cpp + "\n" + dest_nodes_cpp

file_path = "graph.cuh"
with open(file_path, "w") as file:
    file.write(cpp_code)
