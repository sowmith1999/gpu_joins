default:
	g++ -std=c++11 -o cpu_join src/cpu_join.cpp
clang:
	clang++ -std=c++11 -o cpu_join src/cpu_join.cpp
k-ary:
	nvcc k-ary-joins/cuda_based/binary_join.cu -o k_ary_binary 
