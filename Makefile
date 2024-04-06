default:
	clang++ -g -std=c++11 -o cpu_join src/cpu_join.cpp
val:
	make default && valgrind --leak-check=full ./cpu_join
