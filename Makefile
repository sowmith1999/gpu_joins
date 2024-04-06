default:
<<<<<<< HEAD
	clang++ -g -std=c++11 -o cpu_join src/cpu_join.cpp
val:
	make default && valgrind --leak-check=full ./cpu_join
>>>>>>> c23e7ef9389d2084f89f497f8f30a96a0927bd61
