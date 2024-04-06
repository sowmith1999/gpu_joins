default:
	clang++ -g -std=c++11 -o cpu_join src/cpu_join.cpp
val:
	make default && valgrind --leak-check=full ./cpu_join
address:
	clang++ -g -std=c++11 -fsanitize=address -o cpu_join src/cpu_join.cpp
memory:
	clang++ -g -std=c++11 -fsanitize=memory -o cpu_join src/cpu_join.cpp
