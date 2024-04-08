default:
	clang++ -g -std=c++11 -o cpu_instc src/cpu_intsc.cpp
val:
	make default && valgrind --leak-check=full ./cpu_instc
address:
	clang++ -g -std=c++11 -fsanitize=address -o cpu_instc src/cpu_instc.cpp
memory:
	clang++ -g -std=c++11 -fsanitize=memory -o cpu_instc src/cpu_instc.cpp
