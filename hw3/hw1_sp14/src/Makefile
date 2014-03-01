# set this to what I want
OTHER_FLAGS=$(ENV_HPC_OPTS)

# Standard all target
all: hw1

# HW 1
hw1: hw1.o MatrixMultiply.o
	g++ -fopenmp -o hw1 hw1.o MatrixMultiply.o

hw1.o:	hw1.cpp 
	g++ ${OTHER_FLAGS} -m64 -O3 -ffast-math -funroll-loops -march=native -mtune=native -DNDEBUG -DNBOOST_DEBUG -Wall -fopenmp -c hw1.cpp

MatrixMultiply.o : MatrixMultiply.hpp MatrixMultiply.cpp
	g++ ${OTHER_FLAGS} -m64 -O3 -ffast-math -funroll-loops -march=native -mtune=native -DNDEBUG -DNBOOST_DEBUG -fopenmp -c MatrixMultiply.cpp

clean:
	rm -f *.o hw1


