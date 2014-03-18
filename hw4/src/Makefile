# set this to what I want
OTHER_FLAGS=$(ENV_HPC_OPTS)
OPT_FLAGS=$()

# Standard all target
all: hw4

# HW 4
hw4: hw4.o MatrixMultiply.o
	g++ -o hw4 -lpthread hw4.o MatrixMultiply.o

hw4.o:	hw4.cpp 
	g++ ${OTHER_FLAGS} -lpthread -m64 -O3 -ffast-math -funroll-loops -march=native -mtune=native -DNDEBUG -DNBOOST_DEBUG -Wall -c hw4.cpp

MatrixMultiply.o : MatrixMultiply.hpp MatrixMultiply.cpp
	g++ ${OTHER_FLAGS} -lpthread -m64 -O3 -ffast-math -funroll-loops -march=native -mtune=native -DNDEBUG -DNBOOST_DEBUG -Wall -c MatrixMultiply.cpp

clean:
	rm -f *.o hw4