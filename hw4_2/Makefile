# 
# Standard all target
all: hw4

# HW 4
hw4: hw4.o
	g++ -lpthread -std=c++11 -o hw4 hw4.o

hw4.o:	hw4.cpp hw4.hpp
	g++ -lpthread -std=c++11 -m64 -O3 -ffast-math -funroll-loops -march=native -mtune=native -DNDEBUG -DNBOOST_DEBUG -Wall -c hw4.cpp

clean:
	rm -f *.o hw4