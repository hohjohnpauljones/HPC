OTHER_FLAGS= -std=c++0x

all: hw2

hw2: hw2.o
	g++ -o hw2 hw2.o -lpthread

hw2.o: hw2.cpp csvparse.o hw2.hpp
	g++ ${OTHER_FLAGS} -c hw2.cpp csvparse.o -pthread

csvparse.o: csvparse.cpp csvparse.hpp
	g++ ${OTHER_FLAGS} -c csvparse.cpp

clean:
	rm -f *.o hw2;
