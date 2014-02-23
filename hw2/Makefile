OTHER_FLAGS= -std=c++0x

all: hw2

hw2: hw2.o
	g++ -o hw2 hw2.o

hw2.o: hw2.cpp 
	g++ ${OTHER_FLAGS} -c hw2.cpp

clean:
	rm -f *.o hw2;
