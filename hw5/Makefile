CC=mpic++
CFLAGS=-c -fopenmp -m64 -std=c++0x -O3 -ffast-math -funroll-loops -march=native -mtune=native -DNDEBUG -DNBOOST_DEBUG -Wall 
LFLAGS=-o hw5 -fopenmp -lboost_filesystem-mt -lboost_system

all: hw5

hw5: hw5.o
	$(CC) $(LFLAGS) hw5.o
	
hw5.o: hw5.cpp hw5.hpp directory_scanner.cpp
	$(CC) $(CFLAGS) hw5.cpp

producer.o: producer.cpp hw5.hpp
	$(CC) $(CFLAGS) producer.cpp

consumer.o: consumer.cpp hw5.hpp
	$(CC) $(CFLAGS) consumer.cpp
	
run: hw5
	./runtests.sh

clean:
	rm *.o hw5
	
cleanres:
	rm -r MABCB7_* *.csv

analyze: run
	./analyze
