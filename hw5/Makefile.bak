CC=g++
CFLAGS=-c -std=c++11 -m64 -O3 -ffast-math -funroll-loops -march=native -mtune=native -DNDEBUG -DNBOOST_DEBUG -Wall
LFLAGS=-o hw5

all: hw5

hw5: hw5.o producer.o consumer.o
	$(CC) $(LFLAGS) hw5.o producer.o consumer.o
	
hw5.o: hw5.cpp hw5.hpp
	$(CC) $(CFLAGS) hw5.cpp

producer.o: producer.cpp hw5.hpp
	$(CC) $(CFLAGS) producer.cpp

consumer.o: consumer.cpp hw5.hpp
	$(CC) $(CFLAGS) consumer.cpp

clean:
	rm *.o hw5