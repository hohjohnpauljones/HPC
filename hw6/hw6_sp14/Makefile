# flags to pass compiler

# Standard all target
all: myProc

# HW 6

myProc: HW6.o
	nvcc -o hw6 HW6.o
HW6.o: HW6.cu
	nvcc -c HW6.cu -arch=sm_20
run: hw6
	./hw6 7 lena.pgm output.pgm

rerun: hw6
	./hw6 3 output.pgm output2.pgm

clean:
	rm -f *.o hw6
