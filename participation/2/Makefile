all: vectorSquare vectorAdd vector4Add imageProcess

LD_OCL= -lOpenCL
LD_SCOTTGS= -lscottgs
LD_BOOST=

# Use pkg-config to determine the CFLAGS and linker flags
LD_OPENCV=$(shell pkg-config --libs opencv)
CFLAGS_OPENCV=$(shell pkg-config --cflags opencv)

OCL_INCLUDE_PATH=-I /opt/cuda/include/

# Example 1, per-element square of a vector
vectorSquare: vectorSquare.o oclutil.o
	g++ -o $@ $^ ${LD_OCL} ${LD_SCOTTGS}

vectorSquare.o: vectorSquare.cpp
	g++ -O2 -c $^ ${OCL_INCLUDE_PATH}

# Example 2, per-element sum of two vectors
vectorAdd: vectorAdd.o oclutil.o
	g++ -o $@ $^ ${LD_OCL} ${LD_SCOTTGS}

vectorAdd.o: vectorAdd.cpp
	g++ -O2 -c $^ ${OCL_INCLUDE_PATH}

# Example 3, per-element sum of two vectors
vector4Add: vector4Add.o oclutil.o
	g++ -o $@ $^ ${LD_OCL} ${LD_SCOTTGS}

vector4Add.o: vector4Add.cpp
	g++ -O2 -c $^ ${OCL_INCLUDE_PATH}

# Example 4, Image Process / Filter an image
imageProcess: imageProcessing.o oclutil.o kernel_selections.o
	g++ -o $@ $^ ${LD_OCL} ${LD_SCOTTGS} ${LD_OPENCV}

imageProcessing.o: imageProcessing.cpp
	g++ -O2 -c $^ ${OCL_INCLUDE_PATH} ${CFLAGS_OPENCV}



# Utility code from ClusterChimps.org's OCL Tools Package
oclutil.o: oclutil.cpp
	g++ -O2 -c $^ ${OCL_INCLUDE_PATH} 

kernel_selections.o: kernel_selections.cpp
	g++ -O2 -c $^ ${OCL_INCLUDE_PATH} 

clean:
	rm *.o vectorSquare vectorAdd vector4Add imageProcess

# ================== TESTING COMMANDS =====================
test: test_vsquare test_vadd test_v4add

test_vsquare: vectorSquare
	./vectorSquare

test_vadd: vectorAdd
	./vectorAdd

test_v4add: vector4Add
	./vector4Add




