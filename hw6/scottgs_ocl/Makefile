all: imageProcess

LD_OCL= -lOpenCL
LD_SCOTTGS= -lscottgs
LD_BOOST=

# Use pkg-config to determine the CFLAGS and linker flags
LD_OPENCV=$(shell pkg-config --libs opencv)
CFLAGS_OPENCV=$(shell pkg-config --cflags opencv)

OCL_INCLUDE_PATH=-I /opt/cuda/include/

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
	rm *.o imageProcess




