all: vectorSquare vectorAdd vector4Add

LD_OCL= -lOpenCL
LD_SCOTTGS= -lscottgs
LD_BOOST=

OCL_INCLUDE_PATH=

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




# Utility code from ClusterChimps.org's OCL Tools Package
oclutil.o: oclutil.cpp
	g++ -O2 -c $^ ${OCL_INCLUDE_PATH}


clean:
	rm *.o vectorSquare vectorAdd vector4Add

# ================== TESTING COMMANDS =====================
test: test_vsquare test_vadd test_v4add

test_vsquare: vectorSquare
	./vectorSquare

test_vadd: vectorAdd
	./vectorAdd

test_v4add: vector4Add
	./vector4Add




