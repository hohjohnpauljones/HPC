#include "hw4.hpp"

#define P 8

int main(int argc, char * argv[])
{
	
	pthread_t threadID[P]; 
	threadData	threadParam[P];
	std::chrono::high_resolution_clock c;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    double avgMs;
	
	int m;
	
	std::cout << "Homework 4" << std::endl 
		  << "Michael Brush (mabcb7)" << std::endl;
		  
  	if (selfTest(P) == 1)
  	{
  		return 1;
  	}

// ---------------------------------------------
	// BEGIN: Timing Analysis
	// ---------------------------------------------
	std::cout << "Running Timing Analysis" << std::endl
		  << "-----------------------" << std::endl;
	srand(123456);	// use a constant number to seed the pseudo random number generator
			// this way your results at least have the same input on a given system
	
	//const unsigned int ITR=1;

// ---------------------------------------------
	// Build up a set of test matrix-pair sizes
	// ---------------------------------------------
	std::vector<std::pair<std::pair<unsigned short,unsigned short>, std::pair<unsigned short,unsigned short> > > testList;
	
	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(450,250),
					   std::pair<unsigned short,unsigned short>(250,500)) );


	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(1250,1100),
					   std::pair<unsigned short,unsigned short>(1100,900)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(2300,2450),
					   std::pair<unsigned short,unsigned short>(2450,1300)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(3300,3530),
					   std::pair<unsigned short,unsigned short>(3530,4000)) );


	for (std::vector<std::pair<std::pair<unsigned short,unsigned short>, std::pair<unsigned short,unsigned short> > >::const_iterator t=testList.begin();
		t!=testList.end();++t)
	{
		
		// Get matrix size pairs from iterator
		// instantiate matrices, then randomize
		std::pair<unsigned short,unsigned short> lp = t->first;
		std::pair<unsigned short,unsigned short> rp = t->second;
		FloatMatrix l(lp.first,lp.second);
		FloatMatrix r(rp.first,rp.second);
		initRandomMatrix(l);
		initRandomMatrix(r);
		
		FloatMatrix rT = transpose(&r);
		
		FloatMatrix res(lp.first, rp.second);//(float *)malloc(sizeof(float) * lp.first * rp.second);
		int * C = (int *)malloc(sizeof(int));
		//*C = l.size1() * r.size2();
		*C = 0;
		
		const unsigned long opsMaybe = l.size1() * r.size2() * l.size2() + l.size1() + r.size2();
		const unsigned long elements = l.size1() * r.size2();
		
		//Prepare Data
		for (m = 0; m < P; m++)
		{
			threadParam[m].threadNum = m;
			threadParam[m].stepSize = P;
			threadParam[m].cells = C;
			
			threadParam[m].data.lhs = &l;
			threadParam[m].data.rhs = &rT;
			threadParam[m].data.result = &res;
		}
		
		
		//Method 1
		/*
		start = c.now();
		
		//Spawn Threads
		for (m = 0; m < P; m++)
		{
			pthread_create(&threadID[m], NULL, (void *(*)(void *)) &rowInterleavedMM, &threadParam[m]);
		}
		
		//Collect Threads
		for (m = 0; m < P; m++)
		{
			pthread_join(threadID[m], NULL);	
		}
		
		end = c.now();
		
		avgMs = (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000000);
		
		// Compute Ops and Elements
		// Log timing statistics
		std::cout << "Method 1: Row Interleaving" << std::endl
			  << "------------------------------------------------------------------" << std::endl
			  << "1 iterations of matrix multiplication (functor) ran using ("
			  << l.size1() <<","<< l.size2() <<")*("
			  << r.size1() <<","<< r.size2() <<") = ("
			  << l.size1() <<","<< r.size2() <<")" << std::endl
			  << "      Method:Average Time (s): " << avgMs << std::endl
			  << "      Approximate ops: " << opsMaybe << std::endl
			  << "      Computed elements: " << elements << std::endl;
		
		*/
		
		//Method 2
		/*
		start = c.now();
		
		//Spawn Threads
		for (m = 0; m < P; m++)
		{
			pthread_create(&threadID[m], NULL, (void *(*)(void *)) &rowChunkedMM, &threadParam[m]);
		}
		
		//Collect Threads
		for (m = 0; m < P; m++)
		{
			pthread_join(threadID[m], NULL);	
		}
		
		end = c.now();
		
		avgMs = (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000000);
		
		// Compute Ops and Elements
		// Log timing statistics
		std::cout << "Method 2: Row Chunking" << std::endl
			  << "------------------------------------------------------------------" << std::endl
			  << "1 iterations of matrix multiplication (functor) ran using ("
			  << l.size1() <<","<< l.size2() <<")*("
			  << r.size1() <<","<< r.size2() <<") = ("
			  << l.size1() <<","<< r.size2() <<")" << std::endl
			  << "      Method:Average Time (s): " << avgMs << std::endl
			  << "      Approximate ops: " << opsMaybe << std::endl
			  << "      Computed elements: " << elements << std::endl;
		*/
		
		//Method 3
		
		*C = 0;
		
		start = c.now();
		
		//Spawn Threads
		for (m = 0; m < P; m++)
		{
			pthread_create(&threadID[m], NULL, (void *(*)(void *)) &threadPoolMM, &threadParam[m]);
		}
		
		//Collect Threads
		for (m = 0; m < P; m++)
		{
			pthread_join(threadID[m], NULL);	
		}
		
		end = c.now();
		
		avgMs = (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000000);
		
		// Compute Ops and Elements
		// Log timing statistics
		std::cout << "Method 3: Cell Thread Pool" << std::endl
			  << "------------------------------------------------------------------" << std::endl
			  << "1 iterations of matrix multiplication (functor) ran using ("
			  << l.size1() <<","<< l.size2() <<")*("
			  << r.size1() <<","<< r.size2() <<") = ("
			  << l.size1() <<","<< r.size2() <<")" << std::endl
			  << "      Method:Average Time (s): " << avgMs << std::endl
			  << "      Approximate ops: " << opsMaybe << std::endl
			  << "      Computed elements: " << elements << std::endl;
		
		//free(res);
		free(C);
		
	}
	

	return 0;
}
