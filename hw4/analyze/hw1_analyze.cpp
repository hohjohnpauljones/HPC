#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <chrono>

#include <boost/numeric/ublas/io.hpp>

#include "../src/FloatMatrix.hpp"
#include "../src/MatrixMultiply.hpp"

void initRandomMatrix(mabcb7::FloatMatrix& m);

int main(int argc, char * argv[])
{
	
	// ---------------------------------------------
	// BEGIN: Timing Analysis
	// ---------------------------------------------
	std::cout << "Running Timing Analysis" << std::endl
		  << "-----------------------" << std::endl;
	srand(123456);	// use a constant number to seed the pseudo random number generator
			// this way your results at least have the same input on a given system
	
	const unsigned int ITR=100;
	
#if 1
	// ---------------------------------------------
	// Build up a set of test matrix-pair sizes
	// ---------------------------------------------
	std::vector<std::pair<std::pair<unsigned short,unsigned short>, std::pair<unsigned short,unsigned short> > > testList;
	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(50,40),
					   std::pair<unsigned short,unsigned short>(40,60)) );


	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(60,50),
					   std::pair<unsigned short,unsigned short>(50,70)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(70,60),
					   std::pair<unsigned short,unsigned short>(60,80)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(80,70),
					   std::pair<unsigned short,unsigned short>(70,90)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(90,80),
					   std::pair<unsigned short,unsigned short>(80,100)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(100,90),
					   std::pair<unsigned short,unsigned short>(90,110)) );
					   
	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(200,90),
					   std::pair<unsigned short,unsigned short>(90,220)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(200,180),
					   std::pair<unsigned short,unsigned short>(180,220)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(400,360),
					   std::pair<unsigned short,unsigned short>(360,240)) );

	/*				   
	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(500,400),
					   std::pair<unsigned short,unsigned short>(400,600)) );


	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(600,500),
					   std::pair<unsigned short,unsigned short>(500,700)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(700,600),
					   std::pair<unsigned short,unsigned short>(600,800)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(800,700),
					   std::pair<unsigned short,unsigned short>(700,900)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(90,800),
					   std::pair<unsigned short,unsigned short>(800,1000)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(1000,900),
					   std::pair<unsigned short,unsigned short>(900,1100)) );
					   
	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(2000,900),
					   std::pair<unsigned short,unsigned short>(900,2200)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(2000,1800),
					   std::pair<unsigned short,unsigned short>(1800,2200)) );

	testList.push_back( std::make_pair(std::pair<unsigned short,unsigned short>(4000,3600),
					   std::pair<unsigned short,unsigned short>(3600,2400)) );
	
	*/
	
	// ***********************************
	// Test functor
	//	Direct matrix element access
	// ***********************************

	mabcb7::MatrixMultiply mm;

	for (std::vector<std::pair<std::pair<unsigned short,unsigned short>, std::pair<unsigned short,unsigned short> > >::const_iterator t=testList.begin();
		t!=testList.end();++t)
	{
		// Get matrix size pairs from iterator
		// instantiate matrices, then randomize
		std::pair<unsigned short,unsigned short> lp = t->first;
		std::pair<unsigned short,unsigned short> rp = t->second;
		mabcb7::FloatMatrix l(lp.first,lp.second);
		mabcb7::FloatMatrix r(rp.first,rp.second);
		initRandomMatrix(l);
		initRandomMatrix(r);

		// Run Timing Experiment
                std::chrono::high_resolution_clock c;
                std::chrono::high_resolution_clock::time_point start = c.now();
		for (unsigned int i = 0; i < ITR; ++i)
		{
			float * p = mm(l,r);
		}
                std::chrono::high_resolution_clock::time_point stop = c.now();
		double avgMs = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (1000000 * ITR);

		// Compute Ops and Elements
		// Log timing statistics
		const unsigned long opsMaybe = l.size1() * r.size2() * l.size2() + l.size1() + r.size2();
		const unsigned long elements = l.size1() * r.size2();
		std::cout << "------------------------------------------------------------------" << std::endl
			  << ITR << " iterations of matrix multiplication (functor) ran using ("
			  << l.size1() <<","<< l.size2() <<")*("
			  << r.size1() <<","<< r.size2() <<") = ("
			  << l.size1() <<","<< r.size2() <<")" << std::endl
			  << "      :Method:Average Time (s):approximate ops:computed elements" << std::endl
			  << "Data  Point:f:" << avgMs << ":" << opsMaybe<< ":" << elements << std::endl;
	}

#endif

#if 0
	// ***********************************
	// Test first method
	//	Column / Row Vector Proxies
	//	Iterator Dereferencing
	// ***********************************
	
	for (std::vector<std::pair<std::pair<unsigned short,unsigned short>, std::pair<unsigned short,unsigned short> > >::const_iterator t=testList.begin();
		t!=testList.end();++t)
	{
		// Get matrix size pairs from iterator
		// instantiate matrices, then randomize
		std::pair<unsigned short,unsigned short> lp = t->first;
		std::pair<unsigned short,unsigned short> rp = t->second;
		mabcb7::FloatMatrix l(lp.first,lp.second);
		mabcb7::FloatMatrix r(rp.first,rp.second);
		initRandomMatrix(l);
		initRandomMatrix(r);

		// Run Timing Experiment
                std::chrono::high_resolution_clock c;
                std::chrono::high_resolution_clock::time_point start = c.now();
                for (unsigned int i = 0; i < ITR; ++i)
		{
			mabcb7::FloatMatrix p = mm.multiply(l,r);
		}
                std::chrono::high_resolution_clock::time_point stop = c.now();
		double avgMs = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (1000000 * ITR);

		// Compute Ops and Elements
		// Log timing statistics
		const unsigned long opsMaybe = l.size1() * r.size2() * l.size2() + l.size1() + r.size2();
		const unsigned long elements = l.size1() * r.size2();
		std::cout << "------------------------------------------------------------------" << std::endl
			  << ITR << " iterations of matrix multiplication (method) ran using ("
			  << l.size1() <<","<< l.size2() <<")*("
			  << r.size1() <<","<< r.size2() <<") = ("
			  << l.size1() <<","<< r.size2() <<")" << std::endl
			  << "      :Method:Average Time (s):approximate ops:computed elements" << std::endl
			  << "Data Point:m1:" << avgMs << ":" << opsMaybe<< ":" << elements << std::endl;

	}
#endif

#if 0
	// ***********************************
	// Test second method
	//	Column / Row Vector Proxies
	//	Indexed access
	// ***********************************

	for (std::vector<std::pair<std::pair<unsigned short,unsigned short>, std::pair<unsigned short,unsigned short> > >::const_iterator t=testList.begin();
		t!=testList.end();++t)
	{
		// Get matrix size pairs from iterator
		// instantiate matrices, then randomize
		std::pair<unsigned short,unsigned short> lp = t->first;
		std::pair<unsigned short,unsigned short> rp = t->second;
		mabcb7::FloatMatrix l(lp.first,lp.second);
		mabcb7::FloatMatrix r(rp.first,rp.second);
		initRandomMatrix(l);
		initRandomMatrix(r);

		// Run Timing Experiment
                std::chrono::high_resolution_clock c;
                std::chrono::high_resolution_clock::time_point start = c.now();
		for (unsigned int i = 0; i < ITR; ++i)
		{
			mabcb7::FloatMatrix p = mm.multiply2(l,r);
		}
                std::chrono::high_resolution_clock::time_point stop = c.now();
		double avgMs = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (1000000 * ITR);

		const unsigned long opsMaybe = l.size1() * r.size2() * l.size2() + l.size1() + r.size2();
		const unsigned long elements = l.size1() * r.size2();
		std::cout << "------------------------------------------------------------------" << std::endl
			  << ITR << " iterations of matrix multiplication (method2) ran using ("
			  << l.size1() <<","<< l.size2() <<")*("
			  << r.size1() <<","<< r.size2() <<") = ("
			  << l.size1() <<","<< r.size2() <<")" << std::endl
			  << "      :Method:Average Time (s):approximate ops:computed elements" << std::endl
			  << "Data Point:m2:" << avgMs << ":" << opsMaybe<< ":" << elements << std::endl;
	}
#endif

#if 0
	// ***********************************
	// Test third method
	//	Column / Row Vector Proxies
	//	Iterator and stl::inner_product
	// ***********************************

	for (std::vector<std::pair<std::pair<unsigned short,unsigned short>, std::pair<unsigned short,unsigned short> > >::const_iterator t=testList.begin();
		t!=testList.end();++t)
	{
		// Get matrix size pairs from iterator
		// instantiate matrices, then randomize
		std::pair<unsigned short,unsigned short> lp = t->first;
		std::pair<unsigned short,unsigned short> rp = t->second;
		mabcb7::FloatMatrix l(lp.first,lp.second);
		mabcb7::FloatMatrix r(rp.first,rp.second);
		initRandomMatrix(l);
		initRandomMatrix(r);

		// Run Timing Experiment
                std::chrono::high_resolution_clock c;
                std::chrono::high_resolution_clock::time_point start = c.now();
		for (unsigned int i = 0; i < ITR; ++i)
		{
			mabcb7::FloatMatrix p = mm.multiply3(l,r);
		}
                std::chrono::high_resolution_clock::time_point stop = c.now();
		double avgMs = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (1000000 * ITR);

		// Compute Ops and Elements
		// Log timing statistics
		const unsigned long opsMaybe = l.size1() * r.size2() * l.size2() + l.size1() + r.size2();
		const unsigned long elements = l.size1() * r.size2();
		std::cout << "------------------------------------------------------------------" << std::endl
			  << ITR << " iterations of matrix multiplication (method3) ran using ("
			  << l.size1() <<","<< l.size2() <<")*("
			  << r.size1() <<","<< r.size2() <<") = ("
			  << l.size1() <<","<< r.size2() <<")" << std::endl
			  << "      :Method:Average Time (s):approximate ops:computed elements" << std::endl
			  << "Data Point:m3:" << avgMs << ":" << opsMaybe<< ":" << elements << std::endl;
	}
#endif

#if 0
	// ***********************************
	// Test Fourth method
	//	Column / Row Vector Proxies
	//	ublas maxtrix product
	// ***********************************

	for (std::vector<std::pair<std::pair<unsigned short,unsigned short>, std::pair<unsigned short,unsigned short> > >::const_iterator t=testList.begin();
		t!=testList.end();++t)
	{
		// Get matrix size pairs from iterator
		// instantiate matrices, then randomize
		std::pair<unsigned short,unsigned short> lp = t->first;
		std::pair<unsigned short,unsigned short> rp = t->second;
		mabcb7::FloatMatrix l(lp.first,lp.second);
		mabcb7::FloatMatrix r(rp.first,rp.second);
		initRandomMatrix(l);
		initRandomMatrix(r);

		// Run Timing Experiment
                std::chrono::high_resolution_clock c;
                std::chrono::high_resolution_clock::time_point start = c.now();
		for (unsigned int i = 0; i < ITR; ++i)
		{
			mabcb7::FloatMatrix p = mm.multiply4(l,r);
		}
                std::chrono::high_resolution_clock::time_point stop = c.now();
		double avgMs = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (1000000 * ITR);

		// Compute Ops and Elements
		// Log timing statistics
		const unsigned long opsMaybe = l.size1() * r.size2() * l.size2() + l.size1() + r.size2();
		const unsigned long elements = l.size1() * r.size2();
		std::cout << "------------------------------------------------------------------" << std::endl
			  << ITR << " iterations of matrix multiplication (method4) ran using ("
			  << l.size1() <<","<< l.size2() <<")*("
			  << r.size1() <<","<< r.size2() <<") = ("
			  << l.size1() <<","<< r.size2() <<")" << std::endl
			  << "      :Method:Average Time (s):approximate ops:computed elements" << std::endl
			  << "Data Point:m4:" << avgMs << ":" << opsMaybe<< ":" << elements << std::endl;
	}
#endif

	std::cout << "Timing Analysis Completed" << std::endl
		  << "=========================" << std::endl;
	// ---------------------------------------------
	// END: Timing Analysis
	// ---------------------------------------------
}

void initRandomMatrix(mabcb7::FloatMatrix& m)
{
	// Initialize each element.
	// See discussion board for better way, 
	// this was originally posted to be a 
	// simple example of per-element access into the matric
	for (unsigned i = 0; i < m.size1(); ++ i)
	        for (unsigned j = 0; j < m.size2(); ++ j)
		            m (i, j) = (static_cast<float>(rand()) / RAND_MAX) * 100.0;
}

