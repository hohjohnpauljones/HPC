#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <pthread.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

typedef boost::numeric::ublas::matrix<float> FloatMatrix;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

struct mmData {
	FloatMatrix * lhs;
	FloatMatrix * rhs;
	FloatMatrix * result;
} ; 

struct threadData {
	mmData data;
	int threadNum;
	int stepSize;
	int * cells;
} ;

void initRandomMatrix(FloatMatrix& m)
{
	// Initialize each element.
	// See discussion board for better way, 
	// this was originally posted to be a 
	// simple example of per-element access into the matric
	for (unsigned i = 0; i < m.size1(); ++ i)
	        for (unsigned j = 0; j < m.size2(); ++ j)
		            m (i, j) = (static_cast<float>(rand()) / RAND_MAX) * 100.0;
}

FloatMatrix transpose(FloatMatrix& matrix)
{
	int i, j;
	const int m1 = matrix.size1();
	const int m2 = matrix.size2();
	//create new matrix
	FloatMatrix tMatrix(m2, m1);
	
	
	float * tMatrixp = &tMatrix(0,0);
	float * lhsp = &matrix(0,0);
	
	//move data from old matrix to new in transposed format
	for (i = 0; i < m1; ++i)
	{
		for (j = 0; j < m2; ++j)
		{
			//tMatrix(j, i) = matrix(i, j);
			tMatrixp[i + j * m1] = lhsp[j + i * m2];
		}
	}

	return tMatrix;
}

void calcCell(const int i, const int j, const int rhs1, const int rhs2, const int lhs2, float * lhsp, float * rhsp, float * rst)
{
	int temp = 0;
	int k;
	
	for (k = 0; k < rhs2; ++k)
	{
		//temp += lhsp(i, k) * rhs(j, k);
		temp += lhsp[k + i * lhs2] * rhsp[k + j * rhs2];
	}
	rst[j + i * rhs2] = temp;
}


void calcRow(const int i, const int rhs1, const int rhs2, const int lhs2, float * lhsp, float * rhsp, float * rst)
{
	int j;
	for (j = 0; j < rhs2; ++j)
	{
		calcCell(i, j, rhs1, rhs2, lhs2, lhsp, rhsp, rst);
	}
}

void rowInterleavedMM(threadData * tData)
{
	int i, j, k, temp;
	const int tNum = tData->threadNum;
	const int sSize = tData->stepSize;
	const int lhs1 = tData->data.lhs->size1();
	const int lhs2 = tData->data.lhs->size2();
	const int rhs1 = tData->data.rhs->size1();
	const int rhs2 = tData->data.rhs->size2();
	
	float * rst = &tData->data.result->data()[0];
	float * lhsp = &tData->data.lhs->data()[0];
	float * rhsp = &tData->data.rhs->data()[0];
	
	for ( i = tNum; i < lhs1; i += sSize)
	{
		//calcRow(i, rhs1, rhs2, lhs2, lhsp, rhsp, rst);
		for (j = 0; j < rhs1; ++j)
		{
			temp = 0;
			for (k = 0; k < rhs2; ++k)
			{
				temp += lhsp[k + i * lhs2] * rhsp[k + j * rhs2];
			}
			rst[j + i * rhs1] = temp;
		}
	}
}

void rowChunkedMM(threadData * tData)
{
	int i, j, k, temp;
	const int tNum = tData->threadNum;
	const int pNum = tData->stepSize;
	const int lhs1 = tData->data.lhs->size1();
	const int lhs2 = tData->data.lhs->size2();
	const int rhs1 = tData->data.rhs->size1();
	const int rhs2 = tData->data.rhs->size2();
	
	float * rst = &tData->data.result->data()[0];
	float * lhsp = &tData->data.lhs->data()[0];
	float * rhsp = &tData->data.rhs->data()[0];
	
	const int startRow = tNum * (lhs1 / pNum);
	int endRow;
	if (tNum == pNum)
	{
		endRow = lhs1;
	}
	else
	{
		endRow = (tNum + 1) * (lhs1 / pNum);
	}
	
	for ( i = startRow; i < endRow; ++i)
	{
		//calcRow(i, rhs1, rhs2, lhs2, lhsp, rhsp, rst);
		for (j = 0; j < rhs1; ++j)
		{
			temp = 0;
			for (k = 0; k < rhs2; ++k)
			{
				temp += lhsp[k + i * lhs2] * rhsp[k + j * rhs2];
			}
			rst[j + i * rhs1] = temp;
		}
	}
}

void threadPoolMM(threadData * tData)
{
	int c, k , temp;
	//const int tNum = tData->threadNum;
	//const int pNum = tData->stepSize;
	const int lhs1 = tData->data.lhs->size1();
	const int lhs2 = tData->data.lhs->size2();
	const int rhs1 = tData->data.rhs->size1();
	const int rhs2 = tData->data.rhs->size2();
	const int C = lhs1 * rhs1;
	
	float * rst = &tData->data.result->data()[0];
	float * lhsp = &tData->data.lhs->data()[0];
	float * rhsp = &tData->data.rhs->data()[0];
	
	int row;
	int col;
	
	pthread_mutex_lock(&mutex);
	//std::cout << "Thread " << tNum << " value: " << *tData->cells << std::endl;
	c = *tData->cells;
	*tData->cells = c + 1;
	pthread_mutex_unlock(&mutex);
	
	while (c < C)
	{
		
		row = c / rhs1;
		col = c % rhs1;
		
		
		//calcCell(row, col, rhs1, rhs2, lhs2, lhsp, rhsp, rst);
		temp = 0;
		for (k = 0; k < rhs2; ++k)
		{
			temp += lhsp[k + row * lhs2] * rhsp[k + col * rhs2];
		}
		rst[col + row * rhs1] = temp;
		
		pthread_mutex_lock(&mutex);
		//std::cout << "Thread " << tNum << " value: " << *tData->cells << std::endl;
		c = *tData->cells;
		*tData->cells = *tData->cells + 1;
		pthread_mutex_unlock(&mutex);
	}
}



int selfTest(int tNum)
{
	
// ---------------------------------------------
	// BEGIN: Self Test Portion
	// ---------------------------------------------
	std::cout << "Running Self Test" << std::endl
		  << "-----------------" << std::endl;
	FloatMatrix l42(4, 2);
	l42(0,0) = 2; l42(0,1) = 3;
	l42(1,0) = 4; l42(1,1) = 5;
	l42(2,0) = 6; l42(2,1) = 7;
	l42(3,0) = 8; l42(3,1) = 9;

	FloatMatrix r23(2,3);
	r23(0,0) = 2; r23(0,1) = 3; r23(0,2) = 4;
	r23(1,0) = 5; r23(1,1) = 6; r23(1,2) = 7;

	// (19,24,29),(33,42,51),(47,60,73),(61,78,95)
	FloatMatrix er(4,3);
	er(0,0) = 19; er(0,1) = 24; er(0,2) = 29;
	er(1,0) = 33; er(1,1) = 42; er(1,2) = 51;
	er(2,0) = 47; er(2,1) = 60; er(2,2) = 73;
	er(3,0) = 61; er(3,1) = 78; er(3,2) = 95;

	
	//float * res = (float *)malloc(sizeof(float) * 12); // = (l42, r23);
	FloatMatrix res(4,3);
	
	FloatMatrix r23T(3,2);
	
	r23T = transpose(r23);
	
	threadData data;
	
	int * C = (int *)malloc(sizeof(int));
	*C = 0;
	
	
	data.threadNum = 0;
	data.stepSize = 1;
	data.cells = C;
	data.data.lhs = &l42;
	data.data.rhs = &r23T;
	data.data.result = &res;
	
	rowInterleavedMM(&data);
	
	
	std::cout << "Result of Interleaved" << std::endl << l42 << std::endl
		  << "   times  " << std::endl << r23 << std::endl
		  << "  equals  " << std::endl << res << std::endl;

	if ( ! std::equal( er.data().begin(), er.data().end(), res.data().begin() ) )
	{
		std::cerr << "Self Test Expected Result: " << er << std::endl
			  << "           ... but found : " << res << std::endl;
		free(C);
		return 1;
	}
	
	rowChunkedMM(&data);
	
	std::cout << "Result of Blocking" << std::endl << l42 << std::endl
		  << "   times  " << std::endl << r23 << std::endl
		  << "  equals  " << std::endl << res << std::endl;

	if ( ! std::equal( er.data().begin(), er.data().end(), res.data().begin() ) )
	{
		std::cerr << "Self Test Expected Result: " << er << std::endl
			  << "           ... but found : " << res << std::endl;
		free(C);
		return 1;
	}
	
	threadPoolMM(&data);
	
	std::cout << "Result of Thread Pool" << std::endl << l42 << std::endl
		  << "   times  " << std::endl << r23 << std::endl
		  << "  equals  " << std::endl << res << std::endl;

	if ( ! std::equal( er.data().begin(), er.data().end(), res.data().begin() ) )
	{
		std::cerr << "Self Test Expected Result: " << er << std::endl
			  << "           ... but found : " << res << std::endl;
		free(C);
		return 1;
	}
	
	std::cout << "Self Test Complete" << std::endl
		  << "==================" << std::endl;

	// ---------------------------------------------
	// END: Self Test Portion
	// ---------------------------------------------	
	
	free(C);
	
	return 0;
	
}