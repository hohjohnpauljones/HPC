#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <pthread.h>

#include <boost/numeric/ublas/matrix.hpp>

typedef boost::numeric::ublas::matrix<float> FloatMatrix;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

struct mmData {
	FloatMatrix * lhs;
	FloatMatrix * rhs;
	float * result;
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


void calcCell(const int i, const int j, const int rhs1, const int rhs2, const int lhs2, float * lhsp, float * rhsp, float * rst)
{
	int temp = 0;
	int k;
	
	for (k = 0; k < rhs1; ++k)
	{
		temp += lhsp[k + i * lhs2] * rhsp[k + j * rhs1];
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
	int i;
	const int tNum = tData->threadNum;
	const int sSize = tData->stepSize;
	const int lhs1 = tData->data.lhs->size1();
	const int lhs2 = tData->data.lhs->size2();
	const int rhs1 = tData->data.rhs->size1();
	const int rhs2 = tData->data.rhs->size2();
	
	float * rst = tData->data.result;
	float * lhsp = &tData->data.lhs->data()[0];
	float * rhsp = &tData->data.rhs->data()[0];
	
	for ( i = tNum; i < lhs1; i += sSize)
	{
		calcRow(i, rhs1, rhs2, lhs2, lhsp, rhsp, rst);
	}
}

void rowChunkedMM(threadData * tData)
{
	int i;
	const int tNum = tData->threadNum;
	const int pNum = tData->stepSize;
	const int lhs1 = tData->data.lhs->size1();
	const int lhs2 = tData->data.lhs->size2();
	const int rhs1 = tData->data.rhs->size1();
	const int rhs2 = tData->data.rhs->size2();
	
	float * rst = tData->data.result;
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
		calcRow(i, rhs1, rhs2, lhs2, lhsp, rhsp, rst);
	}
}

void threadPoolMM(threadData * tData)
{
	int c;
	const int tNum = tData->threadNum;
	const int pNum = tData->stepSize;
	const int lhs1 = tData->data.lhs->size1();
	const int lhs2 = tData->data.lhs->size2();
	const int rhs1 = tData->data.rhs->size1();
	const int rhs2 = tData->data.rhs->size2();
	const int C = lhs1 * rhs2;
	
	float * rst = tData->data.result;
	float * lhsp = &tData->data.lhs->data()[0];
	float * rhsp = &tData->data.rhs->data()[0];
	
	int row;
	int col;
	
	pthread_mutex_lock(&mutex);
	//std::cout << "Thread " << tNum << " value: " << *tData->cells << std::endl;
	c = *tData->cells;
	*tData->cells = *tData->cells + 1;
	pthread_mutex_unlock(&mutex);
	
	while (c < C)
	{
		
		row = c / rhs2;
		col = c % rhs2;
		//std::cout << "(" << row << "," << col << ")\n";
		calcCell(row, col, rhs1, rhs2, lhs2, lhsp, rhsp, rst);
		
		pthread_mutex_lock(&mutex);
		//std::cout << "Thread " << tNum << " value: " << *tData->cells << std::endl;
		c = *tData->cells;
		*tData->cells = *tData->cells + 1;
		pthread_mutex_unlock(&mutex);
	}
}