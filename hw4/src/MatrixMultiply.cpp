#include <iostream>
#include <cstdlib>

#include "MatrixMultiply.hpp"

#include <exception>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <numeric>

mabcb7::MatrixMultiply::MatrixMultiply() 
{
	;
}

mabcb7::MatrixMultiply::~MatrixMultiply()
{
	;
}

float * mabcb7::MatrixMultiply::operator()(const mabcb7::FloatMatrix& lhs, const mabcb7::FloatMatrix& rhs) const
{
	// Verify acceptable dimensions
	if (lhs.size2() != rhs.size1())
		throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

	mabcb7::FloatMatrix ret(lhs.size1(),rhs.size2());

	float * result = (float *)malloc(sizeof(float) * (lhs.size1() * rhs.size2()));

	// YOUR ALGORIHM WITH COMMENTS GOES HERE:
	
	//loop counters
	int i, j, k;
	int D = 1;		//number of threads
	//matrix boundaries 1 = rows 2 = columns
	const int lhs1 = lhs.size1();
	const int lhs2 = lhs.size2();
	const int rhs1 = rhs.size1();
	const int rhs2 = rhs.size2();
	const int result1 = lhs1;
	const int result2 = rhs2;
	float temp;
	
	//transpose rhs matrix using defined transpose function.
	mabcb7::FloatMatrix tMatrix = mabcb7::MatrixMultiply::transpose(rhs);
	
	//set up direct access to matrix data as a 1D array
	float * rst = result;
	const float * lhsp = &lhs.data()[0];
	const float * rhsp = &tMatrix.data()[0];
	//const float * rhsp = &rhs.data()[0];
	
	calcRowParam params[D];
	pthread_t threadID[D];
	
	params[0].lhsp = &lhs.data()[0];
	params[0].rhsp = &tMatrix.data()[0];
	params[0].result = result;
	params[0].rhs1 = rhs1;
	params[0].rhs2 = rhs2;
	params[0].lhs1 = lhs1;
	params[0].lhs2 = lhs2;
	
	//perform multiplication
	//lhs1
	for ( i = 0; i < 1; ++i)
	{
		params[0].i = i;
		/*for (j = 0; j < rhs2; ++j)
		{
			temp = 0;
			for (k = 0; k < rhs1; ++k)
			{
				temp += lhsp[k + i * lhs2] * rhsp[k + j * rhs1];
			}
			rst[j + i * rhs2] = temp;
		}*/
		pthread_create(&threadID[0], NULL, (void *(*)(void*)) &mabcb7::MatrixMultiply::ComputeRow, &params[0]);
		//mabcb7::MatrixMultiply::ComputeRow(&params[0]);
		pthread_join(threadID[0], NULL);
	}

	return result;
}

void mabcb7::MatrixMultiply::ComputeRow(calcRowParam * d) const
{
	std::cout << "1\n";
	calcRowParam data = *d;
	std::cout << "1a\n";
	int j, k;
	std::cout << "lb\n";
	float temp;
	std::cout << "lc\n";
	float * r = data.result;
	std::cout << "2\n";
	//data.rhs2
	for (j = 0; j < 1; ++j)
	{
		std::cout << "3\n";
		temp = 0;
		std::cout << "4\n";
		for(k = 0; k < data.rhs1; ++k)
		{
			std::cout << temp << std::endl;
			temp += data.lhsp[k + data.i * data.lhs2] * data.rhsp[k + j * data.rhs1];
			//std::cout << temp << std::endl;
		}
		
		//std::cout << "after\n";
		//std::cout << "Storing result " << temp << std::endl;
		//<< " in place of value " << data.result[j + data.i * data.rhs2] 
		//<< " into result[" << j + data.i * data.rhs2 << "]\n";
		
		//data.result[j + data.i * data.rhs2] = temp;
	}
	
	//std::cout << temp << "\n";
	return;
}



mabcb7::FloatMatrix mabcb7::MatrixMultiply::multiply(const mabcb7::FloatMatrix& lhs, const mabcb7::FloatMatrix& rhs) const
{
	// Verify acceptable dimensions
	if (lhs.size2() != rhs.size1())
		throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

	return boost::numeric::ublas::prod(lhs,rhs);
}

mabcb7::FloatMatrix mabcb7::MatrixMultiply::transpose(const mabcb7::FloatMatrix& matrix) const
{
	int i, j;
	const int m1 = matrix.size1();
	const int m2 = matrix.size2();
	//create new matrix
	mabcb7::FloatMatrix tMatrix(m2, m1);
	
	
	float * tMatrixp = &tMatrix(0,0);
	const float * lhsp = &matrix(0,0);
	
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

//function to convert mabcb7::FloatMatrix into a vector of floats
std::vector<float> mabcb7::MatrixMultiply::makeVector(const mabcb7::FloatMatrix& matrix) const
{
	std::vector<float> vmatrix;
	int i;
	int j;
	
	for (i = 0; i < matrix.size1(); ++i)
	{
		for (j = 0; j < matrix.size2(); ++j)
		{
			vmatrix.push_back(matrix(i, j));
		}
	}
	
	return vmatrix;
}

//function to convert mabcb7::FloatMatrix into a transposed vector of floats
std::vector<float> mabcb7::MatrixMultiply::makeVectorTransposed(const mabcb7::FloatMatrix& matrix) const
{
	std::vector<float> vmatrix;
	int i;
	int j;
	
	for (i = 0; i < matrix.size1(); ++i)
	{
		for (j = 0; j < matrix.size2(); ++j)
		{
			vmatrix.push_back(matrix(j, i));
		}
	}
	
	return vmatrix;
}

//function to convert a vector of floats into a mabcb7::FloatMatrix structure
mabcb7::FloatMatrix mabcb7::MatrixMultiply::makeMatrix(const std::vector<float>& vmatrix, const int a, const int b) const
{
	mabcb7::FloatMatrix mMatrix(a, b);
	
	int i, j;
	
	for (i = 0; i < a; ++i)
	{
		for (j = 0; j < b; ++j)
		{
			mMatrix(i, j) = vmatrix[j + i * b];
		}
	}
	
	return mMatrix;
}