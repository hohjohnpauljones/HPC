#include "MatrixMultiply.hpp"

#include <exception>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <numeric>

scottgs::MatrixMultiply::MatrixMultiply() 
{
	;
}

scottgs::MatrixMultiply::~MatrixMultiply()
{
	;
}


scottgs::FloatMatrix scottgs::MatrixMultiply::operator()(const scottgs::FloatMatrix& lhs, const scottgs::FloatMatrix& rhs) const
{
	// Verify acceptable dimensions
	if (lhs.size2() != rhs.size1())
		throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

	scottgs::FloatMatrix result(lhs.size1(),rhs.size2());


	// YOUR ALGORIHM WITH COMMENTS GOES HERE:
	
	//loop counters
	int i, j, k;
	//matrix boundaries 1 = rows 2 = columns
	int const lhs1 = lhs.size1();
	int const lhs2 = lhs.size2();
	int const rhs1 = rhs.size1();
	int const rhs2 = rhs.size2();
	int const result1 = lhs1;
	int const result2 = rhs2;
	float temp;
	
	scottgs::FloatMatrix tMatrix = scottgs::MatrixMultiply::transpose(rhs);
	
	//set up direct access to matrix data as a 1D array
	float * rst = &result.data()[0];
	const float * lhsp = &lhs.data()[0];
	const float * rhsp = &tMatrix.data()[0];
	//const float * rhsp = &rhs.data()[0];
	
	
	
	//perform multiplication 
	for ( i = 0; i < lhs1; ++i)
	{
	#pragma omp parallel for schedule(dynamic) private(temp)
		for (j = 0; j < rhs2; ++j)
		{
			temp = 0;
			for (k = 0; k < rhs1; ++k)
			{
				//temp += lhs(i, k) * rhs(k,j);
				//temp += lhsp[k + i * lhs2] * rhsp[j + k * rhs2];
				temp += lhsp[k + i * lhs2] * rhsp[k + j * rhs1];
			}
			//result(i,j) = temp;
			rst[j + i * rhs2] = temp;
		}
	}
	
	
	//perform multiplication on transposed rhs
	/*
	#pragma omp parallel for
	for ( i = 0; i < lhs1; ++i)
	{
		for (j = 0; j < rhs2; ++j)
		{
			temp = 0;
			for (k = 0; k < rhs1; ++k)
			{
				//temp += lhs(i, k) * rhs(k,j);
				temp += lhsp[k + i * lhs2] * rhsp[k + j * rhs1];
			}
			//result(i,j) = temp;
			rst[j + i * rhs2] = temp;
		}
	}
	*/
	
	/*	Original algorithm
	//transpose one matrix
	scottgs::FloatMatrix tMatrix = scottgs::MatrixMultiply::transpose(rhs);
	
	//set up direct access to matrix data as a 1D array
	float * rst = &result(0,0);				//result
	const float * lhsp = &lhs(0,0);			//left matrix
	const float * rhsp = &tMatrix(0,0);		//right matrix
	
	//perform multiplication
	for (i = 0; i < lhs1; ++i)
	{
		for (j = 0; j < rhs2; ++j)
		{
			for (k = 0; k < lhs2; ++k)
			{
				rst[j + i * rhs2] += lhsp[k + i * lhs2] * rhsp[k + j * rhs1];
			}
		}
	}
	*/
	return result;
}

scottgs::FloatMatrix scottgs::MatrixMultiply::multiply(const scottgs::FloatMatrix& lhs, const scottgs::FloatMatrix& rhs) const
{
	// Verify acceptable dimensions
	if (lhs.size2() != rhs.size1())
		throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

	return boost::numeric::ublas::prod(lhs,rhs);
}

scottgs::FloatMatrix scottgs::MatrixMultiply::transpose(const scottgs::FloatMatrix& matrix) const
{
	int i, j;
	const int m1 = matrix.size1();
	const int m2 = matrix.size2();
	//create new matrix
	scottgs::FloatMatrix tMatrix(m2, m1);
	
	
	float * tMatrixp = &tMatrix(0,0);
	const float * lhsp = &matrix(0,0);
	
	//move data from old matrix to new in transposed format
	#pragma omp parallel for
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

//function to convert scottgs::FloatMatrix into a vector of floats
std::vector<float> scottgs::MatrixMultiply::makeVector(const scottgs::FloatMatrix& matrix) const
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

//function to convert scottgs::FloatMatrix into a transposed vector of floats
std::vector<float> scottgs::MatrixMultiply::makeVectorTransposed(const scottgs::FloatMatrix& matrix) const
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

//function to convert a vector of floats into a scottgs::FloatMatrix structure
scottgs::FloatMatrix scottgs::MatrixMultiply::makeMatrix(const std::vector<float>& vmatrix, const int a, const int b) const
{
	scottgs::FloatMatrix mMatrix(a, b);
	
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