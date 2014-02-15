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
	//create new matrix
	scottgs::FloatMatrix tMatrix(matrix.size2(), matrix.size1());
	int i, j;
	
	//move data from old matrix to new in transposed format
	for (i = 0; i < matrix.size1(); ++i)
	{
		for (j = 0; j < matrix.size2(); ++j)
		{
			tMatrix(j, i) = matrix(i, j);
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