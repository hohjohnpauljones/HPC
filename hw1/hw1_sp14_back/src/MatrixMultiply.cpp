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
	
	int i, j, k;
	scottgs::FloatMatrix tMatrix = scottgs::MatrixMultiply::transpose(rhs);
	std::vector<float> vlhs = makeVector(lhs);
	std::vector<float> vrhs = makeVector(tMatrix);	
	
	for (i = 0; i < lhs.size1(); ++i)
	{
		for (j = 0; j < rhs.size2(); ++j)
		{
			for (k = 0; k < lhs.size2(); ++k)
			{
				//result(i, j) += lhs(i,k) * rhs(k,j);
				//result(i, j) += vlhs[k + i * lhs.size2()] * vrhs[j + k * rhs.size2() - 1];
				result(i, j) += vlhs[k + i * lhs.size2()] * vrhs[k + j * rhs.size1()];
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
	scottgs::FloatMatrix tMatrix(matrix.size2(), matrix.size1());
	int i, j;
	
	for (i = 0; i < matrix.size1(); ++i)
	{
		for (j = 0; j < matrix.size2(); ++j)
		{
			tMatrix(j, i) = matrix(i, j);
		}
	}
	
	return tMatrix;
}

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
