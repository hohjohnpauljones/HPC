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

mabcb7::FloatMatrix mabcb7::MatrixMultiply::operator()(const mabcb7::FloatMatrix& lhs, const mabcb7::FloatMatrix& rhs) const
{
	// Verify acceptable dimensions
	if (lhs.size2() != rhs.size1())
		throw std::logic_error("matrix incompatible lhs.size2() != rhs.size1()");

	mabcb7::FloatMatrix result(lhs.size1(),rhs.size2());


	// YOUR ALGORIHM WITH COMMENTS GOES HERE:
	
	//loop counters
	int i, j, k;
	int D;		//number of threads
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
	float * rst = &result.data()[0];
	const float * lhsp = &lhs.data()[0];
	const float * rhsp = &tMatrix.data()[0];
	//const float * rhsp = &rhs.data()[0];
	
	calcRowParam params[D];
	
	params[0].lhsp = &lhs.data()[0];
	params[0].rhsp = &tMatrix.data()[0];
	params[0].result = &result.data()[0];
	params[0].rhs1 = rhs1;
	params[0].rhs2 = rhs2;
	params[0].lhs1 = lhs1;
	params[0].lhs2 = lhs2;
	
	//perform multiplication
	
	for ( i = 0; i < lhs1; ++i)
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
		
		mabcb7::MatrixMultiply::ComputeRow(params[0]);
	}

	return result;
}

void mabcb7::MatrixMultiply::ComputeRow(calcRowParam data) const
{
	
	int j, k;
	float temp;
	float * r = data.result;

	for (j = 0; j < data.rhs2; ++j)
	{
		temp = 0;
		for(k = 0; k < data.rhs1; ++k)
		{
			temp += data.lhsp[k + data.i * data.lhs2] * data.rhsp[k + j * data.rhs1];
		}
		data.result[j + data.i * data.rhs2] = temp;
		//*(r[0]) = 5;
	}
	
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