#ifndef _mabcb7_MATRIX_MULTIPLY
#define _mabcb7_MATRIX_MULTIPLY

#include "FloatMatrix.hpp"

namespace mabcb7 
{

struct calcRowParam {
	//const float * lhsp;
	//const float * rhsp;
	//float * result;
	const mabcb7::FloatMatrix * lhs;
	const mabcb7::FloatMatrix * rhs;
	mabcb7::FloatMatrix * result;
	int rhs1, rhs2, lhs1, lhs2; 
	int i;
} ;

class MatrixMultiply
{
public: 
	MatrixMultiply();
	~MatrixMultiply();

	///
	/// \brief Use naive element access
	/// \param lhs The left-hand operand to the multiplication
	/// \param rhs The right-hand operand to the multiplication
	/// \returns the result of the matrix multiplication
	mabcb7::FloatMatrix operator()(const mabcb7::FloatMatrix& lhs, const mabcb7::FloatMatrix& rhs) const;

	///
	/// \brief Use boost built-in matrix multiplication
	/// \param lhs The left-hand operand to the multiplication
	/// \param rhs The right-hand operand to the multiplication
	/// \returns the result of the matrix multiplication
	mabcb7::FloatMatrix multiply(const mabcb7::FloatMatrix& lhs, const mabcb7::FloatMatrix& rhs) const;

	void * ComputeRow(void *) const;
	
	mabcb7::FloatMatrix transpose(const mabcb7::FloatMatrix&) const;
	
	std::vector<float> makeVector(const mabcb7::FloatMatrix& matrix) const;
	
	std::vector<float> makeVectorTransposed(const mabcb7::FloatMatrix& matrix) const;
	
	mabcb7::FloatMatrix makeMatrix(const std::vector<float>& vmatrix, const int a, const int b) const;
};

}; // end namespace mabcb7
#endif // mabcb7_MATRIX_MULTIPLY


