//include <scottgs/Timing.hpp>

#include <iostream>
#include <cstdlib>
#include <algorithm>

#include "FloatMatrix.hpp"
#include "MatrixMultiply.hpp"

#include <boost/numeric/ublas/io.hpp>

void initRandomMatrix(scottgs::FloatMatrix& m);

int main(int argc, char * argv[])
{
	std::cout << "Homework 1" << std::endl 
		  << "Michael Brush (mabcb7)" << std::endl; 

	// ---------------------------------------------
	// BEGIN: Self Test Portion
	// ---------------------------------------------
	std::cout << "Running Self Test" << std::endl
		  << "-----------------" << std::endl;
	scottgs::FloatMatrix l42(4, 2);
	l42(0,0) = 2; l42(0,1) = 3;
	l42(1,0) = 4; l42(1,1) = 5;
	l42(2,0) = 6; l42(2,1) = 7;
	l42(3,0) = 8; l42(3,1) = 9;

	scottgs::FloatMatrix r23(2,3);
	r23(0,0) = 2; r23(0,1) = 3; r23(0,2) = 4;
	r23(1,0) = 5; r23(1,1) = 6; r23(1,2) = 7;

	// (19,24,29),(33,42,51),(47,60,73),(61,78,95)
	scottgs::FloatMatrix er(4,3);
	er(0,0) = 19; er(0,1) = 24; er(0,2) = 29;
	er(1,0) = 33; er(1,1) = 42; er(1,2) = 51;
	er(2,0) = 47; er(2,1) = 60; er(2,2) = 73;
	er(3,0) = 61; er(3,1) = 78; er(3,2) = 95;

	scottgs::MatrixMultiply mm;
	scottgs::FloatMatrix result = mm(l42,r23);
	std::cout << "Result of " << std::endl << l42 << std::endl
		  << "   times  " << std::endl << r23 << std::endl
		  << "  equals  " << std::endl << result << std::endl;

	if ( ! std::equal( er.data().begin(), er.data().end(), result.data().begin() ) )
	{
		std::cerr << "Self Test Expected Result: " << er << std::endl
			  << "           ... but found : " << result << std::endl;
		return 1;
	}

	scottgs::FloatMatrix result2 = mm.multiply(l42,r23);
	std::cout << "Result of " << std::endl << l42 << std::endl
		  << "   times  " << std::endl << r23 << std::endl
		  << "  equals  " << std::endl << result2 << std::endl;

	if ( ! std::equal( er.data().begin(), er.data().end(), result2.data().begin() ) )
	{
		std::cerr << "Self Test Expected Result: " << er << std::endl
			  << "           ... but found : " << result2 << std::endl;
		return 1;
	}

	std::cout << "Self Test Complete" << std::endl
		  << "==================" << std::endl;

	// ---------------------------------------------
	// END: Self Test Portion
	// ---------------------------------------------

	return 0;
}
