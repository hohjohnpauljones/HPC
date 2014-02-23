#include <math.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

template <typename T>
std::string vectorToCSV(std::vector<T> v);


struct result 
{
	std::pair<float,float> coord;
	int offset;
	float distance;
	
	bool operator < (const result& str) const
    {
        return (distance < str.distance);
    }
	
};

std::vector<result> circularSubvectorMatch(std::vector<float> svector, std::vector<float> cir)
{
	result temp;
	std::vector<result> results;
	temp.coord = make_pair(cir[0],cir[1]);
	//cir.erase(cir.begin(), cir.begin()+1);
	std::vector<float>(cir.begin()+2, cir.end()).swap(cir);
	
	int i,j;
	const int sizeOfSearch = svector.size();
	const int sizeOfCircle = cir.size();
	
	int offset;
	
	for (offset = 0; offset < sizeOfCircle; offset += 5)
	{
		temp.distance = 0;
		temp.offset = offset;
		j = 0;
		
		for (i = offset; i < offset + sizeOfSearch; ++i)
		{
			//cout << i << " " << offset << ": " << temp.distance << " += |" << svector[j % sizeOfSearch] << " - " << cir[i % sizeOfCircle] << "| " << std::endl;
			temp.distance += fabs(svector[j % sizeOfSearch] - cir[i % sizeOfCircle]);
			j++;
		}
		
		results.push_back(temp);
		
	}
	
	std::sort(results.begin(), results.end());
	results.resize(10);
	return results;
}


std::vector<float> generateRandomVector(unsigned int size)
{
	// Again, there is a better way to do this using STL generator or STL bind
	std::vector<float> rv(size, 0);
	for (std::vector<float>::iterator i=rv.begin(); i!=rv.end(); ++i)
		*i =  ((static_cast<float>(rand()) / RAND_MAX) * 2.0) - 1.0;

	return rv;
}

int runTest()
{
	
	std::vector<float> test_data = {12,13,1,2,3,4,5,6,7,8,9,10,11,12};
	std::vector<float> test_vector = {1,2,3,4,5,6,7,8,9};
	std::vector<result> test_results = circularSubvectorMatch(test_vector, test_data);
	result temp;
	std::vector<result> test_compare;
	int test_pass = 1;
	
	temp.coord = make_pair(test_data[0], test_data[1]);
	temp.offset = 0;
	temp.distance = 0;
	test_compare.push_back(temp);
	temp.coord = make_pair(test_data[0], test_data[1]);
	temp.offset = 5;
	temp.distance = 49;
	test_compare.push_back(temp);
	temp.coord = make_pair(test_data[0], test_data[1]);
	temp.offset = 10;
	temp.distance = 34;
	test_compare.push_back(temp);
	sort(test_compare.begin(), test_compare.end());
	test_compare.resize(10);
	
	//cout << "Test Data: " << test_data << std::endl;
	//cout << "Test Vector: " << test_vector << std::endl;
	std::vector<result>::iterator it_test = test_results.begin();
	std::vector<result>::iterator it_compare = test_compare.begin();
	printf("%9s | %9s | %9s | %8s |\n---------------------------------------------\n", "x", "y", "Offset", "Distance");
	for (it_test; it_test != test_results.end(); ++it_test)
	{
		printf("%10s | %1.6f | %1.6f | %9d | %1.6f |\n", "Result", it_test->coord.first, it_test->coord.second, it_test->offset, it_test->distance);
		printf("%10s | %1.6f | %1.6f | %9d | %1.6f |\n", "Check", it_compare->coord.first, it_compare->coord.second, it_compare->offset, it_compare->distance);
		//cout << it_test->coord.first << ", " << it_test->coord.second << " " << it_test->offset << " " << it_test->distance << std::endl;
		if ((it_test->distance != it_compare->distance) || it_test->offset != it_compare->offset)
		{
			test_pass = 0;
		}
		it_compare++;
	}

	return test_pass;
}


/* ***********************************
	DEFINITION
************************************ */

template <typename T>
std::string vectorToCSV(std::vector<T> v)
{
	std::stringstream ss;
	typename std::vector<T>::const_iterator i=v.begin();
	ss << "['" << (*i)<< "'";
	for (++i ; i!=v.end(); ++i)
		ss <<",'" << (*i) << "'";
	ss << "]";
	return ss.str();
}