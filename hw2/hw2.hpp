#include <math.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <memory.h>
#include <fcntl.h>           /* For O_* constants */
#include <sys/stat.h>        /* For mode constants */
#include <semaphore.h>


template <typename T>
std::string vectorToCSV(std::vector<T> v);

struct result 
{
	int offset;
	float x, y;
	float distance;
	
	bool operator < (const result& str) const
    {
        return (distance < str.distance);
    }
	
};

std::vector<result> circularSubvectorMatch(const std::vector<float>& svector, const std::vector<float>& cir, const int start, const int end, const int d, const int p_num)
{
	//std::ofstream log("time.txt", std::ios_base::app | std::ios_base::out);
	//std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
	const std::vector<float> threeSixty(cir.begin()+2, cir.end());
	result temp;
	std::vector<result> results;
	
	//extract x and y
	temp.x = cir[0];
	temp.y = cir[1];
	
	int i,j;
	const int sizeOfSearch = svector.size();
	const int sizeOfCircle = threeSixty.size();
	
	int offset = 0;
	
	for (offset = start; offset < end; offset += 5)
	{
		temp.distance = 0;
		temp.offset = offset;
		j = 0;
		
		for (i = offset; i < offset + sizeOfSearch; ++i)
		{
			temp.distance += fabs(svector[j] - threeSixty[i % sizeOfCircle]);
			j++;
		}
		results.push_back(temp);
	}
	std::sort(results.begin(), results.end());
	results.resize(d);
	//std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
	
	//std::chrono::duration<double, std::milli> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(end_time - start_time);
	//cout << "P-" << p_num << ": " << time_span.count() << " seconds\n";
	//log << time_span.count() << std::endl;
	
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
	std::vector<result> test_results = circularSubvectorMatch(test_vector, test_data, 0, 12, 10, 1);
	result temp;
	std::vector<result> test_compare;
	int test_pass = 1;
	
	//temp.coord = make_pair(test_data[0], test_data[1]);
	temp.x = test_data[0];
	temp.y = test_data[1];
	temp.offset = 0;
	temp.distance = 0;
	test_compare.push_back(temp);
	//temp.coord = make_pair(test_data[0], test_data[1]);
	temp.x = test_data[0];
	temp.y = test_data[1];
	temp.offset = 5;
	temp.distance = 49;
	test_compare.push_back(temp);
	//temp.coord = make_pair(test_data[0], test_data[1]);
	temp.x = test_data[0];
	temp.y = test_data[1];
	temp.offset = 10;
	temp.distance = 34;
	test_compare.push_back(temp);
	sort(test_compare.begin(), test_compare.end());
	test_compare.resize(10);
	
	//cout << "Test Data: " << test_data << std::endl;
	//cout << "Test Vector: " << test_vector << std::endl;
	std::vector<result>::iterator it_test = test_results.begin();
	std::vector<result>::iterator it_compare = test_compare.begin();
	printf("%10s | %9s | %9s | %9s | %9s |\n-----------------------------------------------------------\n", "", "x", "y", "Offset", "Distance");
	for (it_test; it_test != test_results.end(); ++it_test)
	{
		printf("%10s | %2.6f | %2.6f | %9d | %1.6f |\n", "Result", it_test->x, it_test->y, it_test->offset, it_test->distance);
		printf("%10s | %2.6f | %2.6f | %9d | %1.6f |\n", "Check", it_compare->x, it_compare->y, it_compare->offset, it_compare->distance);
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