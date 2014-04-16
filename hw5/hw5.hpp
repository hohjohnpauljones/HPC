#include <math.h>
#include <stdio.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <exception>
#include <stdexcept>
#include <map>
#include <list>
#include <string>

#include <string.h>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <chrono>

#include <boost/filesystem.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

#include "directory_scanner.cpp"

// +++++++++++++++++++++++++++++++++++++++++++++++++++

// See the boost documentation for the filesystem
// Especially: http://www.boost.org/doc/libs/1_41_0/libs/filesystem/doc/reference.html#Path-decomposition-table
// Link against boost_filesystem-mt (for multithreaded) or boost_filesystem


// +++++++++++++++++++++++++++++++++++++++++++++++++++



using namespace std;
using namespace scottgs;

#define NUMFILES 79;
#define RESULTSSIZE 10000;

typedef std::vector<float> data_v;

typedef std::map<std::pair<float,float>, vector<float> > line_map;
typedef std::vector<std::vector<float> > lineType;

lineType parseFile(const char* filename);

struct worker_param {
	int N;
	float s_vector[29];
	char filenames[100][60];
};

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

struct Row {
	float X;
	float Y;
	data_v data;
	std::vector<result> results;
};

struct row_producer_param {
	string fname;
	std::vector<Row> * rows;
};



void row_producer(string fname, std::vector<Row> * rows);

void row_consumer();

void process_row(data_v * search_v, Row * row, int result_size, int offset, int search_size);

float V_dissimilarity(data_v * V1, data_v * V2, int k);

std::vector<float> generateRandomVector(unsigned int size)
{
	// Again, there is a better way to do this using STL generator or STL bind
	std::vector<float> rv(size, 0);
	for (std::vector<float>::iterator i=rv.begin(); i!=rv.end(); ++i)
		*i =  ((static_cast<float>(rand()) / RAND_MAX) * 2.0) - 1.0;

	return rv;
}


vector<float> splitFloat(const string& s, const string& delim) {
    vector<float> result;
	
    string::const_iterator substart = s.begin(), subend;
	
    while (true) {
        subend = search(substart, s.end(), delim.begin(), delim.end());
        string temp(substart, subend);
        if (!temp.empty()) {
            result.push_back(::atof(temp.c_str()));
        }
        if (subend == s.end()) {
            break;
        }
        substart = subend + delim.size();
    }
    return result;
	}

lineType parseFile(const char* filename)
{

	//std::multimap<std::pair<float,float>, vector<float>> mapa;	//multimap to hold data
	//line_map mapa;
	int count = 0;												//counter for number of lines read and processed
	ifstream myfile;									//file stream
	string line;												//temporary line pointer
	float x;													
	float y;
	std::vector<std::vector<float> > lines;
	
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	
	myfile.open(filename, std::ifstream::in);
	
	if (myfile.is_open())
	  {
		while ( getline (myfile,line) )
		{
			//getline(myfile,line);
			++count;
			vector<float> tokens = splitFloat(line, ",");
			//x = tokens[0];
			//y = tokens[1];
			//tokens.erase(tokens.begin(),tokens.begin()+1);
			
			//mapa.insert(std::pair<std::pair<float, float>, vector<float>>(std::make_pair(x,y), tokens));
			lines.push_back(tokens);
			
		}
	  }
	else
	{
		std::cout << "Unable to open file: " << filename << std::endl;
	}
	
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(stop - start);
	
	//std::cout << "Parsing the file took  " << time_span.count() << " seconds." << std::endl;
	//std::cout << count << " lines were parsed" << std::endl;
	
	myfile.close();
	
	return lines;
	
}

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
	if (results.size() > d)
	{
		results.resize(d);
	}
	//std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
	
	//std::chrono::duration<double, std::milli> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(end_time - start_time);
	//cout << "P-" << p_num << ": " << time_span.count() << " seconds\n";
	//log << time_span.count() << std::endl;
	
	return results;
	
}
/*

void parallel_compute(char * filename, int N, int p, int p_degree)
{
	lineType lines = parseFile(param.filenames[i]);
	std::vector<result> result_tmp;
	std::cout << "Process " << world_rank << " parsed file " << param.filenames[i] << std::endl;
	for (j = 0; j < lines.size(); j++)
	{
		result_tmp.erase(result_tmp.begin(), result_tmp.end());
		result_tmp = circularSubvectorMatch(s_vector, lines[j], 0, 360, N, 1);
		sort(result_tmp.begin(), result_tmp.end());
		if (result_tmp.size() > N)
		{
			result_tmp.resize(N);
		}
		
		results.insert(results.end(), result_tmp.begin(), result_tmp.end());
		sort(results.begin(), results.end());
		if (results.size() > N)
		{
			results.resize(N);
		}
			
		
		
		for(k = 0; k < result_tmp.size(); k++)
		{
			//std::cout << "\t" << "line " << j << " result " << k << ": " << "(" << result_tmp[k].x << ", " << result_tmp[k].y << ") => " << result_tmp[k].distance << std::endl;
		}
	}
}
*/