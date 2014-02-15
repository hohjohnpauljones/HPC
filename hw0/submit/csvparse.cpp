#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <chrono>
using namespace std;

vector<float> splitFloat(const string& s, const string& delim, const bool keep_empty = true) {
    vector<float> result;
    if (delim.empty()) {
        result.push_back(::atof(s.c_str()));
        return result;
    }
    string::const_iterator substart = s.begin(), subend;
    while (true) {
        subend = search(substart, s.end(), delim.begin(), delim.end());
        string temp(substart, subend);
        if (keep_empty || !temp.empty()) {
            result.push_back(::atof(temp.c_str()));
        }
        if (subend == s.end()) {
            break;
        }
        substart = subend + delim.size();
    }
    return result;
	}

int main(int argc, char* argv[]) 
{

	if (argc != 2)
	{
		std::cout << "Error incorrect command line arguments. USAGE: ./csvparse <path to file>" << std::endl;
		return 0;
	}

	std::multimap<std::pair<float,float>, vector<float>> mapa;	//multimap to hold data
	int count = 0;												//counter for number of lines read and processed
	ifstream myfile (argv[1]);									//file stream
	string line;												//temporary line pointer
	float x;													
	float y;
	float xmin;													//min and max values
	float ymin;
	float xmax;
	float ymax;
	vector<float> min;
	vector<float> max;
	
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	
	if (myfile.is_open())
	  {
		while ( getline (myfile,line) )
		{
			++count;
			vector<float> tokens = splitFloat(line, ",");
			x = tokens[0];
			y = tokens[1];
			tokens.erase(tokens.begin(),tokens.begin()+1);
			
			mapa.insert(std::pair<std::pair<float, float>, vector<float>>(std::make_pair(x,y), tokens));
		}
	  }
	
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(stop - start);
	
	std::cout << "Parsing the file took  " << time_span.count() << " seconds." << std::endl;
	std::cout << count << " lines were parsed" << std::endl;
	count = 1;
	
	start = std::chrono::high_resolution_clock::now();
	
	std::map<std::pair<float,float>, vector<float>>::iterator itr = mapa.begin();
	xmin = xmax = itr->first.first;
	ymin = ymax = itr->first.second;
	//load min and max vectors
	for (int i = 0; i < itr->second.size(); ++i)
	{
		min.push_back(itr->second[i]);
		max.push_back(itr->second[i]);
	}
	itr++;

	for (itr; itr !=mapa.end(); ++itr)
	{
		count++;
		if (xmin > itr->first.first)
		{
			xmin = itr->first.first;
		}
		if (xmax < itr->first.first)
		{
			xmax = itr->first.first;
		}
		if (ymin > itr->first.second)
		{
			ymin = itr->first.second;
		}
		if (ymax < itr->first.second)
		{
			ymax = itr->first.second;
		}
		vector<float> numbers = itr->second;
		for (int j = 0; j < numbers.size(); ++j)
		{
			if (min[j] > numbers[j])
			{
				min[j] = numbers[j];
			}
			if (max[j] < numbers[j])
			{
				max[j] = numbers[j];
			}
		}
		
	}
	
	stop = std::chrono::high_resolution_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double> >(stop - start);
	
	std::cout << "Finding the bounding values took  " << time_span.count() << " seconds." << std::endl;
	std::cout << count << " lines searched" << std::endl;
	
	cout << "X Min: " << xmin << std::endl << "Y Min: " << ymin << std::endl <<"X max: " << xmax << std::endl << "Y Max: " << ymax << std::endl;
	
	cout << "Min Vector: " << min[0];
	for (int i = 1; i < min.size(); i++)
	{
		cout << ", " << min[i];
	}
	cout << std::endl;
	
	cout << "Max Vector: " << max[0];
	for (int i = 1; i < max.size(); i++)
	{
		cout << ", " << max[i];
	}
	cout << std::endl;
	
	myfile.close();
	
	return 0;
}


