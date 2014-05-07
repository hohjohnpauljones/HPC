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


typedef std::map<std::pair<float,float>, vector<float>> line_map;
typedef std::vector<std::vector<float>> lineType;

lineType parseFile(const char* filename);


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
	ifstream myfile (filename);									//file stream
	string line;												//temporary line pointer
	float x;													
	float y;
	std::vector<std::vector<float>> lines;
	
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	
	if (myfile.is_open())
	  {
		while ( getline (myfile,line) )
		{
			++count;
			vector<float> tokens = splitFloat(line, ",");
			//x = tokens[0];
			//y = tokens[1];
			//tokens.erase(tokens.begin(),tokens.begin()+1);
			
			//mapa.insert(std::pair<std::pair<float, float>, vector<float>>(std::make_pair(x,y), tokens));
			lines.push_back(tokens);
			
		}
	  }
	
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(stop - start);
	
	std::cout << "Parsing the file took  " << time_span.count() << " seconds." << std::endl;
	std::cout << count << " lines were parsed" << std::endl;
	
	myfile.close();
	
	return lines;
	
}



