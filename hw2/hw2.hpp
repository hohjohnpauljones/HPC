#include <math.h>
#include <random>
#include <algorithm>

using namespace std;


struct result 
{
	std::pair<float,float> coord;
	//float x, y;
	int offset;
	float distance;
};

std::vector<result> circularSubvectorMatch(std::vector<float> svector, std::vector<float> cir, int n)
{
	int i;
	int sizeOfSearch = svector.size();
	int sizeOfCircle = cir.size();
	
	int offset;
	result temp;
	std::vector<result> results;
	temp.coord = make_pair(cir[0], cir[1]);
	//temp.x = cir[0];
	//temp.y = cir[1];
	
	cir.resize(2, cir.size());
	
	for (offset = 0; offset < sizeOfCircle; offset += 5)
	{
		for (i = offset; i < offset + n; i++)
		{
			temp.distance += fabs(svector[i % sizeOfSearch] - cir[i % sizeOfCircle]);
			//cout << i << " " << offset << ": " << temp.distance << " += | " << svector[i % sizeOfSearch] << " - " << cir[i % sizeOfCircle] << " | " << std::endl;
		}
		
		temp.offset = offset;
		results.push_back(temp);
		
	}
	
	//std::sort(results.begin(), results.end());
	//results.resize(10);
	return results;
}


std::vector<float> rvec(unsigned int size)
{
	// Again, there is a better way to do this using STL generator or STL bind
	std::vector<float> rv(size, 0);
	for (std::vector<float>::iterator i=rv.begin(); i!=rv.end(); ++i)
		*i =  ((static_cast<float>(rand()) / RAND_MAX) * 2.0) - 1.0;

	return rv;
}