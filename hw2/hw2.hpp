#include <math.h>

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

std::vector<result> circularSubvectorMatch(std::vector<float> svector, std::vector<float> cir, int n)
{
	int i;
	int sizeOfSearch = svector.size();
	int sizeOfCircle = cir.size();
	
	int offset;
	result temp;
	std::vector<result> results;
	temp.coord = make_pair(cir[0], cir[1]);
	
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
	
	std::sort(results.begin(), results.end());
	results.resize(10);
	return results;
}