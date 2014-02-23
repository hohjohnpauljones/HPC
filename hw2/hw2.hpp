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
	
	for (offset = 0; offset < sizeOfCircle; offset += 1)
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
	
	//std::sort(results.begin(), results.end());
	results.resize(10);
	return results;
}