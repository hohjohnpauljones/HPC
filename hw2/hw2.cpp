#include "csvparse.cpp"
#include "hw2.hpp"

int main(int argc, char* argv[]) 
{
	if (argc != 2)
	{
		std::cout << "Error incorrect command line arguments. USAGE: ./csvparse <path to file>" << std::endl;
		return 0;
	}
	
	line_map mapa = parseFile(argv[1]);
	line_map::iterator itr = mapa.begin();
	std::vector<float> svector {9, 8, 7, 6, 5, 4, 3, 2, 1};
	std::vector<result> results;
	
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	
	for (itr; itr !=mapa.end(); ++itr)
	{
		results = (circularSubvectorMatch(svector, itr->second, 9, itr->first));
	}
	
	sort(results.begin(), results.end());
	results.resize(10);
	
	std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(stop - start);
	
	std::cout << "Swinging through once took:  " << time_span.count() << " seconds." << std::endl;
	
	std::vector<result>::iterator it = results.begin();
	
	for (it; it != results.end(); ++ it)
	{
		cout << it->coord.first << ", " << it->coord.second << " " << it->offset << " " << it->distance << std::endl;
	}
	
	return 0;
}
