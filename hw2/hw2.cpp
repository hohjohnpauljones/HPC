#include "csvparse.cpp"
#include "hw2.hpp"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <string>
#include <cstdlib> //std::system


using namespace boost::interprocess;

int main(int argc, char* argv[]) 
{
	if (argc != 2)
	{
		std::cout << "Error incorrect command line arguments. USAGE: ./csvparse <path to file>" << std::endl;
		return 0;
	}
	
	line_map fileData = parseFile(argv[1]);
	line_map::iterator itr = fileData.begin();
	std::vector<result> results;
	std::vector<float> search = {1,2,3,4,5,6,7,8,9};
	int search_vector_length = 9;
	
	results = circularSubvectorMatch(search, itr->first, search_vector_length);
	
	
	/*
	//1: Load data, D, from le into Shared Memory
	line_map fileData = parseFile(argv[1]);
	line_map::iterator itr = fileData.begin();
	
	int i, j, k;
	int P = 1;			//degree of multiprocessing
	
	std::vector<std::vector<float>> svectors;
	std::vector<result> results;
	int vsize[] = {9, 11, 17, 29};
	int numSearchSizes = sizeof(vsize)/sizeof(int);
	
	/*
	for (itr; itr !=fileData.end(); ++itr)
	{
	
		results = (circularSubvectorMatch({0.0536727,0.0384691,0.00146231,0.0122459,0.0198738,-0.116341,0.0998519,0.0269831,-0.000772231},
										itr->first, 9));
	
	}
	
	
	
		//2: for Each vector size do
		for (i = 0; i < numSearchSizes; i++)
		{
		
			//4: Generate V as a set of 30 random vectors of length s
			const int s = vsize[i];
			
			cout << "Search: " << s << "D" << std::endl << "-----------" << std::endl;
			

			for (j = 0; j < 1; j++)
			{
				//svectors[j] = {0.0536727,0.0384691,0.00146231,0.0122459,0.0198738,-0.116341,0.0998519,0.0269831,-0.000772231};
				
				svectors[j] = rvec(s);
			}
			
			//5: for Each vector v 2 V do
			for (j = 0; j < 1; j++)
			{
			
				std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
				/*
				6: Using P parallel processes
				7: circularSubvectorMatch(s, D=P, N)
				8: . where D=P is an even portion of D searched by one of the P processes
				9: Merge P Partial Results
				10: . Alternative to Merge, you can use Shared Memory and Mutex to merge results during searching
				
				
				std::vector<result> resulttemp = (circularSubvectorMatch(svectors[j], itr->first, s));
	
				//results.insert(results.end(), resulttemp.begin(), resulttemp.end());
				
				//sort(results.begin(), results.end());
				//results.resize(10);

				std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(stop - start);
		
				std::cout << ":  " << time_span.count() << " seconds." << std::endl;

				std::vector<result>::iterator it = results.begin();
				
				for (it; it != results.end(); ++it)
				{
					cout << it->x << ", " << it->y << " " << it->offset << " " << it->distance << std::endl;
				}
				
			}
		}
	}

	
	*/
	return 0;
}


/*
1: Load data, D, from le into Shared Memory
2: for Each vector size s 2 f9;11;17;29g do
3: Run Test Vector against circularSubvectorMatch(Ts, D=P, N) . Verify and display self test results
4: Generate V as a set of 30 random vectors of length s
5: for Each vector v 2 V do
6: Using P parallel processes
7: circularSubvectorMatch(s, D=P, N)
8: . where D=P is an even portion of D searched by one of the P processes
9: Merge P Partial Results
10: . Alternative to Merge, you can use Shared Memory and Mutex to merge results during searching
11: Report Search Time for v
12: Report Match Results for v
13: end for
14: end for
15: Free and Release Shared Memory
16: Report average search time for each size, s


*/