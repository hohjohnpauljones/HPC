#include "csvparse.cpp"
#include "hw2.hpp"

int main(int argc, char* argv[]) 
{
	if (argc != 2)
	{
		std::cout << "Error incorrect command line arguments. USAGE: ./csvparse <path to file>" << std::endl;
		return 0;
	}
	
	int i, m;
	
	std::vector<result> results;
	std::vector<result> tempResults;
	std::vector<float> sVectors[30];
	int sVectors_size = 9;
	
	// 1: Load data, D, from file into Shared Memory -- why does this need to be in shared memory? it isn't mutated at all...
	//line_map mapa = parseFile(argv[1]);
	//line_map::iterator itr = mapa.begin();
	lineType mapa = parseFile(argv[1]);
	const int mapa_count = mapa.size();
	
	
	// 2: for Each vector size s in S = {9,11,17, 29} do
	
	
	// 3: Run Test Vector against circularSubvectorMatch(Ts, D=P, N) . Verify and display self test results	
	
	std::vector<float> test_data = {12,13,1,2,3,4,5,6,7};
	std::vector<float> test_vector = {1,2,3};
	std::vector<result> test_results = circularSubvectorMatch(test_vector, test_data);
	
	//cout << "Test Data: " << test_data << std::endl;
	//cout << "Test Vector: " << test_vector << std::endl;
	std::vector<result>::iterator it_test = test_results.begin();
	for (it_test; it_test != test_results.end(); ++it_test)
	{
		cout << it_test->coord.first << ", " << it_test->coord.second << " " << it_test->offset << " " << it_test->distance << std::endl;
	}
	
	
	// 4: Generate V as a set of 30 random vectors of length s
	for (i = 0; i < 1; i++)
	{
		sVectors[i] = {0.0536727,0.0384691,0.00146231,0.0122459,0.0198738,-0.116341,0.0998519,0.0269831,-0.000772231};
	}

	// 5: for Each vector v in V do
	for (i = 0; i < 1; i++)
	{
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		
		// 6: Using P parallel processes
		// 7: circularSubvectorMatch(s, D=P, N) --D is assumed to be the entire dataset, or an indicator of the portion of the data set assigned
		// 8: . where D=P is an even portion of D searched by one of the P processes
		
		for (m = 0; m < mapa_count; ++m)
		{
			tempResults = circularSubvectorMatch(sVectors[i], mapa[m]);
			
			// 9: Merge P Partial Results
			// 10: . Alternative to Merge, you can use Shared Memory and Mutex to merge results during searching
			results.insert(results.end(), tempResults.begin(), tempResults.end());
			sort(results.begin(), results.end());
			results.resize(10);
		}
		
		std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(stop - start);
		
		// 11: Report Search Time for v
		std::cout << "Swinging through once took:  " << time_span.count() << " seconds." << std::endl;
		
		// 12: Report Match Results for v
		std::vector<result>::iterator it = results.begin();
		for (it; it != results.end(); ++ it)
		{
			cout << it->coord.first << ", " << it->coord.second << " " << it->offset << " " << it->distance << std::endl;
		}
		
		// 13: end for
	}
	
	// 14: end for	
	
	// 15: Free and Release Shared Memory
	// 16: Report average search time for each size, s
	
	
	
	
	
	
	/*
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	
	//for (itr; itr !=mapa.end(); ++itr)
	{
		//results = (circularSubvectorMatch(svector, itr->second, 9, itr->first));
		tempResults = (circularSubvectorMatch(svector, itr->second, 9, itr->first));
		results.insert(results.end(), tempResults.begin(), tempResults.end());
		
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
	*/
	
	return 0;
}
