#include "csvparse.cpp"
#include "hw2.hpp"

int main(int argc, char* argv[]) 
{
	if (argc != 2)
	{
		std::cout << "Error incorrect command line arguments. USAGE: ./csvparse <path to file>" << std::endl;
		return 0;
	}
	
	int i, j, m;
	
	std::vector<result> results;
	std::vector<result> tempResults;
	std::vector<float> sVectors[30];
	int sVectors_size = 9;
	
	// 1: Load data, D, from file into Shared Memory -- why does this need to be in shared memory? it isn't mutated at all...
	//line_map mapa = parseFile(argv[1]);
	//line_map::iterator itr = mapa.begin();
	lineType mapa = parseFile(argv[1]);
	const int mapa_count = mapa.size();
	int sizes[] = {9, 11, 17, 29};
	
	
	// 2: for Each vector size s in S = {9,11,17, 29} do
	for (j = 0; j < 1; j++)
	{
	
	/* 3: Run Test Vector against circularSubvectorMatch(Ts, D=P, N) . Verify and display self test results	
	int test_pass = runTest();
	if (test_pass)
	{
		cout << "Test Passed" << std::endl;
	}
	else
	{
		cout << "Test Failed" << std::endl;
		return 0;
	}
	*/
	
	// 4: Generate V as a set of 30 random vectors of length s
	for (i = 0; i < 30; i++)
	{
		//sVectors[i] = {0.0536727,0.0384691,0.00146231,0.0122459,0.0198738,-0.116341,0.0998519,0.0269831,-0.000772231};
		sVectors[i] = generateRandomVector(sizes[j]);
	}

	// 5: for Each vector v in V do
	for (i = 0; i < 30; i++)
	{
		results.erase(results.begin(), results.end());
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
		std::cout << "Search: " << sizes[j] << "-D" << std::endl
		<< "---------------------" << std::endl
		<< "{" << vectorToCSV(sVectors[i]) << "}" << std::endl;
		
		// 12: Report Match Results for v
		std::vector<result>::iterator it = results.begin();
		printf("%9s | %9s | %9s | %8s |\n---------------------------------------------\n", "x", "y", "Offset", "Distance");
		for (it; it != results.end(); ++ it)
		{
			//cout << it->coord.first << ", " << it->coord.second << " " << it->offset << " " << it->distance << std::endl;
			printf("%1.6f | %1.6f | %9d | %1.6f |\n", it->x, it->y, it->offset, it->distance);
		}
		cout << " Time: " << time_span.count() << " seconds." << std::endl;
		
		
		// 13: end for
	}
	
	// 14: end for	
	}
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
