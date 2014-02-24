#include "csvparse.cpp"
#include "hw2.hpp"

int main(int argc, char* argv[]) 
{
	if (argc != 2)
	{
		std::cout << "Error incorrect command line arguments. USAGE: ./csvparse <path to file>" << std::endl;
		return 0;
	}
	
	int i, j, m, k, start_d, end_d;
	std::vector<result> results;
	std::vector<result> tempResults;
	std::vector<float> sVectors[30];
	int sVectors_size = 9;
	
	//Declare IPC controls and initialize shared memory
	pid_t pid = getpid();
	char MUTEXid[32];
	sprintf(MUTEXid, "semmutex%d", pid);
	sem_t *MUTEXptr = sem_open(MUTEXid, O_CREAT, 0600, 1);
	int shm_id;
	void *shmptr;
	result * res;
	pid_t PID = -1;
	key_t shm_key = 1029384756;
	int D = 10;
	int P = 2;
	int count = 0;
	
	shm_id = shmget(shm_key, sizeof(result)*D*P, IPC_CREAT | 0660);
	shmptr = shmat(shm_id, NULL, 0);
	result * result_shm;
	result_shm = (result*)shmptr;
	
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
	for (i = 0; i < 1; i++)
	{
		sVectors[i] = {0.0536727,0.0384691,0.00146231,0.0122459,0.0198738,-0.116341,0.0998519,0.0269831,-0.000772231};
		//sVectors[i] = generateRandomVector(sizes[j]);
	}

	// 5: for Each vector v in V do
	for (i = 0; i < 1; i++)
	{
		results.erase(results.begin(), results.end());
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		
		// 6: Using P parallel processes
		// 7: circularSubvectorMatch(s, D=P, N) --D is assumed to be the entire dataset, or an indicator of the portion of the data set assigned
		// 8: . where D=P is an even portion of D searched by one of the P processes
		
		for (m = 0; m < mapa_count; ++m)
		{
			count = 0;
			for (k = 0; k < P; k++)
			{
				if (PID != 0)
				{
					if (PID != -1)
					{
						//add PID to the array of child PIDs
					}
					count++;
					PID = fork();
				}
			}
			if (PID == 0) //child
			{
				int const offset = (360 / P) * count;
				//current state: count is child process number
				start_d = (360 / P) * count;
				end_d = offset + (360 / P);
				
				results = circularSubvectorMatch(sVectors[i], mapa[m], start_d, end_d, P, count);
				
				//store results into shared memory
				
				shm_id = shm_id = shmget(shm_key, sizeof(result)*D*P, IPC_CREAT | 0660);
				shmptr = shmat(shm_id, NULL, 0600);
				
				res = (result *) shmptr;
				
				res = res + D * count;
				int d2 = 0;
				for (int d = D * count; d < D * (count + 1); d++)
				{
					cout << count << ": " << results[d2].offset << std::endl;
					res[d] = results[d2];
					d2++;
				}
				exit(0);
			}
			else
			{
				//add last PID to the array of child PIDs
				//wait for all children to run
				//collect data from shared memory 
				//sort results for row
			}
		
		
		
		
			//tempResults = circularSubvectorMatch(sVectors[i], mapa[m], 0 , 0, 1);
			
			// 9: Merge P Partial Results
			// 10: . Alternative to Merge, you can use Shared Memory and Mutex to merge results during searching
			//results.insert(results.end(), tempResults.begin(), tempResults.end());
			sort(results.begin(), results.end());
			results.resize(D);
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
