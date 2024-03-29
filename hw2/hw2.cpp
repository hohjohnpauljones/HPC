#include "csvparse.cpp"
#include "hw2.hpp"

int main(int argc, char* argv[]) 
{
	if (argc != 2)
	{
		std::cout << "Error incorrect command line arguments. USAGE: ./csvparse <path to file>" << std::endl;
		return 0;
	}
	
	
	//Declare variables 
	int i, j, m, k, d; 			//counters
	int start_d, end_d;			//data partitioning boundaries
	std::vector<result> results;		//results vector
	std::vector<float> sVectors[30];	//array of search vectors
	int sVectors_size;					//number of elements in the search vector
	
	//IPC controls
	pid_t pid = getpid();		
	int shm_id;
	void *shmptr;
	result * res;
	pid_t PID = -1;
	key_t shm_key = 1029384756;
	int D = 10;						//SETS THE NUMBER OF TOP RESULTS TO FIND - NOT YET IMPLEMENTED
	int P = 1;						//SETS THE DEGREE OF MULTIPROCESSING
	int count = 0;					//child process counter
	
	//initialize shared memory
	shm_id = shmget(shm_key, sizeof(result)*D*P, IPC_CREAT | 0660);
	shmptr = shmat(shm_id, NULL, 0);
	result * result_shm = (result*)shmptr;
	
	// 1: Load data, D, from file into memory
	lineType mapa = parseFile(argv[1]);
	int sizes[] = {9, 11, 17, 29};
	
	// 2: for Each vector size s in S = {9,11,17, 29} do
	for (j = 0; j < 1; j++)
	{
	
	// 3: Run Test Vector against circularSubvectorMatch(Ts, D=P, N) . Verify and display self test results	
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
	
	
	// 4: Generate V as a set of 30 random vectors of length s
	for (i = 0; i < 30; i++)
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
		pid_t children[P];
		
		for (m = 0; m < mapa.size(); ++m)
		{
			count = 0;
			for (k = 0; k < P; k++)
			{
				if (PID != 0) // parent
				{
					if (PID != -1)
					{
						//add PID to the array of child PIDs
						children[count - 1] = PID;
					}
					count++;
					PID = fork();
				}
			}
			if (PID == 0) //child
			{
				start_d = (360 / P) * (count - 1);
				end_d = (360 / P) * count;
				
				// 7: circularSubvectorMatch(s, D=P, N) --D is assumed to be the entire dataset, or an indicator of the portion of the data set assigned
				// 8: . where D=P is an even portion of D searched by one of the P processes
				circularSubvectorMatch(sVectors[i], mapa[m], start_d, end_d, results);
				
				//store results into shared memory
				shm_id = shmget(shm_key, sizeof(result)*D*P, IPC_CREAT | 0660);
				shmptr = shmat(shm_id, NULL, 0600);
				
				res = (result *) shmptr;
				res = res + D * (count - 1);
				
				memcpy(res, &results[0], sizeof(result) * D);
				
				//for ( d = 0; d < 10; d++)
				//{
					//cout << "d " << d << std::endl;
					//cout << "(count - 1) * 360/P = " << (count - 1) * D << std::endl;
					//cout << count << ": " << results[d].offset << std::endl;
				//	res[d] = results[d];
					//memcpy(res
					//cout << count << ": " << res[d + ((count - 1) * D)].offset << std::endl;
				//}
				exit(0);
			}
			else	// parent
			{
				//add last PID to the array of child PIDs
				children[P-1] = PID;
				//wait for all children to run
				for (d = 0; d < P; d++)
				{
					waitpid(children[d], NULL, 0);
				}
				//collect data from shared memory
				for (int count = 0; count < P; count++)
				{
					for (d = 0; d < D; d++)
					{
						results.push_back(result_shm[d + ((count) * D)]);
					}
				}
			}
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
	shmdt(shmptr);                  // detach the shared memory 
	shmctl(shm_id, IPC_RMID, NULL);  // delete the shared memory segment 
	//sem_unlink(MUTEXid);            // delete the MUTEX semaphore 
	
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
