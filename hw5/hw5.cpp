
#include "hw5.hpp"

// Define a template type, and its iterator	
typedef std::map<std::string,scottgs::path_list_type> content_type;
typedef content_type::const_iterator content_type_citr;

int main(int argc, char * argv[])
{
	std::chrono::high_resolution_clock::time_point wall_start = std::chrono::high_resolution_clock::now();
	int current_worker = 1;
	int i, j, k;
	int N = atoi(argv[2]);
	int numtests = 2;
	double avgWall = 0;
	double avgSerial = 0;
	
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	// Find out rank, size
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	//Custome MPI data type variables
	worker_param param;
	int type_count;
	type_count = 3;		
	
	//define types
	MPI_Datatype types[type_count];
	types[0] = MPI_INT;
	types[1] = MPI_FLOAT;
	types[2] = MPI_CHAR;
	
	//define block lengths
	int block_lengths[type_count];
	block_lengths[0] = 1;
	block_lengths[1] = 29;
	block_lengths[2] = 100 * 60;
	
	//define block displacements
	MPI_Aint displacements[type_count];
	MPI_Aint addr1, addr2, addr3, addr4;
	MPI_Get_address(&param, &addr1);
	MPI_Get_address(&param.N, &addr2);
	MPI_Get_address(&param.s_vector, &addr3);
	MPI_Get_address(&param.filenames, &addr4);
	displacements[0] = addr2 - addr1;
	displacements[1] = addr3 - addr2;
	displacements[2] = addr4 - addr3;
	
	//define datatype
	MPI_Datatype worker_param_type;
	MPI_Type_create_struct(type_count, block_lengths, displacements, types, &worker_param_type);
	MPI_Type_commit(&worker_param_type);
	
	//define data type for return values
	result return_var;
	int ret_type_count = 2;
	
	MPI_Datatype ret_types[ret_type_count];
	ret_types[0] = MPI_INT;
	ret_types[1] = MPI_FLOAT;
	
	//define block lengths
	int ret_block_lengths[ret_type_count];
	ret_block_lengths[0] = 1;
	ret_block_lengths[1] = 3;
	
	//define block displacements
	MPI_Aint ret_displacements[ret_type_count];
	MPI_Get_address(&return_var, &addr1);
	MPI_Get_address(&return_var.offset, &addr2);
	MPI_Get_address(&return_var.x, &addr3);
	ret_displacements[0] = addr2 - addr1;
	ret_displacements[1] = addr3 - addr2;
	
	//define datatype
	MPI_Datatype worker_return_type;
	MPI_Type_create_struct(ret_type_count, ret_block_lengths, ret_displacements, ret_types, &worker_return_type);
	MPI_Type_commit(&worker_return_type);
	
	
	// We are assuming at least 2 processes for this task
	if (world_size < 2) 
	{
		fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1); 
	}
	
	
	
	//Master
	if (world_rank == 0)
	{
		result worker_results[world_size][N];
		double times[numtests][world_size - 1];
		double wallTimes[numtests];
		int n_file;
		std::vector<float> svector;
		
		for (j = 0; j < numtests; j++)
		{
			n_file = 0;
			cout << "Nodes: " << world_size << std::endl;
			// Get the file list from the directory
			content_type directoryContents = scottgs::getFiles(argv[1]);
		
			// For each type of file found in the directory, 
			// List all files of that type
			for (content_type_citr f = directoryContents.begin(); 
				f!=directoryContents.end();
				++f)
			{
				const scottgs::path_list_type file_list(f->second);
				
				for (scottgs::path_list_type::const_iterator i = file_list.begin();
					i!=file_list.end(); ++i)
				{
					boost::filesystem::path file_path(*i);
					strncpy(param.filenames[n_file], file_path.file_string().data(), 60);
					n_file++;
				}
					
			}
			
			param.N = N;
			svector = generateRandomVector(29);
			
			for (k = 0; k < 29; k++)
			{
				param.s_vector[k] = svector[k];
			}
			
			
			for (current_worker = 1; current_worker < world_size; current_worker++)
			{
				
				MPI::COMM_WORLD.Send(&param, 1, worker_param_type, current_worker, 0);
			}
			
			for (int i = 1; i < world_size; i++)
			{
				MPI::COMM_WORLD.Recv(&worker_results[i][0], N, worker_return_type, i, 1);
				MPI::COMM_WORLD.Recv(&times[j][i], 1, MPI::DOUBLE, i , 2);
				//std::cout << "Node " << i << " took " << times[i] << " seconds" << std::endl;
			}
	
			std::vector<result> g_results;
			
			g_results.assign(&worker_results[1][0], &worker_results[1][0] + (world_size - 1) * N);
			
			sort(g_results.begin(), g_results.end());
			if (g_results.size() > N)
			{
				g_results.resize(N);
			}
			
			std::cout << "Search Vector: " << std::endl;
			for (k = 0; k < 29; k++)
			{
				std::cout << svector[k];
				if (k < 28)
				std::cout << ", ";
			}
			std::cout << std::endl;
			
			std::cout << "Results: " << std::endl;
			
			for (k = 0; k < g_results.size(); k++)
			{
				std::cout << "\t" << k << ": " << "(" << g_results[k].x << ", " << g_results[k].y << ") offset: " << g_results[k].offset << " => " << g_results[k].distance << std::endl;
			}
			
			std::chrono::high_resolution_clock::time_point wall_end = std::chrono::high_resolution_clock::now();
	
			std::chrono::duration<double> time_span_wall = std::chrono::duration_cast<std::chrono::duration<double> >(wall_end - wall_start);
			//std::cout << "Wall Time:  " << time_span_wall.count() << " seconds." << std::endl;
			wallTimes[j] = time_span_wall.count();
		}
		
		for (i = 0; i < numtests; i++)
		{
			avgWall += wallTimes[i];
		}
		avgWall = avgWall / numtests;
		
		for (i = 0; i < numtests; i++)
		{
			for (j = 1; j < world_size; j++)
			{
				avgSerial += times[i][j];
			}
		}
		avgSerial = avgSerial / numtests;
		
		//output average timing tatistics
		std::cout << "Average per vector Wall Time: " << avgWall << std::endl;
		std::cout << "Average per vector Serial Time: " << avgSerial << std::endl;
		std::cout << "Paralelization Speedup: " << avgSerial / avgWall << std::endl;
		
	}
	//Worker
	else
	{
		std::vector<result> results;
		std::vector<float> s_vector;
		
		for (k = 0; k < numtests; k++)
		{
			
			std::chrono::high_resolution_clock::time_point worker_start = std::chrono::high_resolution_clock::now();
			results.erase(results.begin(), results.end());
			s_vector.erase(s_vector.begin(), s_vector.end());
	
			MPI::COMM_WORLD.Recv(&param, 1, worker_param_type, 0, 0);
			s_vector.assign(param.s_vector, param.s_vector + 29);
			
			//calculate bounds of data to be processed.
			int numfiles = NUMFILES;
			int number_of_workers = world_size - 1;
			int worker_rank = world_rank - 1;
			int start;
			int end;
			
			
			int chunk = numfiles / number_of_workers;
			int remainder = numfiles - (chunk * number_of_workers);
			
			if (worker_rank < remainder)
			{
				chunk++;
			}
			
			if (worker_rank < remainder)
			{
				start = worker_rank * chunk;
				end = (worker_rank + 1) * chunk;
			}
			else
			{
				start = worker_rank * chunk + remainder;
				end = (worker_rank + 1) * chunk + remainder;
			}
			
			//std::cout << worker_rank << " of " << number_of_workers << " workers has chunk size: " << chunk << " remainder: " << remainder << std::endl;
			
			
			//std::cout << "Process " << world_rank << " start: " << start << " end: " << end << std::endl;
			
			//process data
			for (i = start; i < end; i++)
			{
				//parse line
				lineType lines = parseFile(param.filenames[i]);
				std::vector<result> result_tmp;
				//std::cout << "Process " << world_rank << " parsed file " << param.filenames[i] << std::endl;
				
				#pragma omp parallel for shared(results) private(result_tmp)
				for (j = 0; j < lines.size(); j++)
				{
					//std::cout << "Process " << world_rank << " thread " << omp_get_thread_num() << std::endl; 
					
					result_tmp.erase(result_tmp.begin(), result_tmp.end());
					result_tmp = circularSubvectorMatch(s_vector, lines[j], 0, 360, N, 1);
					sort(result_tmp.begin(), result_tmp.end());
					if (result_tmp.size() > N)
					{
						result_tmp.resize(N);
					}
					
					#pragma omp critical
					{
						results.insert(results.end(), result_tmp.begin(), result_tmp.end());
						sort(results.begin(), results.end());
						if (results.size() > N)
						{
							results.resize(N);
						}
					}	
						
					
				}
			}
			std::chrono::high_resolution_clock::time_point worker_end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time_span_worker = std::chrono::duration_cast<std::chrono::duration<double> >(worker_end - worker_start);
			//std::cout << "Worker " << world_rank << " Time:  " << time_span_worker.count() << " seconds on " << (end - start) << " files." << std::endl;
			
			double timer = time_span_worker.count();
			
			result * returnValue = &results[0];

			MPI::COMM_WORLD.Send(returnValue, N, worker_return_type, 0, 1);
			MPI::COMM_WORLD.Send(&timer, 1, MPI::DOUBLE, 0, 2);
		}
	}
		
	//Finalize MPI
	MPI_Type_free(&worker_param_type);
	MPI_Type_free(&worker_return_type);
	MPI_Finalize();
	
	return 0;	
}
