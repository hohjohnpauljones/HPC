
#include "hw5.hpp"


int main(int argc, char * argv[])
{
	std::chrono::high_resolution_clock::time_point wall_start = std::chrono::high_resolution_clock::now();
	int current_worker = 1;
	int i, j, k;
	
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
	type_count = 3;		//in
	
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
	
	
	// We are assuming at least 2 processes for this task
	if (world_size < 2) 
	{
		fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1); 
	}
	
	
	//Master
	if (world_rank == 0)
	{
		int n_file = 0;
		
		// Define a template type, and its iterator	
		typedef std::map<std::string,scottgs::path_list_type> content_type;
		typedef content_type::const_iterator content_type_citr;
		
		// Get the file list from the directory
		content_type directoryContents = scottgs::getFiles(argv[1]);
	
		// For each type of file found in the directory, 
		// List all files of that type
		for (content_type_citr f = directoryContents.begin(); 
			f!=directoryContents.end();
			++f)
		{
			const scottgs::path_list_type file_list(f->second);
			
			//std::cout << "Showing: " << f->first << " type files (" << file_list.size() << ")" << std::endl;
			for (scottgs::path_list_type::const_iterator i = file_list.begin();
				i!=file_list.end(); ++i)
			{
				//boost::filesystem::path file_path(boost::filesystem::system_complete(*i));
				boost::filesystem::path file_path(*i);
				//std::cout << "\t" << file_path.file_string() << std::endl;
				//filename = file_path.file_string().data();
				//strcpy(&filename, file_path.file_string().data());
				//filename[n_file] = file_path.file_string();
				strncpy(param.filenames[n_file], file_path.file_string().data(), 60);
				//param.filenames[n_file] = file_path.file_string();
				n_file++;
			}
				
		}
		
		param.N = RESULTSSIZE;
			
		std::vector<float> svector = generateRandomVector(29);
		
		for (j = 0; j < 29; j++)
		{
			param.s_vector[j] = svector[j];
		}
		
		for (current_worker = 1; current_worker < world_size; current_worker++)
		{
			
			MPI::COMM_WORLD.Send(&param, 1, worker_param_type, current_worker, 0);
		}
		
		int ret;
		
		for (int i = 1; i < world_size; i++)
		{
			//MPI::COMM_WORLD.Recv(&ret, 1, MPI::INT, i, 0);
			//MPI::COMM_WORLD.Irecv(&ret[i - 1], 1, MPI::INT, i, 0);
			//MPI::COMM_WORLD.Send(&ret, 1, MPI::INT, i, 1);
		}
		
		std::chrono::high_resolution_clock::time_point wall_end = std::chrono::high_resolution_clock::now();
		
		std::chrono::duration<double> time_span_wall = std::chrono::duration_cast<std::chrono::duration<double> >(wall_end - wall_start);
		std::cout << "Wall Time:  " << time_span_wall.count() << " seconds." << std::endl;
	}
	//Worker
	else
	{
		std::chrono::high_resolution_clock::time_point worker_start = std::chrono::high_resolution_clock::now();
		std::vector<result> results;
		//results.erase(results.begin(), results.end());
		int N = RESULTSSIZE;
		std::vector<float> s_vector;
		MPI::COMM_WORLD.Recv(&param, 1, worker_param_type, 0, 0);
		
		//calculate bounds of data to be processed.
		int numfiles = NUMFILES;
		s_vector.assign(param.s_vector, param.s_vector + 29);
		
		int offset = numfiles / (world_size - 1); 
		int start = offset * (world_rank - 1);
		int end = start + offset;
		
		if (world_rank == world_size - 1)
		{
			end += numfiles % (world_size - 1);	
		}
		
		//process data
		
		for (i = start; i < end; i++)
		//for (i = 0; i < 1; i++)
		{
			//if (world_rank == 1)
			{
				//std::cout << "Process " << world_rank << " recieved " << i << std::endl;
				
				//parse line
				lineType lines = parseFile(param.filenames[i]);
				std::vector<result> result_tmp;
				//std::cout << "Process " << world_rank << " parsed file " << param.filenames[i] << std::endl;
				
				#pragma omp parallel for shared(results) private(j, k, result_tmp)
				for (j = 0; j < lines.size(); j++)
				{
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
					
					
					//for(k = 0; k < result_tmp.size(); k++)
					{
						//std::cout << "\t" << "line " << j << " result " << k << ": " << "(" << result_tmp[k].x << ", " << result_tmp[k].y << ") => " << result_tmp[k].distance << std::endl;
					}
				}
			}
		}
		/*
		std::cout << "Process " << world_rank << " Result set:" << std::endl;
		for(k = 0; k < results.size(); k++)
		{
			std::cout << "\t" << k << ": " << "(" << results[k].x << ", " << results[k].y << ") => " << results[k].distance << std::endl;
		}
		*/
		std::chrono::high_resolution_clock::time_point worker_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_span_worker = std::chrono::duration_cast<std::chrono::duration<double> >(worker_end - worker_start);
		std::cout << "Worker " << world_rank << " Time:  " << time_span_worker.count() << " seconds." << std::endl;
		int ret = 1;
		//MPI::COMM_WORLD.Send(&ret, 1, MPI::INT, current_worker, 0);
	}
	
	//Finalize MPI
	MPI_Type_free(&worker_param_type);
	MPI_Finalize();
	
	return 0;	
}