
//#include "hw5.hpp"

#include <math.h>
#include <stdio.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <exception>
#include <stdexcept>
#include <map>
#include <list>
#include <string>
#include <omp.h>

#include <string.h>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <chrono>

#include <boost/filesystem.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"


int main(int argc, char * argv[])
{
	
	int current_worker = 1;
	int i, j, k;
	
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	// Find out rank, size
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	// We are assuming at least 2 processes for this task
	if (world_size < 2) 
	{
		fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1); 
	}
	
	
	//Master
	if (world_rank == 0)
	{
		/*int n_file = 0;
		
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
		}*/
		
		for (current_worker = 1; current_worker < world_size; current_worker++)
		{
			
			//MPI::COMM_WORLD.Send(&param, 1, worker_param_type, current_worker, 0);
			MPI::COMM_WORLD.Send(&current_worker, 1, MPI::INT, current_worker, 0);
		}
		
		for (int i = 1; i < world_size; i++)
		{
			//MPI::COMM_WORLD.Irecv(&ret[i - 1], 1, MPI::INT, i, 0);
			//MPI::COMM_WORLD.Send(&ret, 1, MPI::INT, i, 1);
		}
	}
	//Worker
	else
	{
		int current_worker;
		
		MPI::COMM_WORLD.Recv(&current_worker, 1, MPI::INT, 0, 0);
		
		#pragma omp parallel for
		for (i = 0; i < 10; i++)
		{
			std::cout << "Process " << world_rank << " thread " << omp_get_thread_num() << " recieved " << i + current_worker << std::endl; 
		}
		/*
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
				std::cout << "Process " << world_rank << " parsed file " << param.filenames[i] << std::endl;
				
				#pragma omp parallel for private(result_tmp)
				for (j = 0; j < lines.size(); j++)
				{
					result_tmp.erase(result_tmp.begin(), result_tmp.end());
					result_tmp = circularSubvectorMatch(s_vector, lines[j], 0, 360, N, 1);
					sort(result_tmp.begin(), result_tmp.end());
					if (result_tmp.size() > N)
					{
						result_tmp.resize(N);
					}
					
					results.insert(results.end(), result_tmp.begin(), result_tmp.end());
					sort(results.begin(), results.end());
					if (results.size() > N)
					{
						results.resize(N);
					}
						
					
					
					for(k = 0; k < result_tmp.size(); k++)
					{
						//std::cout << "\t" << "line " << j << " result " << k << ": " << "(" << result_tmp[k].x << ", " << result_tmp[k].y << ") => " << result_tmp[k].distance << std::endl;
					}
				}
			}
		}
		std::cout << "Process " << world_rank << " Result set:" << std::endl;
		for(k = 0; k < results.size(); k++)
		{
			std::cout << "\t" << k << ": " << "(" << results[k].x << ", " << results[k].y << ") => " << results[k].distance << std::endl;
		}
		*/
	}
		
		
		
		
		
		
		
		
		//merge results
		
		//return results
		//MPI::COMM_WORLD.Send(&filename[end], 1, MPI::INT, 0, world_rank);
		
		/*
		for (i = 0; i < 29; i++)
		{
			if (world_rank == 1)
			std::cout << "Process " << world_rank << " recieved " << param.s_vector[i] << std::endl;
		}
		*/
		/*
		if (world_rank == 1)
		{
			for (i = 0; i < NUMFILES; i++)
			{
			//param.filenames[i] = "filename " + i;
			//strncpy(param.filenames[i], "filename " + i, 20);
			cout << "Process " << world_rank << " recieved " << param.filenames[i] << std::endl;
			//printf("Process %d recieved %60s\n", world_rank, param.filenames[i]);
			}
		}
		*/
		
	
	//Finalize MPI
	//MPI_Type_free(&worker_param_type);
	MPI_Finalize();
	
	return 0;	
}