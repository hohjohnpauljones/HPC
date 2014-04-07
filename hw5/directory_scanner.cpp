/* 
  Sample code that reads a directory that was 
  specified and lists the text files that exists
  
  Purpose: Demonstrate use of Boost:Filesystem and directory scanning for files.
  
  Build: g++ directory_scanner.cpp -o directory_scanner -lboost_filesystem-mt
*/
#include <exception>
#include <stdexcept>
#include <map>
#include <list>
#include <string>
#include <sstream>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>


// +++++++++++++++++++++++++++++++++++++++++++++++++++

#include <boost/filesystem.hpp>

// See the boost documentation for the filesystem
// Especially: http://www.boost.org/doc/libs/1_41_0/libs/filesystem/doc/reference.html#Path-decomposition-table
// Link against boost_filesystem-mt (for multithreaded) or boost_filesystem
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

// +++++++++++++++++++++++++++++++++++++++++++++++++++

namespace scottgs {

// ===== DEFINITIONS =======
typedef std::list<boost::filesystem::path> path_list_type;
std::map<std::string,path_list_type> getFiles(boost::filesystem::path dir);

// ===== IMPLEMENTATIONS =======
std::map<std::string,path_list_type> getFiles(boost::filesystem::path dir)
{
	// Define my map keys
	const std::string regular_file("REGULAR");
	const std::string directory_file("DIRECTORY");
	const std::string other_file("OTHER");
	
	// This is a return object
	// REGULAR -> {file1r,file2r,...,fileNr}
	// DIRECTORY -> {file1d,file2d,...,fileNd}
	// ...
	std::map<std::string,path_list_type> directoryContents;
	
	// Change to the absolute system path, instead of relative
	//boost::filesystem::path dirPath(boost::filesystem::system_complete(dir));
	boost::filesystem::path dirPath(dir);
	
	// Verify existence and directory status
	if ( !boost::filesystem::exists( dirPath ) ) 
	{
		std::stringstream msg;
		msg << "Error: " << dirPath.file_string() << " does not exist " << std::endl;
		throw std::runtime_error(msg.str());
	}
	
	if ( !boost::filesystem::is_directory( dirPath ) ) 
	{
		std::stringstream msg;
		msg << "Error: " << dirPath.file_string() << " is not a directory " << std::endl;
		throw std::runtime_error(msg.str());
	}
/*
#ifdef <strong>GJS_DEBUG_PRINT</strong>			
	std::cout << "Processing directory: " << dirPath.directory_string() << std::endl;
#endif	
*/
	// A director iterator... is just that, 
	// an iterator through a directory... crazy!
	boost::filesystem::directory_iterator end_iter;
	for ( boost::filesystem::directory_iterator dir_itr( dirPath );
		dir_itr != end_iter;
		++dir_itr )
	{
		// Attempt to test file type and push into correct list
		try
		{
			if ( boost::filesystem::is_directory( dir_itr->status() ) )
			{
				// Note, for path the "/" operator is overloaded to append to the path
				directoryContents[directory_file].push_back(dir_itr->path());
#ifdef GJS_DEBUG_PRINT				
				std::cout << dir_itr->path().filename() << " [directory]" << std::endl;
#endif
			}
			else if ( boost::filesystem::is_regular_file( dir_itr->status() ) )
			{
				directoryContents[regular_file].push_back(dir_itr->path());
#ifdef GJS_DEBUG_PRINT				
				std::cout << "Found regular file: " << dir_itr->path().filename() << std::endl;
#endif
			}
			else
			{
				directoryContents[other_file].push_back(dir_itr->path());
#ifdef GJS_DEBUG_PRINT				
				std::cout << dir_itr->path().filename() << " [other]" << std::endl;
#endif
			}

		}
		catch ( const std::exception & ex )
		{
			std::cerr << dir_itr->path().filename() << " " << ex.what() << std::endl;
		}
	}
	
	
	return directoryContents;
}


}; // end scottgs namespace

main (int argc, char * argv[])
{

	int current_worker = 1;
	//char * filenames[500] = "";
	//string filenames[100];
	std::string filename;

	if (argc == 1)
	{
		std::cout << "Usage: " << argv[0] << " <directory> " << std::endl;
		return 1;
	}

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

	//std::cout << "I am Process " << world_rank;

	if (world_rank == 0)
	{
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
			
			std::cout << "Showing: " << f->first << " type files (" << file_list.size() << ")" << std::endl;
			for (scottgs::path_list_type::const_iterator i = file_list.begin();
				i!=file_list.end(); ++i)
			{
				//boost::filesystem::path file_path(boost::filesystem::system_complete(*i));
				boost::filesystem::path file_path(*i);
				//std::cout << "\t" << file_path.file_string() << std::endl;
				//filename = file_path.file_string().data();
				//strcpy(&filename, file_path.file_string().data());
				filename = file_path.file_string();
				MPI_Send(&filename, 500, MPI_CHAR, current_worker, world_rank, MPI_COMM_WORLD);
				current_worker = (current_worker + 1) % world_size;
			}
				
		}
		int ret = 1;
		for (int i = 1; i < world_size; i++)
		{
			MPI_Send(&ret, 1, MPI_INT, i, world_rank, MPI_COMM_WORLD);
		}
	}
	else
	{
		//non blocking recieve for end
		int ret;
		MPI::Request end_request;
		MPI::Request work_request;
		//bool retBool = false;
		//bool workBool = false;
		
		end_request = MPI::COMM_WORLD.Irecv(&ret, 1, MPI::INT, 0, 1);
		work_request = MPI::COMM_WORLD.Irecv(&filename, 500, MPI::CHAR, 0, 0);
		
		//MPI_Irecv(&ret, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &end_request);
		//MPI_Irecv(&filename, 500, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &work_request);
		
		//while end hasn't recieved
		while (!end_request.Test())
		{
			//MPI_Test(&work_request, &workBool, MPI_STATUS_IGNORE);
			
			if (work_request.Test())
			{
				//MPI_Irecv(&filename, 500, MPI_CHAR, 0, 0, MPI_COMM_WORLD, work_request);
				MPI::COMM_WORLD.Irecv(&filename, 500, MPI::CHAR, 0, 0);
    			//printf("Process %d received name %s from process 0\n", world_rank, filename);
    			std::cout << "Process " << world_rank << " recieved name " << filename << " from proces 0\n";
			}
    		//MPI_Test(&end_request, &retBool,  MPI_STATUS_IGNORE);
		}
		
		MPI_Finalize();
		
		return ret;
	}
	
	 MPI_Finalize();
	
	return 0;
}