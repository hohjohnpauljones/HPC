
#include "hw5.hpp"


int main(int argc, char * argv[])
{
	
	int current_worker = 1;
	int ret = 1;
	//std::string filename[100];
	int filename[100] = {0};
	
	
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
		for (int i = 0; i < 100; i++)
		{
			filename[i] = i + 2;
		}
		
		ret = 0;
		for (int i = 1; i < 100; i++)
		{
			MPI::COMM_WORLD.Isend(&filename[i], 1, MPI::INT, current_worker, 0);
			//MPI::COMM_WORLD.Send(filename[i], 1, MPI::INT, i, 0);
			current_worker = ((current_worker + 1) % world_size);
			if (current_worker == 0)
			current_worker++;
		}
		
		
		//MPI::COMM_WORLD.Scatter(filename, 1, MPI::INT, filename, 1, MPI::INT, 0);
		
		for (int i = 1; i < world_size; i++)
		{
			MPI::COMM_WORLD.Isend(&ret, 1, MPI::INT, i, 1);
			//MPI::COMM_WORLD.Send(&ret, 1, MPI::INT, i, 1);
		}
	}
	//Worker
	else
	{
		MPI::Request end_request;
		MPI::Request work_request;
		
		
		end_request = MPI::COMM_WORLD.Irecv(&ret, 1, MPI::INT, 0, 1);
		work_request = MPI::COMM_WORLD.Irecv(&filename[world_rank], 500, MPI::CHAR, 0, 0);
		
		while(!end_request.Test())
		{
			if (work_request.Test())
			{
				work_request = MPI::COMM_WORLD.Irecv(&filename[world_rank], 500, MPI::CHAR, 0, 0);
				std::cout << "Process " << world_rank << " recieved " << filename[world_rank] << std::endl;
			}
		}
		
		//end_request = MPI::COMM_WORLD.Recv(&ret, 1, MPI::INT, 0, 1);
		//work_request = MPI::COMM_WORLD.Recv(&filename, 500, MPI::CHAR, 0, 0);
		
		//work_request.Wait();
		end_request.Wait();
		
		MPI_Finalize();
		
		return ret;
	}
	
	MPI_Finalize();
	
	return 0;	
}