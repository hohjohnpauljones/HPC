
#include "hw5.hpp"


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
		int n_file = 0;
		
		int ret;
		int N = RESULTSSIZE;
		result worker_results[world_size][N];
		
		for (int i = 1; i < world_size; i++)
		{
			//MPI::COMM_WORLD.Recv(&ret, 1, MPI::INT, i, 1);
			//MPI::COMM_WORLD.Recv(&worker_results[i][0], 1, worker_return_type, i, 1);
			//std::cout << "Recieved worker " << i << " results: " << worker_results[i][0].x << ", " << worker_results[i][0].y << " - " << worker_results[i][0].offset << " => " << worker_results[i][0].distance << std::endl;
			MPI::COMM_WORLD.Recv(&worker_results[i][0], N, worker_return_type, i, 1);
			//std::cout << "Recieved worker " << i << " results: " << worker_results[i][0].x << ", " << worker_results[i][0].y << " - " << worker_results[i][0].offset << " => " << worker_results[i][0].distance << std::endl;
			//std::cout << "Recieved worker " << i << " results: " << worker_results[i][1].x << ", " << worker_results[i][1].y << " - " << worker_results[i][1].offset << " => " << worker_results[i][1].distance << std::endl;
			//MPI::COMM_WORLD.Recv(&times[i], 1, MPI::DOUBLE, i , 2);
			//std::cout << "Node " << i << " took " << times[i] << " seconds" << std::endl;
			//MPI::COMM_WORLD.Irecv(&ret[i - 1], 1, MPI::INT, i, 0);
			//MPI::COMM_WORLD.Send(&ret, 1, MPI::INT, i, 1);
		}

		std::vector<result> g_results;
		
		for (int i = 1; i < world_size; i++)
		{
			for (j = 0; j < N; j++)
			{
				result tmp = worker_results[i][j];
				std::cout << worker_results[i][j].x << ", " << worker_results[i][j].y << " - " << worker_results[i][j].offset << " => " << worker_results[i][j].distance << std::endl;
				//g_results.push_back(tmp);
			}
		}
		
		g_results.assign(&worker_results[1][0], &worker_results[1][0] + (world_size - 1) * N);
		
		std::cout << "Global results of size: " << g_results.size() << std::endl;
		
		for (k = 0; k < g_results.size(); k++)
		{
			std::cout << "\t" << k << ": " << "(" << g_results[k].x << ", " << g_results[k].y << ") => " << g_results[k].distance << std::endl;
		}
		
		
		sort(g_results.begin(), g_results.end());
		if (g_results.size() > N)
		{
			g_results.resize(N);
		}

		std::cout << "Global results of size: " << g_results.size() << std::endl;
		
		for (k = 0; k < g_results.size(); k++)
		{
			std::cout << "\t" << k << ": " << "(" << g_results[k].x << ", " << g_results[k].y << ") => " << g_results[k].distance << std::endl;
		}

	}
	//Worker
	else
	{
		std::vector<result> results;
		//results.erase(results.begin(), results.end());
		int N = RESULTSSIZE;
		
		for (i = 0; i < N; i++)
		{
			result tmp;
			tmp.x = i * world_rank;
			tmp.y = i * world_rank + .1;
			tmp.offset = i * world_rank + .2;
			tmp.distance = i * world_rank + .3;
			results.push_back(tmp);
		}
		
		//result * returnValue = &results[0];
		
		MPI::COMM_WORLD.Send(&results[0], N, worker_return_type, 0, 1);
	}
		
	//Finalize MPI
	MPI_Type_free(&worker_param_type);
	MPI_Type_free(&worker_return_type);
	MPI_Finalize();
	
	return 0;	
}