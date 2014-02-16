#include "csvparse.cpp"
#include "hw2.hpp"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <string>
#include <cstdlib> //std::system


using namespace boost::interprocess;

int main(int argc, char* argv[]) 
{
	if (argc != 2)
	{
		std::cout << "Error incorrect command line arguments. USAGE: ./csvparse <path to file>" << std::endl;
		return 0;
	}
	
	//Define an STL compatible allocator of ints that allocates from the managed_shared_memory.
	//This allocator will allow placing containers in the segment
	typedef allocator<int, managed_shared_memory::segment_manager>  ShmemAllocator;

	//Alias a vector that uses the previous STL-like allocator so that allocates
	//its values from the segment
	typedef vector<int, ShmemAllocator> MyVector;
	
	//Create a new segment with given name and size
	managed_shared_memory segment(create_only, "MySharedMemory", 65536);

	
	//Initialize shared memory STL-compatible allocator
	const ShmemAllocator alloc_inst (segment.get_segment_manager());

	//Construct a vector named "MyVector" in shared memory with argument alloc_inst
	MyVector *myvector = segment.construct<MyVector>("MyVector")(alloc_inst);
	
	
	
	
	//Initialize shared memory STL-compatible allocator
	//const allocator<int, managed_shared_memory::segment_manager> alloc_inst (segment.get_segment_manager());

	//Construct a vector named "MyVector" in shared memory with argument alloc_inst
	//boost::container::vector<float, allocator<float, managed_shared_memory::segment_manager>> * fileData = segment.construct<fileData>("fileData")(alloc_inst);
	
	parseFile(argv[1], fileData);
	std::vector<float>::iterator itr = fileData.begin();
	std::vector<float> svector {9, 8, 7, 6, 5, 4, 3, 2, 1};
	std::vector<result> results;
	
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	
	for (itr; itr !=fileData.end(); ++itr)
	{
		results = (circularSubvectorMatch(svector, itr->first, 9));
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
	
	shared_memory_object::remove("MySharedMemory");
	
	return 0;
}
