#include <CL/cl.h>

#include "oclutil.h"

#include <fstream>
#include <iostream>
#include <string>

#include <scottgs/Timing.hpp>

#define DATA_SIZE (1024*1240)

#define OUTPUT_TIMING(msg,seconds) \
std::cout << msg << seconds << " (s) " << std::endl;

void goldenAdd(float *arrA, float *arrB, float* result, unsigned int count);

int main(int argc, char* argv[])
{
	//int devType=CL_DEVICE_TYPE_CPU;
	int devType=CL_DEVICE_TYPE_GPU;

	cl_int err;     // error code returned from api calls

	size_t global;  // global domain size for our calculation
	size_t local;   // local domain size for our calculation

        scottgs::Timing timer;
        timer.start();
        double timerSeconds = 0;

	// -----------------------
	// OpenCL context variables
	// -----------------------
	cl_platform_id cpPlatform; // OpenCL platform
	cl_device_id device_id;    // compute device id
	cl_context context;        // compute context
	cl_command_queue commands; // compute command queue
	cl_program program;        // compute program
	cl_kernel kernel;          // compute kernel

	// Connect to a compute device
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	if (err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to find a platform!" << std::endl;
		return EXIT_FAILURE;
	}


	// Get a device of the appropriate type
	err = clGetDeviceIDs(cpPlatform, devType, 1, &device_id, NULL);
	if (err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to create a device group!" << std::endl;
		return EXIT_FAILURE;
	}
  
	// Create a compute context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context) {
	    std::cerr << "Error: Failed to create a compute context!" << std::endl;
	    return EXIT_FAILURE;
	}
  
	// Create a command commands
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands) 
	{
		std::cerr << "Error: Failed to create a command commands!" << std::endl;
    		return EXIT_FAILURE;
  	}

        // Timing check 
        timerSeconds = timer.getSplitElapsedTime();
        OUTPUT_TIMING("Context Setup Time: ",timerSeconds);
        timer.split();
        
        // Read the kernel source
	size_t kernelLength = 0;
	char *kernelSource = ocltLoadKernelSrc("addVector.cl", &kernelLength);
	
	if (0 == kernelLength)
	{
                std::cerr << "Error: Failed to load kernel from source!" << std::endl;
                return EXIT_FAILURE;
        }

	program = clCreateProgramWithSource(context, 1, 
				(const char **) &kernelSource,
                		&kernelLength, &err);

	if (!program) 
	{
		std::cerr << "Error: Failed to create compute program!" << std::endl;
		return EXIT_FAILURE;
	}
	else
		std::cout << "Program Created from source" << std::endl;
	
	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) 
	{
		size_t len;
		char buffer[2048];

		std::cerr << "Error: Failed to build program executable!" << std::endl;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
				  sizeof(buffer), buffer, &len);
		std::cerr << buffer << std::endl;
		exit(1);
	}

	// Create the compute kernel in the program
	kernel = clCreateKernel(program, "addVector", &err);
	if (!kernel || err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to create compute kernel!" << std::endl;
		exit(1);
	}
	
	// Timing check
	timerSeconds = timer.getSplitElapsedTime();
        OUTPUT_TIMING("Kernel/Program Build Time : ",timerSeconds);
        timer.split();

	// create data for the run
	float* dataA = new float[DATA_SIZE];    // original data set given to device
	float* dataB = new float[DATA_SIZE];    // original data set given to device
	float* results = new float[DATA_SIZE]; 	// results returned from device
	unsigned int correct;               	// number of correct results returned
	cl_mem inputA;                       	// device memory used for the input array
	cl_mem inputB;				// device memory used for the input array
	cl_mem output;                      	// device memory used for the output array

	// Fill the vector with random float values
	unsigned int count = DATA_SIZE;
	for(int i = 0; i < count; i++)
	{
		dataA[i] = random() / (float)RAND_MAX;
		dataB[i] = random() / (float)RAND_MAX;
	}
	// Create the device memory vectors
	//
	inputA = clCreateBuffer(context,  CL_MEM_READ_ONLY,  
			 sizeof(float) * count, NULL, NULL);
	inputB = clCreateBuffer(context,  CL_MEM_READ_ONLY,  
			 sizeof(float) * count, NULL, NULL);
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
			  sizeof(float) * count, NULL, NULL);
	if (!inputA || !inputB || !output) 
	{
		std::cerr << "Error: Failed to allocate device memory!" << std::endl;
		exit(1);
	}   

	// Transfer the input vector into device memory
	err = clEnqueueWriteBuffer(commands, inputA, 
			     CL_TRUE, 0, sizeof(float) * count, 
			     dataA, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, inputB, 
			     CL_TRUE, 0, sizeof(float) * count, 
			     dataB, 0, NULL, NULL);

	if (err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to write to source arrays!" << std::endl;
		exit(1);
	}

	// Timing Check
        timerSeconds = timer.getSplitElapsedTime();
        OUTPUT_TIMING("Memory Allocation/Population Time : ",timerSeconds);
        timer.split();


	// Set the arguments to the compute kernel
	err = 0;
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputA);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputB);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);

	if (err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to set kernel arguments! " << err << std::endl;
		exit(1);
	}

	// Get the maximum work group size for executing the kernel on the device
	err = clGetKernelWorkGroupInfo(kernel, device_id, 
				 CL_KERNEL_WORK_GROUP_SIZE, 
				 sizeof(local), &local, NULL);
	if (err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to retrieve kernel work group info! "
	 	<<  err << std::endl;
		exit(1);
	}

	// Execute the kernel over the vector using the 
	// maximum number of work group items for this device
	global = count;

	// Timer check
        timerSeconds = timer.getSplitElapsedTime();
        OUTPUT_TIMING("Kernel Args / Workgroup Time : ",timerSeconds);
        timer.split();

	err = clEnqueueNDRangeKernel(commands, kernel, 
			       1, NULL, &global, &local, 
			       0, NULL, NULL);
	if (err) 
	{
		std::cerr << "Error: Failed to execute kernel!" << std::endl;
		return EXIT_FAILURE;
	}

	// Wait for all commands to complete
	clFinish(commands);

	// Timing check
        timerSeconds = timer.getSplitElapsedTime();
        OUTPUT_TIMING("Kernel Execution Time : ",timerSeconds);
        timer.split();

	// Read back the results from the device to verify the output
	//
	err = clEnqueueReadBuffer( commands, output,
			     CL_TRUE, 0, sizeof(float) * count,
			     results, 0, NULL, NULL ); 
	if (err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to read output array! " <<  err << std::endl;
		exit(1);
	}

	// Timing check
	timerSeconds = timer.getSplitElapsedTime();
        OUTPUT_TIMING("Output Fetch From Device Time : ",timerSeconds);
        timer.split();

	// Validate our results
	//
	correct = 0;

        float* goldenResult = new float[DATA_SIZE];

        timer.split();
        goldenAdd(dataA, dataB, goldenResult, count);

        // Timing check 
        timerSeconds = timer.getSplitElapsedTime();
        OUTPUT_TIMING("Golden (Naive) Time : ",timerSeconds);
        timer.split();

	for(int i = 0; i < count; i++) {
		if(results[i] == goldenResult[i])
			correct++;
	}

	// Print a brief summary detailing the results
	std::cout << "Computed " << correct << "/" << count << " correct values" << std::endl;
	std::cout << "Computed " << 100.f * (float)correct/(float)count
	<< "% correct values" << std::endl;

	// Shutdown and cleanup
	delete [] dataA;
	delete [] dataB;
	delete [] results;

	clReleaseMemObject(inputA);
	clReleaseMemObject(inputB);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	// Timing check
	timerSeconds = timer.getSplitElapsedTime();
        OUTPUT_TIMING("Total Time : ",timerSeconds);
        timer.split();

	return 0;
}

void goldenAdd(float *arrA, float *arrB, float* result, unsigned int count)
{
	 for (unsigned int i=0; i<count; ++i)
         	result[i] = arrA[i] + arrB[i];
}
