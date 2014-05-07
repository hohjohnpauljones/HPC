/*
  OpenCL Main program for an image processing
  example, Median filter
  
  NOTE: Much of the host code is done with
  Open Computer Vision library (OpenCV)
  
*/

#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>

#include <CL/cl.h>

#include <boost/filesystem.hpp>
#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <scottgs/Timing.hpp>

// Get the scottgs::kernelSelector object
#include "kernel_selections.hpp"

// Get the OpenCL utility functions
#include "oclutil.h"

#define OUTPUT_TIMING(msg,seconds) \
std::cout << msg << seconds << " (s) " << std::endl;


bool setupOpenClContext(cl_platform_id *cpPlatform, cl_device_id *device_id, cl_context *context, 
			cl_command_queue *commands, int devType=CL_DEVICE_TYPE_GPU);
			
bool setupOpenClProgram(const std::string& kernelFunction, cl_program *program, 
			cl_kernel *kernel, cl_context *context, cl_device_id *device_id);
			
void freeOpenClContext(cl_context *context, cl_command_queue *commands);
void freeOpenClKernelProgram(cl_program *program, cl_kernel *kernel);
			
int main(int argc, char *argv[])
{
	// ============================================
	// --------------------------------------------
	// Check and Validate the arguments
	// --------------------------------------------
	if (argc < 3) 
	{
		std::cerr << "Usage: " << argv[0] << " kernelSector inputFile" << std::endl;
		return 1;
	}
	
	const std::string kernelSelector(argv[1]);
	const std::string inputFile(argv[2]);

	if (!boost::filesystem::exists(inputFile))
	{
		std::cerr << "Usage: " << argv[0] << " kernelSector inputFile" << std::endl;
		std::cerr << inputFile << " not found" << std::endl;
		return 2;
	}
	

	std::cout << "Input File: " << inputFile << std::endl 
		  << "KernelFunction: " << kernelSelector << std::endl;

	// ============================================
	// --------------------------------------------
	// The general Flow:
	//  1) Load the portable network graphics (PNG) 
	//	into a CV Mat class
	//  2) Push to compute device memory
	//  3) Compute Median Filter response on
	//	device
	//  3) Pull from compute device memory
	//  4) Verify against OpenCV
	// --------------------------------------------
	
	//scottgs::Timing timer;
	//timer.start();
	double timerSeconds=0;//timer.getTotalElapsedTime();
	
	// see: http://opencv.willowgarage.com/documentation/cpp/reading_and_writing_images_and_video.html#cv-imread
	cv::Mat imageData(cv::imread(inputFile.c_str(), -1 )); 	// Zero flag for grayscale, but that truncates to 1-byte
								// Using -1 does an "as-is" read of the data... 
	
	// Documentation : http://opencv.itseez.com/doc/tutorials/core/mat_the_basic_image_container/mat_the_basic_image_container.html#matthebasicimagecontainer
	//		 : http://opencv.itseez.com/modules/core/doc/basic_structures.html#mat
	const int mFlags = imageData.flags;
	const int mDims = imageData.dims ;
	const int mRows = imageData.rows;
	const int mCols = imageData.cols;
	const int imgPixels = mRows * mCols;	// assumes single band 
	
	// Mat metadata
	std::cout << "Matrix header data: " << std::endl 
		  << "Flags: " << mFlags << std::endl
		  << "Dims: " << mDims << std::endl
		  << "Rows: " << mRows << std::endl
		  << "Cols: " << mCols << std::endl
		  << "Total Pixels: " << imgPixels << std::endl
		  << "Contigous: " << imageData.isContinuous() << std::endl
		  << "Element Size (bytes): " << imageData.elemSize() << std::endl;
		
	if (imgPixels < 1)
	{
		std::cerr << "The read in of the input is empty!" << std::endl;
		return 4;
	}
	
	// Translate the as-is image data to unsigned char on the host
	cv::Mat imageDataFloat(mRows, mCols, CV_32FC1);	// CV_32FC1 is an 32-bit float, singe channel matrix
	imageData.assignTo(imageDataFloat, CV_32FC1);		// Convert from input to desired (we can also use convertTo)
	
		
	// verification that data is not zero
	cv::Mat zeros = cv::Mat::zeros(mRows, mCols, CV_32FC1);
	if (std::equal(imageDataFloat.begin<float>(),
			imageDataFloat.end<float>(),
			zeros.begin<float>())
		)
	{
			std::cerr << "The float copy of the input is zero!" << std::endl;
			return 4;
	}
	else
		std::cout << "The float copy is verified to be not all zero!" << std::endl;

	// Timing check
	timerSeconds = 0;//timer.getSplitElapsedTime();
        OUTPUT_TIMING("Image Data Read Time : ",timerSeconds);
        //timer.split();

	// -----------------------
	// OpenCL context variables
	// -----------------------
	cl_int err;     		// error code returned from api calls
	cl_platform_id cpPlatform;	// OpenCL platform
	cl_device_id device_id;		// compute device id
	cl_context context;		// compute context
	cl_command_queue commands;	// compute command queue
	cl_program program;		// compute program
	cl_kernel kernel;		// compute kernel

	// Setup Context of execution, except kernel and program
	if (!setupOpenClContext(&cpPlatform, &device_id, &context, &commands, CL_DEVICE_TYPE_GPU)) // use a GPU
	{
		std::cerr << "ERROR: The Failed to setup the OpenCL Context" << std::endl;
		return 1;
	}
			
	// Timing check 
        timerSeconds = 0;//timer.getSplitElapsedTime();
        OUTPUT_TIMING("Context Setup Time: ",timerSeconds);
        //timer.split();

	// Setup kernel program
	if (!setupOpenClProgram(kernelSelector, &program, &kernel, &context, &device_id))
	{
		std::cerr << "ERROR: The Failed to setup the OpenCL Program/Kernel" << std::endl;
		freeOpenClContext(&context, &commands);
		return 1;
	}

	// Timing check
	timerSeconds = 0;//timer.getSplitElapsedTime();
        OUTPUT_TIMING("Kernel/Program Build Time : ",timerSeconds);
        //timer.split();

	// Allocate device memories
	cl_mem inputImage;			// device memory used for the input array
	cl_mem outputImage;                    	// device memory used for the output array

	// Create the device memory vectors
	//
	inputImage = clCreateBuffer(context,  CL_MEM_READ_ONLY,  
			 sizeof(float) * imgPixels, NULL, NULL);
	outputImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
			  sizeof(float) * imgPixels, NULL, NULL);
	if (!inputImage || !outputImage) 
	{
		std::cerr << "Error: Failed to allocate device memory!" << std::endl;
		freeOpenClContext(&context, &commands);
		freeOpenClKernelProgram(&program, &kernel);
		return 1;	
	}   

	// If data is still continous, we can push 
	// it to straight to device memory, else we
	// will need a loop structure over the rows
	//	(an exercise for reader)
	if (imageDataFloat.isContinuous()) 
	{
		// The location of the data
		const float* internalImageDataFloatBuffer = imageDataFloat.ptr<float>(0);	

		// Transfer the input vector into device memory
		err = clEnqueueWriteBuffer(commands, inputImage, 
			     CL_TRUE, 0, sizeof(float) * imgPixels, 
			     internalImageDataFloatBuffer, 0, NULL, NULL);
		if (err != CL_SUCCESS) 
		{
			std::cerr << "Failed to copy the image data to the device" << std::endl;
			
			clReleaseMemObject(inputImage);
			clReleaseMemObject(outputImage);

			freeOpenClContext(&context, &commands);
			freeOpenClKernelProgram(&program, &kernel);

			return 1;
		}
		
	}
	else // Non-continous data, need to loop through and do mem-copy per row
	{
		std::cerr << "ERROR: The data is not continous after assignment from USHORT to FLOAT" << std::endl;
		
		clReleaseMemObject(inputImage);
		clReleaseMemObject(outputImage);

		freeOpenClContext(&context, &commands);
		freeOpenClKernelProgram(&program, &kernel);

		return 1;
	}
		
	// Timing Check
        timerSeconds = 0;//timer.getSplitElapsedTime();
        OUTPUT_TIMING("Memory Allocation/Population Time : ",timerSeconds);
        //timer.split();

	// Set the arguments to the compute kernel
	err = 0;
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);
	err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &imgPixels);

	if (err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to set kernel arguments! " << err << std::endl;

		clReleaseMemObject(inputImage);
		clReleaseMemObject(outputImage);

		freeOpenClContext(&context, &commands);
		freeOpenClKernelProgram(&program, &kernel);

		return 1;
	}

	// Get the maximum work group size for executing the kernel on the device
	size_t maxLocalSize;
	err = clGetKernelWorkGroupInfo(kernel, device_id, 
				 CL_KERNEL_WORK_GROUP_SIZE, 
				 sizeof(maxLocalSize), &maxLocalSize, NULL);
	if (err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to retrieve kernel work group info! "
	 	<<  err << std::endl;

		clReleaseMemObject(inputImage);
		clReleaseMemObject(outputImage);

		freeOpenClContext(&context, &commands);
		freeOpenClKernelProgram(&program, &kernel);

		return 1;
	}

	if (64 > maxLocalSize)
	{
		std::cerr << "Error: Device Too Puny, cannot support an 8x8 work-group! "
	 	<<  err << std::endl;

		clReleaseMemObject(inputImage);
		clReleaseMemObject(outputImage);

		freeOpenClContext(&context, &commands);
		freeOpenClKernelProgram(&program, &kernel);

		return 1;	
	}
	
	size_t global[2] = { mCols, mRows};
	size_t local[2]= { 8, 8}; // This results in 8x8 = 64 work-items per work-group


	std::cout << "Global domain, aka NDRange = (" << global[0] << "x" << global[1] << ")" << std::endl
		  << "Work-group dimensions = (" << local[0] << "x" << local[1] << ")" << std::endl;
		
	// Timer check
        timerSeconds = 0;//timer.getSplitElapsedTime();
        OUTPUT_TIMING("Kernel Args / Workgroup Time : ",timerSeconds);
        //timer.split();

	// Execute the kernel over the vector using the 
	// maximum number of work group items for this device
	err = clEnqueueNDRangeKernel(commands, kernel, 
			       2, NULL, global, local,  // note the 2 for 2 dimensions
			       0, NULL, NULL);
			       
	if (err) 
	{
		std::cerr << "Error: Failed to execute kernel!" << err << std::endl;

		clReleaseMemObject(inputImage);
		clReleaseMemObject(outputImage);

		freeOpenClContext(&context, &commands);
		freeOpenClKernelProgram(&program, &kernel);

		return 1;
	}

	// Wait for all commands to complete
	clFinish(commands);

	// Timing check
        timerSeconds = 0;//timer.getSplitElapsedTime();
        OUTPUT_TIMING("Kernel Execution Time : ",timerSeconds);
        //timer.split();


	// Read back the results from the device to verify the output
	//
	float *imageDataBuffer_postFilter = new float[imgPixels];	// Allocate additional host memory
	err = clEnqueueReadBuffer( commands, outputImage,
		     CL_TRUE, 0, sizeof(float) * imgPixels,
		     imageDataBuffer_postFilter, 0, NULL, NULL ); 

	if (err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to read output array! " <<  err << std::endl;

		delete imageDataBuffer_postFilter;

		clReleaseMemObject(inputImage);
		clReleaseMemObject(outputImage);

		freeOpenClContext(&context, &commands);
		freeOpenClKernelProgram(&program, &kernel);

		return 1;
	}

	// Timing check
	timerSeconds = 0;//timer.getSplitElapsedTime();
        OUTPUT_TIMING("Results Read From Device Time : ",timerSeconds);
        //timer.split();
	
	// Write out the result
	cv::Mat outputMatCatcher(mRows, mCols, CV_32FC1, imageDataBuffer_postFilter);
	cv::Mat outputMatWritten(mRows, mCols, CV_16UC1);
	outputMatCatcher.convertTo(outputMatWritten, CV_16UC1);

	std::string outputFile(inputFile + "_" + kernelSelector + ".png");
	if (boost::filesystem::exists(outputFile))
	{
		std::cout << "Removing existing file: " << outputFile << std::endl;
		boost::filesystem::remove(outputFile);
		
	}
	
	if (!cv::imwrite(outputFile.c_str(), outputMatWritten))
		std::cerr << "Error writing test output" << std::endl;
	else
		std::cout << "Wrote output: " << outputFile << std::endl;
		
		// Timing check
	timerSeconds = 0;//timer.getSplitElapsedTime();
        OUTPUT_TIMING("Result File Write Time : ",timerSeconds);
        //timer.split();

	delete imageDataBuffer_postFilter;


	// Timing check
	timerSeconds = 0;//timer.getSplitElapsedTime();
        OUTPUT_TIMING("Total Time : ",timerSeconds);
        //timer.split();
	

	clReleaseMemObject(inputImage);
	clReleaseMemObject(outputImage);

	freeOpenClContext(&context, &commands);
	freeOpenClKernelProgram(&program, &kernel);

	return 0;
}

bool setupOpenClContext(cl_platform_id *cpPlatform, cl_device_id *device_id,
			cl_context *context, cl_command_queue *commands, int devType)
{
	cl_int err;     // error code returned from api calls

	// Connect to a compute device
	err = clGetPlatformIDs(1, cpPlatform, NULL);
	if (err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to find a platform!" << std::endl;
		return false;
	}


	// Get a device of the appropriate type
	err = clGetDeviceIDs(*cpPlatform, devType, 1, device_id, NULL);
	if (err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to create a device group!" << std::endl;
		return false;
	}
  
	// Create a compute context
	*context = clCreateContext(0, 1, device_id, NULL, NULL, &err);
	if (!context) {
	    std::cerr << "Error: Failed to create a compute context!" << std::endl;
	    return false;
	}
  
	// Create a command commands
	*commands = clCreateCommandQueue(*context, *device_id, 0, &err);
	if (!commands) 
	{
		std::cerr << "Error: Failed to create a command commands!" << std::endl;
    		return false;
  	}
  	
  	return true;
}	

bool setupOpenClProgram(const std::string& kernelFunction, cl_program *program, cl_kernel *kernel, 
			cl_context *context, cl_device_id *device_id)
{
	// Variable that becomes globally visible
	scottgs::KernelSelections kernelSelector;
	// Populatd the global
	kernelSelector.addKernelSourceSelection("medianFilter", "medianFilter.cl");
	kernelSelector.addKernelSourceSelection("noFilter", "noFilter.cl");
	kernelSelector.addKernelSourceSelection("sobelFilter", "sobelFilter.cl");


	cl_int err;     // error code returned from api calls
	
	// Lookup the source file for the kernel function
	const std::string kernelSourceFile(kernelSelector.getKernelSourceFile(kernelFunction));

        // Read the kernel source
	size_t kernelLength = 0;
	char *kernelSource = ocltLoadKernelSrc(kernelSourceFile.c_str(), &kernelLength);
	
	if (0 == kernelLength)
	{
                std::cerr << "Error: Failed to load kernel from source!" << oclErrorString(err) << std::endl;
                return EXIT_FAILURE;
        }

	*program = clCreateProgramWithSource(*context, 1, 
				(const char **) &kernelSource,
                		&kernelLength, &err);

	if (!(*program)) 
	{
		std::cerr << "Error: Failed to create compute program!" << oclErrorString(err) << std::endl;
		return false;
	}
	else
		std::cout << "Program Created from source" << std::endl;
	
	// Build the program executable
	err = clBuildProgram(*program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) 
	{
		size_t len;
		char buffer[2048];

		std::cerr << "Error: Failed to build program executable!" << oclErrorString(err) << std::endl;
		clGetProgramBuildInfo(*program, *device_id, CL_PROGRAM_BUILD_LOG,
				  sizeof(buffer), buffer, &len);
		std::cerr << buffer << std::endl;
		return false;
	}

	// Create the compute kernel in the program
	*kernel = clCreateKernel(*program, kernelFunction.c_str(), &err);
	if (!(*kernel) || err != CL_SUCCESS) 
	{
		std::cerr << "Error: Failed to create compute kernel!" << oclErrorString(err) << std::endl;
		return false;
	}

	return true;
}

void freeOpenClContext(cl_context *context, cl_command_queue *commands)
{
	clReleaseCommandQueue(*commands);
	clReleaseContext(*context);
}

void freeOpenClKernelProgram(cl_program *program, cl_kernel *kernel)
{
	clReleaseProgram(*program);
	clReleaseKernel(*kernel);

}
