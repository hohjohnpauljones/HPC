#ifndef SCOTTGS_OPENCL_ENGINE
#define 
/*
  OpenCL Engine
*/


namespace scottgs {

class OpenClEngine
{
public:

	OpenClEngine();
	~OpenClEngine();
	
	bool initializeContext(int computeDeviceType);

	

private:
	// -----------------------
	// OpenCL context variables
	// -----------------------
	cl_platform_id cpPlatform; // OpenCL platform
	cl_device_id device_id;    // compute device id
	cl_context context;        // compute context
	cl_command_queue commands; // compute command queue
	cl_program program;        // compute program
	cl_kernel kernel;          // compute kernel

	std::map<std::string,std::string> kernelPrograms;

}; // END: OpenClEngine

}; // END: scottgs namespace

#endif

