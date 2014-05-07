__kernel void noFilter(
	__global float* inputImage,
	__global float* outputImage,
	const unsigned int size)
{

	const unsigned int x = get_global_id(0);
	const unsigned int y = get_global_id(1);

	const int width = get_global_size(0);

	// The work-group size may or may not 
	// be a clean sub-block of input/output
	// Check for early exit on last workgroup
	
	if (y >= height || x >= width)
		return ;
	
	const int yOffset = y * width;

	outputImage[yOffset + x] = inputImage[yOffset + x];		

}

