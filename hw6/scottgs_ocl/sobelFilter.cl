__kernel void sobelFilter(
	__global float* inputImage,
	__global float* outputImage,
	const unsigned int size)
{

	const unsigned int x = get_global_id(0);
	const unsigned int y = get_global_id(1);

	const int width = get_global_size(0);
	const int height = get_global_size(1);

	// The work-group size may or may not 
	// be a clean sub-block of input/output
	// Check for early exit on last workgroup
	// Also, just process within the image where
	// we have a valid 3x3 neighborhoos
	if (y >= (height-1) || x >= (width-1) || x == 0 || y == 0)
		return ;

	const int yOffset = y * width;
	const int yPrev = yOffset - width;
	const int yNext = yOffset + width;

	// Initialize the 3x3 neighborhood to the center pixel
	// I am packing this into an v8 because the center pixel
	// is never used in Sobel
	const float8 neighborhood = {inputImage[yPrev + x - 1] , inputImage[yPrev + x] , inputImage[yPrev + x + 1],
				 inputImage[yOffset + x - 1]     ,                         inputImage[yOffset + x + 1],
				 inputImage[yNext + x - 1]       , inputImage[yNext + x] , inputImage[yNext + x + 1]};

	// Horizontal and Vertical gradient ip-kernels
	// Like above, packing this into an v8
	const float8 kernelH = { -1,  0,  1, -2, 2, -1, 0, 1 };
	const float8 kernelV = { -1, -2, -1, 0,  0,  1, 2, 1 };

	// Generate the gradient components via convolution
	// which is a vector dot-product now that we have 
	// collected the neighborhood and the kernel into a vector
	// NOTE: the built-in vector ops are for float, float2, 
	// 	and float4 ... only. Therefore, we decouple hi/lo 
	//	and sum the sub-dot-products
	const float gH = dot(kernelH.lo,neighborhood.lo) + dot(kernelH.hi,neighborhood.hi);
	const float gV = dot(kernelV.lo,neighborhood.lo) + dot(kernelV.hi,neighborhood.hi);
	
	// Convert the gradients to a magnitude, 
	outputImage[yOffset + x] = sqrt((gH * gH)  +  (gV * gV));
	
	// alternatively we could compute the 
	// direction with arctan2
	// --> outputImage[yOffset + x] = atan2(gV, gH);	

}
