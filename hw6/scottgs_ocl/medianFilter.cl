__kernel void medianFilter(
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
	if (y >= height || x >= width)
		return ;


	const int yOffset = y * width;
	const int yPrev = yOffset - width;
	const int yNext = yOffset + width;

	float neighborhood[9];

	// Initialize to something that medians to the center pixel
	neighborhood[0] = 0;
	neighborhood[1] = 0;
	neighborhood[2] = 0;
	neighborhood[3] = 0;

	neighborhood[4] = inputImage[yOffset + x];		

	neighborhood[5] = MAXFLOAT;
	neighborhood[6] = MAXFLOAT;
	neighborhood[7] = MAXFLOAT;
	neighborhood[8] = MAXFLOAT;

	//get pixels within aperture
	// If we are not an edge pixel
	if (y>0 && y < (height-1) && x>0 && x < (width-1))
	{
		neighborhood[0] = inputImage[yPrev + x - 1];
		neighborhood[1] = inputImage[yPrev + x];
		neighborhood[2] = inputImage[yPrev + x + 1];

		neighborhood[3] = inputImage[yOffset + x - 1];
		// neighborhood[4] = center pixel, set above
		neighborhood[5] = inputImage[yOffset + x + 1];
		
		neighborhood[7] = inputImage[yNext + x - 1];
		neighborhood[8] = inputImage[yNext + x];
		neighborhood[9] = inputImage[yNext + x + 1];

	}
	else
	{
		// Test for corners
		if (0 == y && 0 == x)
		{	
			// Top-Left Corner Pixel
			neighborhood[3] = inputImage[yOffset + x + 1];
			neighborhood[5] = inputImage[yNext + x];
			neighborhood[6] = inputImage[yNext + x + 1];
		}
		else if (y == (height -1) && x == (width - 1))
		{
			// Bottom-Right Corner Pixel
			neighborhood[3] = inputImage[yOffset + x -1 ];
			neighborhood[5] = inputImage[yPrev + x];
			neighborhood[6] = inputImage[yPrev + x - 1];
		}
		else if (0 == y && x == (width - 1))
		{	
			// Top-Right Corner Pixel
			neighborhood[3] = inputImage[yOffset + x - 1];
			neighborhood[5] = inputImage[yNext + x];
			neighborhood[6] = inputImage[yNext + x - 1];
		}
		else if (y == (height -1) && x == (width - 1))
		{
			// Bottom-Left Corner Pixel
			neighborhood[3] = inputImage[yOffset + x + 1 ];
			neighborhood[5] = inputImage[yPrev + x];
			neighborhood[6] = inputImage[yPrev + x + 1 ];
		}
		// Test for Edges
		else if (0 == x)
		{
			// Left Edge
			neighborhood[2] = inputImage[yPrev + x ];
			neighborhood[3] = inputImage[yNext + x];
			
			neighborhood[5] = inputImage[yPrev + x +1 ];
			neighborhood[6] = inputImage[yOffset + x + 1 ];
			neighborhood[7] = inputImage[yNext + x + 1 ];
		}
		else if (x == (width - 1))
		{

			// Right Edge
			neighborhood[2] = inputImage[yPrev + x ];
			neighborhood[3] = inputImage[yNext + x];
			
			neighborhood[5] = inputImage[yPrev + x - 1 ];
			neighborhood[6] = inputImage[yOffset + x - 1 ];
			neighborhood[7] = inputImage[yNext + x - 1 ];
		}
		else if (0 == y)
		{
			// Top Edge
			neighborhood[2] = inputImage[yOffset + x - 1 ];
			neighborhood[3] = inputImage[yOffset + x + 1];
			
			neighborhood[5] = inputImage[yNext + x - 1 ];
			neighborhood[6] = inputImage[yNext + x ];
			neighborhood[7] = inputImage[yNext + x + 1 ];
		}
		else if (y == (height - 1))
		{
			// Bottom Edge
			neighborhood[2] = inputImage[yOffset + x - 1 ];
			neighborhood[3] = inputImage[yOffset + x + 1];
			
			neighborhood[5] = inputImage[yPrev + x - 1 ];
			neighborhood[6] = inputImage[yPrev + x ];
			neighborhood[7] = inputImage[yPrev + x + 1 ];
		}
	}
	
	// All the pixel values are set, 
	// now we need to sort set the 
	// output to the median
	
	//perform partial bitonic sort to 
	// find median
	
	// Step 1
	float uiMin = fmin(neighborhood[0], neighborhood[1]);
	float uiMax = fmax(neighborhood[0], neighborhood[1]);
	neighborhood[0] = uiMin;
	neighborhood[1] = uiMax;

	uiMin = fmin(neighborhood[3], neighborhood[2]);
	uiMax = fmax(neighborhood[3], neighborhood[2]);
	neighborhood[3] = uiMin;
	neighborhood[2] = uiMax;

	// Step 2
	uiMin = fmin(neighborhood[2], neighborhood[0]);
	uiMax = fmax(neighborhood[2], neighborhood[0]);
	neighborhood[2] = uiMin;
	neighborhood[0] = uiMax;

	uiMin = fmin(neighborhood[3], neighborhood[1]);
	uiMax = fmax(neighborhood[3], neighborhood[1]);
	neighborhood[3] = uiMin;
	neighborhood[1] = uiMax;

	//Step 3
	uiMin = fmin(neighborhood[1], neighborhood[0]);
	uiMax = fmax(neighborhood[1], neighborhood[0]);
	neighborhood[1] = uiMin;
	neighborhood[0] = uiMax;

	uiMin = fmin(neighborhood[3], neighborhood[2]);
	uiMax = fmax(neighborhood[3], neighborhood[2]);
	neighborhood[3] = uiMin;
	neighborhood[2] = uiMax;

	//Step 1
	uiMin = fmin(neighborhood[5], neighborhood[4]);
	uiMax = fmax(neighborhood[5], neighborhood[4]);
	neighborhood[5] = uiMin;
	neighborhood[4] = uiMax;

	uiMin = fmin(neighborhood[7], neighborhood[8]);
	uiMax = fmax(neighborhood[7], neighborhood[8]);
	neighborhood[7] = uiMin;
	neighborhood[8] = uiMax;

	// Step 2
	uiMin = fmin(neighborhood[6], neighborhood[8]);
	uiMax = fmax(neighborhood[6], neighborhood[8]);
	neighborhood[6] = uiMin;
	neighborhood[8] = uiMax;

	// Step 3
	uiMin = fmin(neighborhood[6], neighborhood[7]);
	uiMax = fmax(neighborhood[6], neighborhood[7]);
	neighborhood[6] = uiMin;
	neighborhood[7] = uiMax;

	uiMin = fmin(neighborhood[4], neighborhood[8]);
	uiMax = fmax(neighborhood[4], neighborhood[8]);
	neighborhood[4] = uiMin;
	neighborhood[8] = uiMax;

	// Step 4
	uiMin = fmin(neighborhood[4], neighborhood[6]);
	uiMax = fmax(neighborhood[4], neighborhood[6]);
	neighborhood[4] = uiMin;
	neighborhood[6] = uiMax;

	uiMin = fmin(neighborhood[5], neighborhood[7]);
	uiMax = fmax(neighborhood[5], neighborhood[7]);
	neighborhood[5] = uiMin;
	neighborhood[7] = uiMax;

	// Step 5
	uiMin = fmin(neighborhood[4], neighborhood[5]);
	uiMax = fmax(neighborhood[4], neighborhood[5]);
	neighborhood[4] = uiMin;
	neighborhood[5] = uiMax;

	uiMin = fmin(neighborhood[6], neighborhood[7]);
	uiMax = fmax(neighborhood[6], neighborhood[7]);
	neighborhood[6] = uiMin;
	neighborhood[7] = uiMax;

	uiMin = fmin(neighborhood[0], neighborhood[8]);
	uiMax = fmax(neighborhood[0], neighborhood[8]);
	neighborhood[0] = uiMin;
	neighborhood[8] = uiMax;

	// Step 6
	neighborhood[4] = fmax(neighborhood[0], neighborhood[4]);
	neighborhood[5] = fmax(neighborhood[1], neighborhood[5]);

	neighborhood[6] = fmax(neighborhood[2], neighborhood[6]);
	neighborhood[7] = fmax(neighborhood[3], neighborhood[7]);

	// Step 7
	neighborhood[4] = fmin(neighborhood[4], neighborhood[6]);
	neighborhood[5] = fmin(neighborhood[5], neighborhood[7]);

	// Step 8, store found median into result
	outputImage[yOffset + x] = fmin(neighborhood[4], neighborhood[5]);
}

