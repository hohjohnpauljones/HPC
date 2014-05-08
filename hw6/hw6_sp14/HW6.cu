#include <iostream>
#include <vector>

#include <cstdio>
#include <cstdlib>

typedef unsigned char uint8_t;

/* This function swaps two numbers
   Arguments :
			 a, b - the numbers to be swapped
   */
__device__ void swap(uint8_t &a, uint8_t &b)
{
	int temp;
	temp = a;
	a = b;
	b = temp;
}

/* This function splits the array around the pivot
   Arguments :
			 array - the array to be split
			 pivot - pivot element whose position will be returned
			 startIndex - index of the first element of the section
			 endIndex - index of the last element of the section
   Returns :
		   the position of the pivot
   */
__device__ int SplitArray(uint8_t* array, int pivot, int startIndex, int endIndex)
{
	int leftBoundary = startIndex;
	int rightBoundary = endIndex;
	
	while(leftBoundary < rightBoundary)			   //shuttle pivot until the boundaries meet
	{
		 while( pivot < array[rightBoundary]		  //keep moving until a lesser element is found
				&& rightBoundary > leftBoundary)	  //or until the leftBoundary is reached
		 {
			  rightBoundary--;						//move left
		 }
		 swap(array[leftBoundary], array[rightBoundary]);
		 
		 while( pivot >= array[leftBoundary]		  //keep moving until a greater or equal element is found
				&& leftBoundary < rightBoundary)	  //or until the rightBoundary is reached
		 {
			  leftBoundary++;						 //move right
		 }
		 swap(array[rightBoundary], array[leftBoundary]);
	}
	return leftBoundary;
}

/* This function does the quicksort
   Arguments :
			 array - the array to be sorted
			 startIndex - index of the first element of the section
			 endIndex - index of the last element of the section
   */
__device__ void QuickSort(uint8_t* array, int startIndex, int endIndex)
{
	int pivot = array[startIndex];	//pivot element is the leftmost element
	int splitPoint;
	
	if(endIndex > startIndex)
	{
		splitPoint = SplitArray(array, pivot, startIndex, endIndex);
		array[splitPoint] = pivot;
		QuickSort(array, startIndex, splitPoint-1);   //Quick sort first half
		QuickSort(array, splitPoint+1, endIndex);	 //Quick sort second half
	}
}

__global__ void medianFilter3( uint8_t *d_input, uint8_t *d_output) {
        // map from threadIdx/BlockIdx to pixel position^M
        int x = blockIdx.x;
        int y = blockIdx.y;
        int dim = 3;

	const int yOffset = y * gridDim.x;
	const int yPrev = yOffset - gridDim.x;
	const int yNext = yOffset + gridDim.x;
	
	uint8_t neighborhood[9];
	
	
	if (y > 0 && y < (gridDim.y - 1) && x > 0 && x < (gridDim.x - 1))
	{

        	neighborhood[0] = d_input[yPrev + x - 1];
        	neighborhood[1] = d_input[yPrev + x];
        	neighborhood[2] = d_input[yPrev + x + 1];
        	neighborhood[3] = d_input[yOffset + x - 1];

        	neighborhood[4] = d_input[yOffset + x];

        	neighborhood[5] = d_input[yOffset + x + 1];
        	neighborhood[6] = d_input[yNext + x - 1];
        	neighborhood[7] = d_input[yNext + x];
        	neighborhood[8] = d_input[yNext + x + 1];
	}
	else
	{
		neighborhood[0] = 0;
                neighborhood[1] = 0;
                neighborhood[2] = 0;
                neighborhood[3] = 0;

                neighborhood[4] = d_input[yOffset + x];

                neighborhood[5] = 255;
                neighborhood[6] = 255;
                neighborhood[7] = 255;
                neighborhood[8] = 255;
	}

	//sort neighborhood
	QuickSort(neighborhood, 0, 9);
	
	// assign pixel to median

	d_output[yOffset + x] = neighborhood[5];

}

__global__ void medianFilter7( uint8_t *d_input, uint8_t *d_output) {
        // map from threadIdx/BlockIdx to pixel position^M
        int x = blockIdx.x;
        int y = blockIdx.y;
        int dim = 7;

	const int yOffset = y * gridDim.x;
	const int yPrev = yOffset - gridDim.x;
	const int yNext = yOffset + gridDim.x;
	
	int yOffsets[7];
	
	yOffsets[0] = yOffset - gridDim.x * 3;
	yOffsets[1] = yOffset - gridDim.x * 2;
	yOffsets[2] = yOffset - gridDim.x * 1;
	yOffsets[3] = yOffset ;
	yOffsets[4] = yOffset + gridDim.x * 1;
	yOffsets[5] = yOffset + gridDim.x * 2;
	yOffsets[6] = yOffset + gridDim.x * 3;
	
	uint8_t neighborhood[7 * 7];
	
	
	if (y > 0 && y < (gridDim.y - 1) && x > 0 && x < (gridDim.x - 1))
	{

        	neighborhood[0] = d_input[yOffsets[0] + x - 3];
        	neighborhood[1] = d_input[yOffsets[0] + x - 2];
        	neighborhood[2] = d_input[yOffsets[0] + x - 1];
        	neighborhood[3] = d_input[yOffsets[0] + x - 0];
        	neighborhood[4] = d_input[yOffsets[0] + x + 1];
        	neighborhood[5] = d_input[yOffsets[0] + x + 2];
        	neighborhood[6] = d_input[yOffsets[0] + x + 3];
        	
		neighborhood[7] = d_input[yOffsets[1] + x - 3];
        	neighborhood[8] = d_input[yOffsets[1] + x - 2];
        	neighborhood[9] = d_input[yOffsets[1] + x - 1];
        	neighborhood[10] = d_input[yOffsets[1] + x - 0];
        	neighborhood[11] = d_input[yOffsets[1] + x + 1];
        	neighborhood[12] = d_input[yOffsets[1] + x + 2];
        	neighborhood[13] = d_input[yOffsets[1] + x + 3];
        	
        	neighborhood[14] = d_input[yOffsets[2] + x - 3];
        	neighborhood[15] = d_input[yOffsets[2] + x - 2];
        	neighborhood[16] = d_input[yOffsets[2] + x - 1];
        	neighborhood[17] = d_input[yOffsets[2] + x - 0];
        	neighborhood[18] = d_input[yOffsets[2] + x + 1];
        	neighborhood[19] = d_input[yOffsets[2] + x + 2];
        	neighborhood[20] = d_input[yOffsets[2] + x + 3];
        	
        	neighborhood[21] = d_input[yOffsets[3] + x - 3];
        	neighborhood[22] = d_input[yOffsets[3] + x - 2];
        	neighborhood[23] = d_input[yOffsets[3] + x - 1];
        	
        	neighborhood[24] = d_input[yOffsets[3] + x - 0];
        	
        	neighborhood[25] = d_input[yOffsets[3] + x + 1];
        	neighborhood[26] = d_input[yOffsets[3] + x + 2];
        	neighborhood[27] = d_input[yOffsets[3] + x + 3];
        	
        	neighborhood[28] = d_input[yOffsets[4] + x - 3];
        	neighborhood[29] = d_input[yOffsets[4] + x - 2];
        	neighborhood[30] = d_input[yOffsets[4] + x - 1];
        	neighborhood[31] = d_input[yOffsets[4] + x - 0];
        	neighborhood[32] = d_input[yOffsets[4] + x + 1];
        	neighborhood[33] = d_input[yOffsets[4] + x + 2];
        	neighborhood[34] = d_input[yOffsets[4] + x + 3];
        	
        	neighborhood[35] = d_input[yOffsets[5] + x - 3];
        	neighborhood[35] = d_input[yOffsets[5] + x - 2];
        	neighborhood[37] = d_input[yOffsets[5] + x - 1];
        	neighborhood[38] = d_input[yOffsets[5] + x - 0];
        	neighborhood[39] = d_input[yOffsets[5] + x + 1];
        	neighborhood[40] = d_input[yOffsets[5] + x + 2];
        	neighborhood[41] = d_input[yOffsets[5] + x + 3];
        	
        	neighborhood[42] = d_input[yOffsets[6] + x - 3];
        	neighborhood[43] = d_input[yOffsets[6] + x - 2];
        	neighborhood[44] = d_input[yOffsets[6] + x - 1];
        	neighborhood[45] = d_input[yOffsets[6] + x - 0];
        	neighborhood[46] = d_input[yOffsets[6] + x + 1];
        	neighborhood[47] = d_input[yOffsets[6] + x + 2];
        	neighborhood[48] = d_input[yOffsets[6] + x + 3];
        	
	}
	else
	{
        	neighborhood[0] = 0;
        	neighborhood[1] = 0;
        	neighborhood[2] = 0;
        	neighborhood[3] = 0;
        	neighborhood[4] = 0;
        	neighborhood[5] = 0;
        	neighborhood[6] = 0;
        	
		neighborhood[7] =  0;
        	neighborhood[8] =  0;
        	neighborhood[9] =  0;
        	neighborhood[10] = 0;
        	neighborhood[11] = 0;
        	neighborhood[12] = 0;
        	neighborhood[13] = 0;
        	
        	neighborhood[14] = 0;
        	neighborhood[15] = 0;
        	neighborhood[16] = 0;
        	neighborhood[17] = 0;
        	neighborhood[18] = 0;
        	neighborhood[19] = 0;
        	neighborhood[20] = 0;
        	
        	neighborhood[21] = 0;
        	neighborhood[22] = 0;
        	neighborhood[23] = 0;
        	
        	neighborhood[24] = d_input[yOffsets[3] + x - 0];
        	
        	neighborhood[25] = 255;
        	neighborhood[26] = 255;
        	neighborhood[27] = 255;
        	
        	neighborhood[28] = 255;
        	neighborhood[29] = 255;
        	neighborhood[30] = 255;
        	neighborhood[31] = 255;
        	neighborhood[32] = 255;
        	neighborhood[33] = 255;
        	neighborhood[34] = 255;
        	
        	neighborhood[35] = 255;
        	neighborhood[35] = 255;
        	neighborhood[37] = 255;
        	neighborhood[38] = 255;
        	neighborhood[39] = 255;
        	neighborhood[40] = 255;
        	neighborhood[41] = 255;
        	
        	neighborhood[42] = 255;
        	neighborhood[43] = 255;
        	neighborhood[44] = 255;
        	neighborhood[45] = 255;
        	neighborhood[46] = 255;
        	neighborhood[47] = 255;
        	neighborhood[48] = 255;
	}

	//sort neighborhood
	QuickSort(neighborhood, 0, 7 * 7);
	
	// assign pixel to median

	d_output[yOffset + x] = neighborhood[24];

}

__global__ void medianFilter11( uint8_t *d_input, uint8_t *d_output) {
        // map from threadIdx/BlockIdx to pixel position^M
        int x = blockIdx.x;
        int y = blockIdx.y;
        int dim = 11;

	int yOffsets[11];
	const int yOffset = y * gridDim.x;
	
	yOffsets[0] = yOffset - gridDim.x * 5;
	yOffsets[1] = yOffset - gridDim.x * 4;
	yOffsets[2] = yOffset - gridDim.x * 3;
	yOffsets[3] = yOffset - gridDim.x * 2;
	yOffsets[4] = yOffset - gridDim.x * 1;
	yOffsets[5] = yOffset;
	yOffsets[6] = yOffset + gridDim.x * 1;
	yOffsets[7] = yOffset + gridDim.x * 2;
	yOffsets[8] = yOffset + gridDim.x * 3;
	yOffsets[9] = yOffset + gridDim.x * 4;
	yOffsets[10] = yOffset + gridDim.x * 5;
	
	uint8_t neighborhood[11 * 11];
	
	
	if (y > 0 && y < (gridDim.y - 1) && x > 0 && x < (gridDim.x - 1))
	{
		//for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				for (int k = 0; k < dim / 2; k++)
				{
        				neighborhood[dim * (dim - j - 1) + k] = d_input[yOffsets[j] + x + k];
        				neighborhood[dim * (dim - j - 1) + k + (dim / 2)] = d_input[yOffsets[j] + x - k];
				}
			}
		}
		/*
        	neighborhood[0] = d_input[yPrev + x - 1];
        	neighborhood[1] = d_input[yPrev + x];
        	neighborhood[2] = d_input[yPrev + x + 1];
        	neighborhood[3] = d_input[yOffset + x - 1];

        	neighborhood[4] = d_input[yOffset + x];

        	neighborhood[5] = d_input[yOffset + x + 1];
        	neighborhood[6] = d_input[yNext + x - 1];
        	neighborhood[7] = d_input[yNext + x];
        	neighborhood[8] = d_input[yNext + x + 1];
        	*/
	}
	else
	{
		for (int i = 0; i < 11 * 11 / 2; i++)
		{
			neighborhood[i] = 0;
		}
		neighborhood[60] = d_input[yOffset + x];
		for (int i = 61; i < 11*11; i++)
		{
			neighborhood[i] = 255;
		}
		
		/*neighborhood[0] = 0;
            neighborhood[1] = 0;
		neighborhood[2] = 0;
		neighborhood[3] = 0;
		
		neighborhood[4] = d_input[yOffset + x];
		
		neighborhood[5] = 255;
		neighborhood[6] = 255;
		neighborhood[7] = 255;
		neighborhood[8] = 255;*/
	}

	//sort neighborhood
	QuickSort(neighborhood, 0, 11*11);
	
	// assign pixel to median

	d_output[yOffset + x] = neighborhood[60];

}

int main (int argc, char *argv[]) {

    if (argc != 4) // Change me per specs
        return 1;

    int dim = atoi(argv[1]);
	int height, width;
    char magic_number[4], input[10];
    int gray_scale;

    //Reads from argv[1] the input pgm file
    FILE *fp = fopen(argv[2],"r");
    fgets(magic_number, 4, fp);
    magic_number[2] = '\0';
	//read up to 10 characters or new line
    fgets(input, 10, fp);
    height = atoi(input);
    fgets(input, 10, fp);
    width = atoi(input);
    fgets(input, 10, fp);
    gray_scale = atoi(input);
    std::vector<uint8_t> mat(height * width);
    //Populates the arrays grabing each pixel from the image and storing it into the vector.
    for (int i= 0; i < height * width; i++)
        mat[i] = fgetc(fp);

    fclose(fp);

    std::vector<uint8_t> median(height * width);
    uint8_t *d_input, *d_output;
    cudaMalloc((void **) &d_input, height * width * sizeof(uint8_t));
    cudaMalloc((void **) &d_output, height * width * sizeof(uint8_t));
	//copy the image that we read, into d_input and send it over to the GPU's memory
    cudaMemcpy(d_input, &mat[0], height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // TODO - Fill median.
	dim3 grid(height, width);

	if (dim == 3)
	{
		medianFilter3<<<grid,1>>>(d_input, d_output);
	}
	else if (dim == 7)
	{
		medianFilter7<<<grid,1>>>(d_input, d_output);
	}
	else if (dim == 11)
	{
		medianFilter11<<<grid,1>>>(d_input, d_output);
	}
	else if (dim == 15)
	{}
	else
	{
		std::cout << "Unsuported Filter Size" << std::endl;
		return 1;
	}
    	cudaMemcpy(&median[0], d_output, height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    	cudaFree(d_input);
    	cudaFree(d_output);

    //Writes the new pgm picture
    fp = fopen(argv[3], "w");
    fprintf(fp, "%s\n%d\n%d\n%d\n", magic_number, height, width, gray_scale);
    for (int i=0;i<median.size();i++)
        fputc(median[i], fp);
    fclose(fp);

    return 0;
}

