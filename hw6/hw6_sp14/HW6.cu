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
        //int mid = dim / 2 + 1;^M
        //int offset = x + y * gridDim.x;^M
        //int offset2 = offset;^M
        //offset2 = x + (gridDim.y - y);^M
        //offset2 = y + x * gridDim.y;^M
        //offset2 = y + (gridDim.x * (gridDim.y - x - 1));^M
	//d_output[offset] = d_input[offset2];

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
	else if (dim == 5)
	{}
	else if (dim == 7)
	{}
	else if (dim == 11)
	{}
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

