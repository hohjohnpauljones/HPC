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
		for(int i=0; i<dim; i++){
			//for(int j=0; j<dim; j++){
				neighborhood[i*gridDim.x + 0] = d_input[i*gridDim.x + x - 1];
				neighborhood[i*gridDim.x + 1] = d_input[i*gridDim.x + x];
				neighborhood[i*gridDim.x + 2] = d_input[i*gridDim.x + x + 1];			
			//}
		}
        	/*neighborhood[0] = d_input[yPrev + x - 1];
        	neighborhood[1] = d_input[yPrev + x];
        	neighborhood[2] = d_input[yPrev + x + 1];
        	
        	neighborhood[3] = d_input[yOffset + x - 1];

        	neighborhood[4] = d_input[yOffset + x];

        	neighborhood[5] = d_input[yOffset + x + 1];
        	
        	neighborhood[6] = d_input[yNext + x - 1];
        	neighborhood[7] = d_input[yNext + x];
        	neighborhood[8] = d_input[yNext + x + 1];*/
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
	
	const int yOffset1 = yOffset - gridDim.x * 3;
	const int yOffset2 = yOffset - gridDim.x * 2;
	const int yOffset3 = yOffset - gridDim.x * 1;
	const int yOffset5 = yOffset + gridDim.x * 1;
	const int yOffset6 = yOffset + gridDim.x * 2;
	const int yOffset7 = yOffset + gridDim.x * 3;
	
	
	
	if (y >= gridDim.y || x >= gridDim.x)
		return ;
	
	int yOffsets[7];
	
	yOffsets[0] = yOffset - gridDim.x * 3;
	yOffsets[1] = yOffset - gridDim.x * 2;
	yOffsets[2] = yOffset - gridDim.x * 1;
	yOffsets[3] = yOffset;
	yOffsets[4] = yOffset + gridDim.x * 1;
	yOffsets[5] = yOffset + gridDim.x * 2;
	yOffsets[6] = yOffset + gridDim.x * 3;
	
	uint8_t neighborhood[7 * 7];
	
	
	if (y > 0 && y < (gridDim.y - 1) && x > 0 && x < (gridDim.x - 1))
	{

        	neighborhood[0] = d_input[yOffset1 + x - 3];
        	neighborhood[1] = d_input[yOffset1 + x - 2];
        	neighborhood[2] = d_input[yOffset1 + x - 1];
        	neighborhood[3] = d_input[yOffset1 + x - 0];
        	neighborhood[4] = d_input[yOffset1 + x + 1];
        	neighborhood[5] = d_input[yOffset1 + x + 2];
        	neighborhood[6] = d_input[yOffset1 + x + 3];
        	
		neighborhood[7] = d_input[yOffset2 + x - 3];
        	neighborhood[8] = d_input[yOffset2 + x - 2];
        	neighborhood[9] = d_input[yOffset2 + x - 1];
        	neighborhood[10] = d_input[yOffset2 + x - 0];
        	neighborhood[11] = d_input[yOffset2 + x + 1];
        	neighborhood[12] = d_input[yOffset2 + x + 2];
        	neighborhood[13] = d_input[yOffset2 + x + 3];
        	
        	neighborhood[14] = d_input[yOffset3 + x - 3];
        	neighborhood[15] = d_input[yOffset3 + x - 2];
        	neighborhood[16] = d_input[yOffset3 + x - 1];
        	neighborhood[17] = d_input[yOffset3 + x - 0];
        	neighborhood[18] = d_input[yOffset3 + x + 1];
        	neighborhood[19] = d_input[yOffset3 + x + 2];
        	neighborhood[20] = d_input[yOffset3 + x + 3];
        	
        	neighborhood[21] = d_input[yOffset + x - 3];
        	neighborhood[22] = d_input[yOffset + x - 2];
        	neighborhood[23] = d_input[yOffset + x - 1];
        	
        	neighborhood[24] = d_input[yOffset + x - 0];
        	
        	neighborhood[25] = d_input[yOffset + x + 1];
        	neighborhood[26] = d_input[yOffset + x + 2];
        	neighborhood[27] = d_input[yOffset + x + 3];
        	
        	neighborhood[28] = d_input[yOffset5 + x - 3];
        	neighborhood[29] = d_input[yOffset5 + x - 2];
        	neighborhood[30] = d_input[yOffset5 + x - 1];
        	neighborhood[31] = d_input[yOffset5 + x - 0];
        	neighborhood[32] = d_input[yOffset5 + x + 1];
        	neighborhood[33] = d_input[yOffset5 + x + 2];
        	neighborhood[34] = d_input[yOffset5 + x + 3];
        	
        	neighborhood[35] = d_input[yOffset6 + x - 3];
        	neighborhood[35] = d_input[yOffset6 + x - 2];
        	neighborhood[37] = d_input[yOffset6 + x - 1];
        	neighborhood[38] = d_input[yOffset6 + x - 0];
        	neighborhood[39] = d_input[yOffset6 + x + 1];
        	neighborhood[40] = d_input[yOffset6 + x + 2];
        	neighborhood[41] = d_input[yOffset6 + x + 3];
        	
        	neighborhood[42] = d_input[yOffset7 + x - 3];
        	neighborhood[43] = d_input[yOffset7 + x - 2];
        	neighborhood[44] = d_input[yOffset7 + x - 1];
        	neighborhood[45] = d_input[yOffset7 + x - 0];
        	neighborhood[46] = d_input[yOffset7 + x + 1];
        	neighborhood[47] = d_input[yOffset7 + x + 2];
        	neighborhood[48] = d_input[yOffset7 + x + 3];
        	
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
        	
        	neighborhood[24] = d_input[yOffset + x - 0];
        	
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
	
	
	//if (y > 0 && y < (gridDim.y - 1) && x > 0 && x < (gridDim.x - 1))
	if(true)
	{
		//for (int i = 0; i < dim; i++)
		
			for (int j = 0; j < dim; j++)
			{
				for (int k = 0; k < dim / 2; k++)
				{
        				neighborhood[dim * (dim - j - 1) + k] = 45;//d_input[yOffsets[j] + x + k];
        				neighborhood[dim * (dim - j - 1) + k + (dim / 2)] = 45;//d_input[yOffsets[j] + x - k];
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
		for (int i = 0; i < 60; i++)
		{
			neighborhood[i] = 0;
		}
		//neighborhood[60] = d_input[yOffset + x];
		for (int i = 61; i < 121; i++)
		{
			neighborhood[i] = 255;
		}
	}

	//sort neighborhood
	QuickSort(neighborhood, 0, 11*11);
	
	// assign pixel to median

	d_output[yOffset + x] = neighborhood[60];

}

__global__ void medianFilter15( uint8_t *d_input, uint8_t *d_output) {
        // map from threadIdx/BlockIdx to pixel position^M
        int x = blockIdx.x;
        int y = blockIdx.y;
        int dim = 15;
	
	const int yOffset = y * gridDim.x;
	
	int yOffsets[15];
	
	yOffsets[0] = yOffset - gridDim.x * 7;
	yOffsets[1] = yOffset - gridDim.x * 6;
	yOffsets[2] = yOffset - gridDim.x * 5;
	yOffsets[3] = yOffset - gridDim.x * 4;
	yOffsets[4] = yOffset - gridDim.x * 3;
	yOffsets[5] = yOffset - gridDim.x * 2;
	yOffsets[6] = yOffset - gridDim.x * 1;
	yOffsets[7] = yOffset;
	yOffsets[8] = yOffset + gridDim.x * 1;
	yOffsets[9] = yOffset + gridDim.x * 2;
	yOffsets[10] = yOffset + gridDim.x * 3;
	yOffsets[11] = yOffset + gridDim.x * 4;
	yOffsets[12] = yOffset + gridDim.x * 5;
	yOffsets[13] = yOffset + gridDim.x * 6;
	yOffsets[14] = yOffset + gridDim.x * 7;
	
	uint8_t neighborhood[15 * 15];
	
	
	if (y > 0 && y < (gridDim.y - 1) && x > 0 && x < (gridDim.x - 1))
	{
		//Row 1
        	neighborhood[0] = d_input[yOffsets[0] + x - 7];
        	neighborhood[1] = d_input[yOffsets[0] + x - 6];
        	neighborhood[2] = d_input[yOffsets[0] + x - 5];
        	neighborhood[3] = d_input[yOffsets[0] + x - 4];
        	neighborhood[4] = d_input[yOffsets[0] + x - 3];
        	neighborhood[5] = d_input[yOffsets[0] + x - 2];
        	neighborhood[6] = d_input[yOffsets[0] + x - 1];
        	neighborhood[7] = d_input[yOffsets[0] + x + 0];
        	neighborhood[8] = d_input[yOffsets[0] + x + 1];
        	neighborhood[9] = d_input[yOffsets[0] + x + 2];
        	neighborhood[10] = d_input[yOffsets[0] + x - 3];
        	neighborhood[11] = d_input[yOffsets[0] + x - 4];
        	neighborhood[12] = d_input[yOffsets[0] + x - 5];
        	neighborhood[13] = d_input[yOffsets[0] + x - 6];
        	neighborhood[14] = d_input[yOffsets[0] + x - 7];
        	
        	//Row 2
        	neighborhood[15] = d_input[yOffsets[1] + x - 7];
        	neighborhood[16] = d_input[yOffsets[1] + x - 6];
        	neighborhood[17] = d_input[yOffsets[1] + x - 5];
        	neighborhood[18] = d_input[yOffsets[1] + x - 4];
        	neighborhood[19] = d_input[yOffsets[1] + x - 3];
        	neighborhood[20] = d_input[yOffsets[1] + x - 2];
        	neighborhood[21] = d_input[yOffsets[1] + x - 1];
        	neighborhood[22] = d_input[yOffsets[1] + x + 0];
        	neighborhood[23] = d_input[yOffsets[1] + x + 1];
        	neighborhood[24] = d_input[yOffsets[1] + x + 2];
        	neighborhood[25] = d_input[yOffsets[1] + x - 3];
        	neighborhood[26] = d_input[yOffsets[1] + x - 4];
        	neighborhood[27] = d_input[yOffsets[1] + x - 5];
        	neighborhood[28] = d_input[yOffsets[1] + x - 6];
        	neighborhood[29] = d_input[yOffsets[1] + x - 7];
        	
        	//Row 3
        	neighborhood[30] = d_input[yOffsets[2] + x - 7];
        	neighborhood[31] = d_input[yOffsets[2] + x - 6];
        	neighborhood[32] = d_input[yOffsets[2] + x - 5];
        	neighborhood[33] = d_input[yOffsets[2] + x - 4];
        	neighborhood[34] = d_input[yOffsets[2] + x - 3];
        	neighborhood[35] = d_input[yOffsets[2] + x - 2];
        	neighborhood[36] = d_input[yOffsets[2] + x - 1];
        	neighborhood[37] = d_input[yOffsets[2] + x + 0];
        	neighborhood[38] = d_input[yOffsets[2] + x + 1];
        	neighborhood[39] = d_input[yOffsets[2] + x + 2];
        	neighborhood[40] = d_input[yOffsets[2] + x - 3];
        	neighborhood[41] = d_input[yOffsets[2] + x - 4];
        	neighborhood[42] = d_input[yOffsets[2] + x - 5];
        	neighborhood[43] = d_input[yOffsets[2] + x - 6];
        	neighborhood[44] = d_input[yOffsets[2] + x - 7];
        	
        	//Row 4
        	neighborhood[45] = d_input[yOffsets[3] + x - 7];
        	neighborhood[46] = d_input[yOffsets[3] + x - 6];
        	neighborhood[47] = d_input[yOffsets[3] + x - 5];
        	neighborhood[48] = d_input[yOffsets[3] + x - 4];
        	neighborhood[49] = d_input[yOffsets[3] + x - 3];
        	neighborhood[50] = d_input[yOffsets[3] + x - 2];
        	neighborhood[51] = d_input[yOffsets[3] + x - 1];
        	neighborhood[52] = d_input[yOffsets[3] + x + 0];
        	neighborhood[53] = d_input[yOffsets[3] + x + 1];
        	neighborhood[54] = d_input[yOffsets[3] + x + 2];
        	neighborhood[55] = d_input[yOffsets[3] + x - 3];
        	neighborhood[56] = d_input[yOffsets[3] + x - 4];
        	neighborhood[57] = d_input[yOffsets[3] + x - 5];
        	neighborhood[58] = d_input[yOffsets[3] + x - 6];
        	neighborhood[59] = d_input[yOffsets[3] + x - 7];
        	
        	//Row 5
        	neighborhood[60] = d_input[yOffsets[4] + x - 7];
        	neighborhood[61] = d_input[yOffsets[4] + x - 6];
        	neighborhood[62] = d_input[yOffsets[4] + x - 5];
        	neighborhood[63] = d_input[yOffsets[4] + x - 4];
        	neighborhood[64] = d_input[yOffsets[4] + x - 3];
        	neighborhood[65] = d_input[yOffsets[4] + x - 2];
        	neighborhood[66] = d_input[yOffsets[4] + x - 1];
        	neighborhood[67] = d_input[yOffsets[4] + x + 0];
        	neighborhood[68] = d_input[yOffsets[4] + x + 1];
        	neighborhood[69] = d_input[yOffsets[4] + x + 2];
        	neighborhood[70] = d_input[yOffsets[4] + x - 3];
        	neighborhood[71] = d_input[yOffsets[4] + x - 4];
        	neighborhood[72] = d_input[yOffsets[4] + x - 5];
        	neighborhood[73] = d_input[yOffsets[4] + x - 6];
        	neighborhood[74] = d_input[yOffsets[4] + x - 7];
        	
        	//Row 6
        	neighborhood[75] = d_input[yOffsets[5] + x - 7];
        	neighborhood[76] = d_input[yOffsets[5] + x - 6];
        	neighborhood[77] = d_input[yOffsets[5] + x - 5];
        	neighborhood[78] = d_input[yOffsets[5] + x - 4];
        	neighborhood[79] = d_input[yOffsets[5] + x - 3];
        	neighborhood[80] = d_input[yOffsets[5] + x - 2];
        	neighborhood[81] = d_input[yOffsets[5] + x - 1];
        	neighborhood[82] = d_input[yOffsets[5] + x + 0];
        	neighborhood[83] = d_input[yOffsets[5] + x + 1];
        	neighborhood[84] = d_input[yOffsets[5] + x + 2];
        	neighborhood[85] = d_input[yOffsets[5] + x - 3];
        	neighborhood[86] = d_input[yOffsets[5] + x - 4];
        	neighborhood[87] = d_input[yOffsets[5] + x - 5];
        	neighborhood[88] = d_input[yOffsets[5] + x - 6];
        	neighborhood[89] = d_input[yOffsets[5] + x - 7];
        	
        	//Row 7
        	neighborhood[90] = d_input[yOffsets[6] + x - 7];
        	neighborhood[91] = d_input[yOffsets[6] + x - 6];
        	neighborhood[92] = d_input[yOffsets[6] + x - 5];
        	neighborhood[93] = d_input[yOffsets[6] + x - 4];
        	neighborhood[94] = d_input[yOffsets[6] + x - 3];
        	neighborhood[95] = d_input[yOffsets[6] + x - 2];
        	neighborhood[96] = d_input[yOffsets[6] + x - 1];
        	neighborhood[97] = d_input[yOffsets[6] + x + 0];
        	neighborhood[98] = d_input[yOffsets[6] + x + 1];
        	neighborhood[99] = d_input[yOffsets[6] + x + 2];
        	neighborhood[100] = d_input[yOffsets[6] + x - 3];
        	neighborhood[101] = d_input[yOffsets[6] + x - 4];
        	neighborhood[102] = d_input[yOffsets[6] + x - 5];
        	neighborhood[103] = d_input[yOffsets[6] + x - 6];
        	neighborhood[104] = d_input[yOffsets[6] + x - 7];
        	
        	//Row 8
        	neighborhood[105] = d_input[yOffsets[7] + x - 7];
        	neighborhood[106] = d_input[yOffsets[7] + x - 6];
        	neighborhood[107] = d_input[yOffsets[7] + x - 5];
        	neighborhood[108] = d_input[yOffsets[7] + x - 4];
        	neighborhood[109] = d_input[yOffsets[7] + x - 3];
        	neighborhood[110] = d_input[yOffsets[7] + x - 2];
        	neighborhood[111] = d_input[yOffsets[7] + x - 1];
        	
        	neighborhood[112] = d_input[yOffsets[7] + x + 0];
        	
        	neighborhood[113] = d_input[yOffsets[7] + x + 1];
        	neighborhood[114] = d_input[yOffsets[7] + x + 2];
        	neighborhood[115] = d_input[yOffsets[7] + x - 3];
        	neighborhood[116] = d_input[yOffsets[7] + x - 4];
        	neighborhood[117] = d_input[yOffsets[7] + x - 5];
        	neighborhood[118] = d_input[yOffsets[7] + x - 6];
        	neighborhood[119] = d_input[yOffsets[7] + x - 7];
        	
        	//Row 9
        	neighborhood[120] = d_input[yOffsets[8] + x - 7];
        	neighborhood[121] = d_input[yOffsets[8] + x - 6];
        	neighborhood[122] = d_input[yOffsets[8] + x - 5];
        	neighborhood[123] = d_input[yOffsets[8] + x - 4];
        	neighborhood[124] = d_input[yOffsets[8] + x - 3];
        	neighborhood[125] = d_input[yOffsets[8] + x - 2];
        	neighborhood[126] = d_input[yOffsets[8] + x - 1];
        	neighborhood[127] = d_input[yOffsets[8] + x + 0];
        	neighborhood[128] = d_input[yOffsets[8] + x + 1];
        	neighborhood[129] = d_input[yOffsets[8] + x + 2];
        	neighborhood[130] = d_input[yOffsets[8] + x - 3];
        	neighborhood[131] = d_input[yOffsets[8] + x - 4];
        	neighborhood[132] = d_input[yOffsets[8] + x - 5];
        	neighborhood[133] = d_input[yOffsets[8] + x - 6];
        	neighborhood[134] = d_input[yOffsets[8] + x - 7];
        	
        	//Row 10
        	neighborhood[135] = d_input[yOffsets[9] + x - 7];
        	neighborhood[136] = d_input[yOffsets[9] + x - 6];
        	neighborhood[137] = d_input[yOffsets[9] + x - 5];
        	neighborhood[138] = d_input[yOffsets[9] + x - 4];
        	neighborhood[139] = d_input[yOffsets[9] + x - 3];
        	neighborhood[140] = d_input[yOffsets[9] + x - 2];
        	neighborhood[141] = d_input[yOffsets[9] + x - 1];
        	neighborhood[142] = d_input[yOffsets[9] + x + 0];
        	neighborhood[143] = d_input[yOffsets[9] + x + 1];
        	neighborhood[144] = d_input[yOffsets[9] + x + 2];
        	neighborhood[145] = d_input[yOffsets[9] + x - 3];
        	neighborhood[146] = d_input[yOffsets[9] + x - 4];
        	neighborhood[147] = d_input[yOffsets[9] + x - 5];
        	neighborhood[148] = d_input[yOffsets[9] + x - 6];
        	neighborhood[149] = d_input[yOffsets[9] + x - 7];
        	
        	//Row 11
        	neighborhood[150] = d_input[yOffsets[10] + x - 7];
        	neighborhood[151] = d_input[yOffsets[10] + x - 6];
        	neighborhood[152] = d_input[yOffsets[10] + x - 5];
        	neighborhood[153] = d_input[yOffsets[10] + x - 4];
        	neighborhood[154] = d_input[yOffsets[10] + x - 3];
        	neighborhood[155] = d_input[yOffsets[10] + x - 2];
        	neighborhood[156] = d_input[yOffsets[10] + x - 1];
        	neighborhood[157] = d_input[yOffsets[10] + x + 0];
        	neighborhood[158] = d_input[yOffsets[10] + x + 1];
        	neighborhood[159] = d_input[yOffsets[10] + x + 2];
        	neighborhood[160] = d_input[yOffsets[10] + x - 3];
        	neighborhood[161] = d_input[yOffsets[10] + x - 4];
        	neighborhood[162] = d_input[yOffsets[10] + x - 5];
        	neighborhood[163] = d_input[yOffsets[10] + x - 6];
        	neighborhood[164] = d_input[yOffsets[10] + x - 7];
      	
      	//Row 12
        	neighborhood[165] = d_input[yOffsets[11] + x - 7];
        	neighborhood[166] = d_input[yOffsets[11] + x - 6];
        	neighborhood[167] = d_input[yOffsets[11] + x - 5];
        	neighborhood[167] = d_input[yOffsets[11] + x - 4];
        	neighborhood[169] = d_input[yOffsets[11] + x - 3];
        	neighborhood[170] = d_input[yOffsets[11] + x - 2];
        	neighborhood[171] = d_input[yOffsets[11] + x - 1];
        	neighborhood[172] = d_input[yOffsets[11] + x + 0];
        	neighborhood[173] = d_input[yOffsets[11] + x + 1];
        	neighborhood[174] = d_input[yOffsets[11] + x + 2];
        	neighborhood[175] = d_input[yOffsets[11] + x - 3];
        	neighborhood[176] = d_input[yOffsets[11] + x - 4];
        	neighborhood[177] = d_input[yOffsets[11] + x - 5];
        	neighborhood[178] = d_input[yOffsets[11] + x - 6];
        	neighborhood[179] = d_input[yOffsets[11] + x - 7];
        	
        	//Row 13
        	neighborhood[180] = d_input[yOffsets[12] + x - 7];
        	neighborhood[181] = d_input[yOffsets[12] + x - 6];
        	neighborhood[182] = d_input[yOffsets[12] + x - 5];
        	neighborhood[183] = d_input[yOffsets[12] + x - 4];
        	neighborhood[184] = d_input[yOffsets[12] + x - 3];
        	neighborhood[185] = d_input[yOffsets[12] + x - 2];
        	neighborhood[186] = d_input[yOffsets[12] + x - 1];
        	neighborhood[187] = d_input[yOffsets[12] + x + 0];
        	neighborhood[188] = d_input[yOffsets[12] + x + 1];
        	neighborhood[189] = d_input[yOffsets[12] + x + 2];
        	neighborhood[190] = d_input[yOffsets[12] + x - 3];
        	neighborhood[191] = d_input[yOffsets[12] + x - 4];
        	neighborhood[192] = d_input[yOffsets[12] + x - 5];
        	neighborhood[193] = d_input[yOffsets[12] + x - 6];
        	neighborhood[194] = d_input[yOffsets[12] + x - 7];
        	
        	//Row 14
		neighborhood[195] = d_input[yOffsets[13] + x - 7];
        	neighborhood[196] = d_input[yOffsets[13] + x - 6];
        	neighborhood[197] = d_input[yOffsets[13] + x - 5];
        	neighborhood[198] = d_input[yOffsets[13] + x - 4];
        	neighborhood[199] = d_input[yOffsets[13] + x - 3];
        	neighborhood[200] = d_input[yOffsets[13] + x - 2];
        	neighborhood[201] = d_input[yOffsets[13] + x - 1];
        	neighborhood[202] = d_input[yOffsets[13] + x + 0];
        	neighborhood[203] = d_input[yOffsets[13] + x + 1];
        	neighborhood[204] = d_input[yOffsets[13] + x + 2];
        	neighborhood[205] = d_input[yOffsets[13] + x - 3];
        	neighborhood[206] = d_input[yOffsets[13] + x - 4];
        	neighborhood[207] = d_input[yOffsets[13] + x - 5];
        	neighborhood[208] = d_input[yOffsets[13] + x - 6];
        	neighborhood[209] = d_input[yOffsets[13] + x - 7];
      	
      	//Row 15
        	neighborhood[210] = d_input[yOffsets[14] + x - 7];
        	neighborhood[211] = d_input[yOffsets[14] + x - 6];
        	neighborhood[212] = d_input[yOffsets[14] + x - 5];
        	neighborhood[213] = d_input[yOffsets[14] + x - 4];
        	neighborhood[214] = d_input[yOffsets[14] + x - 3];
        	neighborhood[215] = d_input[yOffsets[14] + x - 2];
        	neighborhood[216] = d_input[yOffsets[14] + x - 1];
        	neighborhood[217] = d_input[yOffsets[14] + x + 0];
        	neighborhood[218] = d_input[yOffsets[14] + x + 1];
        	neighborhood[219] = d_input[yOffsets[14] + x + 2];
        	neighborhood[220] = d_input[yOffsets[14] + x - 3];
        	neighborhood[221] = d_input[yOffsets[14] + x - 4];
        	neighborhood[222] = d_input[yOffsets[14] + x - 5];
        	neighborhood[223] = d_input[yOffsets[14] + x - 6];
        	neighborhood[224] = d_input[yOffsets[14] + x - 7];
        	
        	
	}
	else
	{
        	//Row 1
        	neighborhood[0] = 0;
        	neighborhood[1] = 0;
        	neighborhood[2] = 0;
        	neighborhood[3] = 0;
        	neighborhood[4] = 0;
        	neighborhood[5] = 0;
        	neighborhood[6] = 0;
        	neighborhood[7] = 0;
        	neighborhood[8] = 0;
        	neighborhood[9] = 0;
        	neighborhood[10] = 0;
        	neighborhood[11] = 0;
        	neighborhood[12] = 0;
        	neighborhood[13] = 0;
        	neighborhood[14] = 0;
        	
        	//Row 2
        	neighborhood[15] = 0;
        	neighborhood[16] = 0;
        	neighborhood[17] = 0;
        	neighborhood[18] = 0;
        	neighborhood[19] = 0;
        	neighborhood[20] = 0;
        	neighborhood[21] = 0;
        	neighborhood[22] = 0;
        	neighborhood[23] = 0;
        	neighborhood[24] = 0;
        	neighborhood[25] = 0;
        	neighborhood[26] = 0;
        	neighborhood[27] = 0;
        	neighborhood[28] = 0;
        	neighborhood[29] = 0;
        	
        	//Row 3
        	neighborhood[30] = 0;
        	neighborhood[31] = 0;
        	neighborhood[32] = 0;
        	neighborhood[33] = 0;
        	neighborhood[34] = 0;
        	neighborhood[35] = 0;
        	neighborhood[36] = 0;
        	neighborhood[37] = 0;
        	neighborhood[38] = 0;
        	neighborhood[39] = 0;
        	neighborhood[40] = 0;
        	neighborhood[41] = 0;
        	neighborhood[42] = 0;
        	neighborhood[43] = 0;
        	neighborhood[44] = 0;
        	
        	//Row 4
        	neighborhood[45] = 0;
        	neighborhood[46] = 0;
        	neighborhood[47] = 0;
        	neighborhood[48] = 0;
        	neighborhood[49] = 0;
        	neighborhood[50] = 0;
        	neighborhood[51] = 0;
        	neighborhood[52] = 0;
        	neighborhood[53] = 0;
        	neighborhood[54] = 0;
        	neighborhood[55] = 0;
        	neighborhood[56] = 0;
        	neighborhood[57] = 0;
        	neighborhood[58] = 0;
        	neighborhood[59] = 0;
        	
        	//Row 5
        	neighborhood[60] = 0;
        	neighborhood[61] = 0;
        	neighborhood[62] = 0;
        	neighborhood[63] = 0;
        	neighborhood[64] = 0;
        	neighborhood[65] = 0;
        	neighborhood[66] = 0;
        	neighborhood[67] = 0;
        	neighborhood[68] = 0;
        	neighborhood[69] = 0;
        	neighborhood[70] = 0;
        	neighborhood[71] = 0;
        	neighborhood[72] = 0;
        	neighborhood[73] = 0;
        	neighborhood[74] = 0;
        	
        	//Row 6
        	neighborhood[75] = 0;
        	neighborhood[76] = 0;
        	neighborhood[77] = 0;
        	neighborhood[78] = 0;
        	neighborhood[79] = 0;
        	neighborhood[80] = 0;
        	neighborhood[81] = 0;
        	neighborhood[82] = 0;
        	neighborhood[83] = 0;
        	neighborhood[84] = 0;
        	neighborhood[85] = 0;
        	neighborhood[86] = 0;
        	neighborhood[87] = 0;
        	neighborhood[88] = 0;
        	neighborhood[89] = 0;
        	
        	//Row 7
        	neighborhood[90] = 0;
        	neighborhood[91] = 0;
        	neighborhood[92] = 0;
        	neighborhood[93] = 0;
        	neighborhood[94] = 0;
        	neighborhood[95] = 0;
        	neighborhood[96] = 0;
        	neighborhood[97] = 0;
        	neighborhood[98] = 0;
        	neighborhood[99] = 0;
        	neighborhood[100] = 0;
        	neighborhood[101] = 0;
        	neighborhood[102] = 0;
        	neighborhood[103] = 0;
        	neighborhood[104] = 0;
        	
        	//Row 8
        	neighborhood[105] = 0;
        	neighborhood[106] = 0;
        	neighborhood[107] = 0;
        	neighborhood[108] = 0;
        	neighborhood[109] = 0;
        	neighborhood[110] = 0;
        	neighborhood[111] = 0;
        	
        	neighborhood[112] = d_input[yOffsets[7] + x + 0];
        	
        	neighborhood[113] = 255;
        	neighborhood[114] = 255;
        	neighborhood[115] = 255;
        	neighborhood[116] = 255;
        	neighborhood[117] = 255;
        	neighborhood[118] = 255;
        	neighborhood[119] = 255;
        	
        	//Row 9
        	neighborhood[120] = 255;
        	neighborhood[121] = 255;
        	neighborhood[122] = 255;
        	neighborhood[123] = 255;
        	neighborhood[124] = 255;
        	neighborhood[125] = 255;
        	neighborhood[126] = 255;
        	neighborhood[127] = 255;
        	neighborhood[128] = 255;
        	neighborhood[129] = 255;
        	neighborhood[130] = 255;
        	neighborhood[131] = 255;
        	neighborhood[132] = 255;
        	neighborhood[133] = 255;
        	neighborhood[134] = 255;
        	
        	//Row 10
        	neighborhood[135] = 255;
        	neighborhood[136] = 255;
        	neighborhood[137] = 255;
        	neighborhood[138] = 255;
        	neighborhood[139] = 255;
        	neighborhood[140] = 255;
        	neighborhood[141] = 255;
        	neighborhood[142] = 255;
        	neighborhood[143] = 255;
        	neighborhood[144] = 255;
        	neighborhood[145] = 255;
        	neighborhood[146] = 255;
        	neighborhood[147] = 255;
        	neighborhood[148] = 255;
        	neighborhood[149] = 255;
        	
        	//Row 11
        	neighborhood[150] = 255;
        	neighborhood[151] = 255;
        	neighborhood[152] = 255;
        	neighborhood[153] = 255;
        	neighborhood[154] = 255;
        	neighborhood[155] = 255;
        	neighborhood[156] = 255;
        	neighborhood[157] = 255;
        	neighborhood[158] = 255;
        	neighborhood[159] = 255;
        	neighborhood[160] = 255;
        	neighborhood[161] = 255;
        	neighborhood[162] = 255;
        	neighborhood[163] = 255;
        	neighborhood[164] = 255;
      	
      	//Row 12
        	neighborhood[165] = 255;
        	neighborhood[166] = 255;
        	neighborhood[167] = 255;
        	neighborhood[167] = 255;
        	neighborhood[169] = 255;
        	neighborhood[170] = 255;
        	neighborhood[171] = 255;
        	neighborhood[172] = 255;
        	neighborhood[173] = 255;
        	neighborhood[174] = 255;
        	neighborhood[175] = 255;
        	neighborhood[176] = 255;
        	neighborhood[177] = 255;
        	neighborhood[178] = 255;
        	neighborhood[179] = 255;
        	
        	//Row 13
        	neighborhood[180] = 255;
        	neighborhood[181] = 255;
        	neighborhood[182] = 255;
        	neighborhood[183] = 255;
        	neighborhood[184] = 255;
        	neighborhood[185] = 255;
        	neighborhood[186] = 255;
        	neighborhood[187] = 255;
        	neighborhood[188] = 255;
        	neighborhood[189] = 255;
        	neighborhood[190] = 255;
        	neighborhood[191] = 255;
        	neighborhood[192] = 255;
        	neighborhood[193] = 255;
        	neighborhood[194] = 255;
        	
        	//Row 14
		neighborhood[195] = 255;
        	neighborhood[196] = 255;
        	neighborhood[197] = 255;
        	neighborhood[198] = 255;
        	neighborhood[199] = 255;
        	neighborhood[200] = 255;
        	neighborhood[201] = 255;
        	neighborhood[202] = 255;
        	neighborhood[203] = 255;
        	neighborhood[204] = 255;
        	neighborhood[205] = 255;
        	neighborhood[206] = 255;
        	neighborhood[207] = 255;
        	neighborhood[208] = 255;
        	neighborhood[209] = 255;
      	
      	//Row 15
        	neighborhood[210] = 255;
        	neighborhood[211] = 255;
        	neighborhood[212] = 255;
        	neighborhood[213] = 255;
        	neighborhood[214] = 255;
        	neighborhood[215] = 255;
        	neighborhood[216] = 255;
        	neighborhood[217] = 255;
        	neighborhood[218] = 255;
        	neighborhood[219] = 255;
        	neighborhood[220] = 255;
        	neighborhood[221] = 255;
        	neighborhood[222] = 255;
        	neighborhood[223] = 255;
        	neighborhood[224] = 255;
	}

	//sort neighborhood
	QuickSort(neighborhood, 0, 15 * 15);
	
	// assign pixel to median

	d_output[yOffset + x] = neighborhood[112];

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
	dim3 grid(5, 5);

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
	{
		medianFilter15<<<grid,1>>>(d_input, d_output);
	}
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

