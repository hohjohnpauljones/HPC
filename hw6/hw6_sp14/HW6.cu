#include <iostream>
#include <vector>

#include <cstdio>
#include <cstdlib>

//#include <chrono>

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
	int x = blockIdx.x * blockDim.x + threadIdx.x; //threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y; //blockIdx.y;
	int dim = 3;
	int dim_1d = dim*dim; //turning 2d square into single row length
	
	const int rowSize = gridDim.x * blockDim.x;

	int yOffsets[3];
	yOffsets[0] = (y-1) * rowSize;
	yOffsets[1] = y * rowSize;
	yOffsets[2] = (y+1) * rowSize;


//	printf("blockId.x = %d | blockId.y = %d | threadIdx.x = %d | threadIdx.y = %d | blockDim.x = %d | blockDim.y = %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x, blockDim.y);	
	uint8_t neighborhood[3*3];
	
	if (y > 0 && y < (gridDim.y * blockDim.y - 1) && x > 0 && x < (rowSize - 1))
	{
		for(int i=0; i<dim_1d; i+=dim){
			neighborhood[i]		= d_input[yOffsets[i/dim] + x - 1];
			neighborhood[i + 1] 	= d_input[yOffsets[i/dim] + x];
			neighborhood[i + 2] 	= d_input[yOffsets[i/dim] + x + 1];			
		}
	}
	else
	{
		neighborhood[0] = 0;
		neighborhood[1] = 0;
		neighborhood[2] = 0;
		neighborhood[3] = 0;
		
		neighborhood[4] = d_input[yOffsets[1] + x];
		
		neighborhood[5] = 255;
		neighborhood[6] = 255;
		neighborhood[7] = 255;
		neighborhood[8] = 255;
	}

	//sort neighborhood
	QuickSort(neighborhood, 0, 3 * 3 - 1);
	/*
	if(neighborhood[4] == NULL){	
		printf("X at position is NULL and x is %d and y is %d\n", x, y);
	}else{	
		printf("X at position is NOT NULL and x is %d and y %d\n",x,y );
	}*/
	
	// assign pixel to median
	d_output[yOffsets[1] + x] = neighborhood[4];

}

__global__ void medianFilter7( uint8_t *d_input, uint8_t *d_output) {
	
	// map from threadIdx/BlockIdx to pixel position^M
	int x = blockIdx.x * blockDim.x + threadIdx.x; //threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y; //blockIdx.y;
	int dim = 7;
	int dim_1d = dim*dim; //turning 2d square into single row length
	
	const int rowSize = gridDim.x * blockDim.x;

	int yOffsets[7];
	yOffsets[0] = (y-3) * rowSize;
	yOffsets[1] = (y-2) * rowSize;
	yOffsets[2] = (y-1) * rowSize;
	yOffsets[3] = y * rowSize;
	yOffsets[4] = (y+1)* rowSize;
	yOffsets[5] = (y+2)* rowSize;
	yOffsets[6] = (y+3)* rowSize;
	
	uint8_t neighborhood[7*7];

//	printf("blockId.x = %d | blockId.y = %d | threadIdx.x = %d | threadIdx.y = %d | blockDim.x = %d | blockDim.y = %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x, blockDim.y);	
	if (y > 2 && y < (gridDim.y * blockDim.y - 3) && x > 2 && x < (rowSize - 3))
	{
		for(int i=0; i<dim_1d; i+=dim){
			neighborhood[i] = 	d_input[yOffsets[i/dim] + x - 3];
			neighborhood[i + 1] = 	d_input[yOffsets[i/dim] + x - 2];
			neighborhood[i + 2] = 	d_input[yOffsets[i/dim] + x - 1];
			neighborhood[i + 3] = 	d_input[yOffsets[i/dim] + x];
			
			neighborhood[i + 4] = 	d_input[yOffsets[i/dim] + x + 1];
			neighborhood[i + 5] = 	d_input[yOffsets[i/dim] + x + 2];
			neighborhood[i + 6] = 	d_input[yOffsets[i/dim] + x + 3];			
		}	
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
		
		neighborhood[24] = d_input[yOffsets[3] + x];
		
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
	QuickSort(neighborhood, 0, 7 * 7 - 1);
	
	d_output[yOffsets[3] + x] = neighborhood[24];
}

__global__ void medianFilter11( uint8_t *d_input, uint8_t *d_output) {
	// map from threadIdx/BlockIdx to pixel position^M
	int x = blockIdx.x * blockDim.x + threadIdx.x; //threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y; //blockIdx.y;
	int dim = 11;
	int dim_1d = dim*dim; //turning 2d square into single row length
	
	const int rowSize = gridDim.x * blockDim.x;

	int yOffsets[11];
	yOffsets[0] = (y-5) * rowSize;
	yOffsets[1] = (y-4) * rowSize;
	yOffsets[2] = (y-3) * rowSize;
	yOffsets[3] = (y -2)* rowSize;
	yOffsets[4] = (y-1)* rowSize;
	yOffsets[5] = (y)* rowSize;
	yOffsets[6] = (y+1)* rowSize;
	yOffsets[7] = (y+2)* rowSize;
	yOffsets[8] = (y+3)* rowSize;
	yOffsets[9] = (y+4)* rowSize;
	yOffsets[10] = (y+5)* rowSize;
	
	uint8_t neighborhood[11*11];
	
	if (y > 4 && y < (gridDim.y * blockDim.y - 5)&& x > 4 && x < (rowSize - 5))
	{
		for(int i=0; i<dim_1d; i+=dim){
			neighborhood[i] = 	d_input[yOffsets[i/dim] + x - 5];
			neighborhood[i + 1] = 	d_input[yOffsets[i/dim] + x - 4];
			neighborhood[i + 2] = 	d_input[yOffsets[i/dim] + x - 3];
			neighborhood[i + 3] = 	d_input[yOffsets[i/dim] + x - 2];
			
			neighborhood[i + 4] = 	d_input[yOffsets[i/dim] + x - 1];
			neighborhood[i + 5] = 	d_input[yOffsets[i/dim] + x];
			neighborhood[i + 6] = 	d_input[yOffsets[i/dim] + x + 1];
			neighborhood[i + 7] = 	d_input[yOffsets[i/dim] + x + 2];
			neighborhood[i + 8] = 	d_input[yOffsets[i/dim] + x + 3];
			neighborhood[i + 9] = 	d_input[yOffsets[i/dim] + x + 4];
			neighborhood[i + 10] = 	d_input[yOffsets[i/dim] + x + 5];			
		}
	}
	else
	{
		for (int i = 0; i < 60; i++)
		{
			neighborhood[i] = 0;
		}
		neighborhood[60] = d_input[yOffsets[5] + x];
		for (int i = 61; i < 121; i++)
		{
			neighborhood[i] = 255;
		}
	}

	//sort neighborhood
	QuickSort(neighborhood, 0, 11*11 - 1);
	
	// assign pixel to median

	d_output[yOffsets[5] + x] = neighborhood[60];

}

__global__ void medianFilter15( uint8_t *d_input, uint8_t *d_output) {
    // map from threadIdx/BlockIdx to pixel position^M
	int x = blockIdx.x * blockDim.x + threadIdx.x; //threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y; //blockIdx.y;
	int dim = 15;
	int dim_1d = dim*dim; //turning 2d square into single row length
	
	const int rowSize = gridDim.x * blockDim.x;

	int yOffsets[15];
	yOffsets[0] = (y-7) * rowSize;
	yOffsets[1] = (y-6) * rowSize;
	yOffsets[2] = (y-5) * rowSize;
	yOffsets[3] = (y -4)* rowSize;
	yOffsets[4] = (y-3)* rowSize;
	yOffsets[5] = (y-2)* rowSize;
	yOffsets[6] = (y-1)* rowSize;
	yOffsets[7] = (y)* rowSize;
	yOffsets[8] = (y+1)* rowSize;
	yOffsets[9] = (y+2)* rowSize;
	yOffsets[10] = (y+3)* rowSize;
	yOffsets[11] = (y+4)* rowSize;
	yOffsets[12] = (y+5)* rowSize;
	yOffsets[13] = (y+6)* rowSize;
	yOffsets[14] = (y+7)* rowSize;
	
	uint8_t neighborhood[15*15];
	
	if (y > 6 && y < (gridDim.y * blockDim.y - 7) && x > 6 && x < (rowSize - 7))
	{
		for(int i=0; i<dim_1d; i+=dim){
			neighborhood[i] = 	d_input[yOffsets[i/dim] + x - 7];
			neighborhood[i + 1] = 	d_input[yOffsets[i/dim] + x - 6];
			neighborhood[i + 2] = 	d_input[yOffsets[i/dim] + x - 5];
			neighborhood[i + 3] = 	d_input[yOffsets[i/dim] + x - 4];
			neighborhood[i + 4] = 	d_input[yOffsets[i/dim] + x - 3];
			neighborhood[i + 5] = 	d_input[yOffsets[i/dim] + x - 2];
			neighborhood[i + 6] = 	d_input[yOffsets[i/dim] + x - 1];
			neighborhood[i + 7] = 	d_input[yOffsets[i/dim] + x];
			neighborhood[i + 8] = 	d_input[yOffsets[i/dim] + x + 1];
			neighborhood[i + 9] = 	d_input[yOffsets[i/dim] + x + 2];
			neighborhood[i + 10] = 	d_input[yOffsets[i/dim] + x + 3];
			neighborhood[i + 11] = 	d_input[yOffsets[i/dim] + x + 4];
			neighborhood[i + 12] = 	d_input[yOffsets[i/dim] + x + 5];
			neighborhood[i + 13] = 	d_input[yOffsets[i/dim] + x + 6];
			neighborhood[i + 14] = 	d_input[yOffsets[i/dim] + x + 7];			
		}
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
	QuickSort(neighborhood, 0, 15 * 15 - 1);
	
	// assign pixel to median

	d_output[yOffsets[7] + x] = neighborhood[112];
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
    
    //start time
    //std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> median(height * width);
    uint8_t *d_input, *d_output;
    cudaMalloc((void **) &d_input, height * width * sizeof(uint8_t));
    cudaMalloc((void **) &d_output, height * width * sizeof(uint8_t));
	//copy the image that we read, into d_input and send it over to the GPU's memory
    cudaMemcpy(d_input, &mat[0], height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // TODO - Fill median.
	dim3 grid(64, 64);
    dim3 block(8, 8);
    
	if (dim == 3)
	{
		medianFilter3<<<grid,block>>>(d_input, d_output);
	}
	else if (dim == 7)
	{
		medianFilter7<<<grid,block>>>(d_input, d_output);
	}
	else if (dim == 11)
	{
		medianFilter11<<<grid,block>>>(d_input, d_output);
	}
	else if (dim == 15)
	{
		medianFilter15<<<grid,block>>>(d_input, d_output);
	}
	else
	{
		std::cout << "Unsuported Filter Size" << std::endl;
		return 1;
	}
	cudaMemcpy(&median[0], d_output, height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_output);
	
	//end time
	//std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	
	//std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(end - start);
	//std::cout << "Time:  " << time_span_wall.count() << " seconds." << std::endl;

    //Writes the new pgm picture
    fp = fopen(argv[3], "w");
    fprintf(fp, "%s\n%d\n%d\n%d\n", magic_number, height, width, gray_scale);
    for (int i=0;i<median.size();i++){
        	//printf("%c\n", median[i]);
		fputc(median[i], fp);
	}
    fclose(fp);

    return 0;
}
