#include <iostream>
#include <vector>

#include <cstdio>
#include <cstdlib>

typedef unsigned char uint8_t;

int main (int argc, char *argv[]) {

    if (argc != 3) // Change me per specs
        return 1;

    int height, width;
    char magic_number[4], input[10];
    int gray_scale;

    //Reads from argv[1] the input pgm file
    FILE *fp = fopen(argv[1],"r");
    fgets(magic_number, 4, fp);
    magic_number[2] = '\0';
    fgets(input, 10, fp);
    height = atoi(input);
    fgets(input, 10, fp);
    width = atoi(input);
    fgets(input, 10, fp);
    gray_scale = atoi(input);

    std::vector<uint8_t> mat(height * width);
    //Populates the arrays
    for (int i= 0; i < height * width; i++)
        mat[i] = fgetc(fp);

    fclose(fp);

    std::vector<uint8_t> median(height * width);
    uint8_t *d_input, *d_output;
    cudaMalloc((void **) &d_input, height * width * sizeof(uint8_t));
    cudaMalloc((void **) &d_output, height * width * sizeof(uint8_t));
    cudaMemcpy(d_input, &mat[0], height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // TODO - Fill median.
	
	for (int i = 0; i < height * width; i++)
		median.push_back(mat[i]);
		//median[i] = mat[i];
	
    cudaMemcpy(&median[0], d_output, height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    //Writes the new pgm picture
    fp = fopen(argv[2], "w");
    fprintf(fp, "%s\n%d\n%d\n%d\n", magic_number, height, width, gray_scale);
    for (int i=0;i<median.size();i++)
        fputc(median[i], fp);
    fclose(fp);

    return 0;
}
