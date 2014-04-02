#include "hw5.hpp"

float V_dissimilarity(data_v * V1, data_v * V2, int k) 
{
	float temp = 0;
	int i;
	
	for (i = 0; i < k; ++i)
	{
		temp += fabs((*V1)[i] - (*V2)[i]);
	}
	
	temp = temp / k;
	
	return temp;
}

void process_row(data_v * search_v, Row * row, int result_size, int offset, int search_size)
{
	/*
	for (i = 0; i < row.size; i += offset)
	{
		partition = &row.data[i]
		row.result.add(v_dissimilarity(parition, search_v, search_size));
	}
	sort(row.result);
	truncate(row.result,result_size)
	
	*/
	
}

void row_consumer()
{
	
}