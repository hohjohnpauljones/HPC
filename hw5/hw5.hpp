#include <math.h>
#include <stdio.h>
#include <vector>
#include <string>

using namespace std;

typedef std::vector<float> data_v;

struct result {
	float X;
	float Y;
	int offset;
	float distance;
	
	bool operator < (const result& str) const
    {
        return (distance < str.distance);
    }
};

struct Row {
	float X;
	float Y;
	data_v data;
	std::vector<result> results;
};

struct row_producer_param {
	string fname;
	std::vector<Row> * rows;
};



void row_producer(string fname, std::vector<Row> * rows);

void row_consumer();

void process_row(data_v * search_v, Row * row, int result_size, int offset, int search_size);

float V_dissimilarity(data_v * V1, data_v * V2, int k);

/*
void parse_file (name, n)
{
	vector<Row> rows;
	sem_available = 0;
	count_c = 0;
	
	spawn producer
	spawn n consumers;
	wait producer;
	wait consumer;
}
*/
