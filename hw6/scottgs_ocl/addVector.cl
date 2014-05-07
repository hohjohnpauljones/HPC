__kernel void addVector(
	__global float* a,
	__global float* b,
	__global float* c,
	const unsigned int size)
{
	unsigned int i = get_global_id(0);
	if(i < size)
		c[i] = a[i] + b[i];
}

