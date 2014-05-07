__kernel void add4Vector(
	__global float4* a,
	__global float4* b,
	__global float4* c,
	const unsigned int size)
{
	unsigned int i = get_global_id(0);
	if(i < size)
		c[i] = a[i] + b[i];
}

