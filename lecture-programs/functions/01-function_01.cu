#include <stdio.h>
#include <cuda.h>
__host__ __device__ void dhfun()
{
    printf("I can run on both CPU and GPU.\n");
}
__device__ unsigned dfun(unsigned *vector, unsigned vectorsize, unsigned id)
{
    if (id == 0)
        dhfun();
    if (id < vectorsize)
    {
        vector[id] = id;
        return 1;
    }
    else
    {
        return 0;
    }
}
__global__ void dkernel(unsigned *vector, unsigned vectorsize)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    dfun(vector, vectorsize, id);
}
__host__ void hostfun()
{
    printf("I am simply like another function running on CPU. Calling dhfun\n");
    dhfun();
}

int main(){
    hostfun();
}