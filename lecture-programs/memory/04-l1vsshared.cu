#include <cuda.h>
#include <stdlib.h>

#define BLOCKSIZE 1024

__global__ void dkernel()
{
    __shared__ unsigned data[BLOCKSIZE];
    data[threadIdx.x] = threadIdx.x;
}
int main()
{
    cudaFuncSetCacheConfig(dkernel, cudaFuncCachePreferL1);
    // cudaFuncSetCacheConfig(dkernel, cudaFuncCachePreferShared);
    dkernel<<<1, BLOCKSIZE>>>();
    cudaDeviceSynchronize();
}