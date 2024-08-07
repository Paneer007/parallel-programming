#include <stdio.h>
#include <cuda.h>

__device__ int lockvar;
__global__ void k1()
{
    while (atomicCAS(&lockvar, 0, 1))
        ;
    printf("Block %d, Thread %d is executing critical section.\n", blockIdx.x, threadIdx.x);
    lockvar = 0;
}
int main()
{
    cudaMemset(&lockvar, 0, sizeof(int)); // lock initialization.
    k1<<<64, 1>>>();
    // k1<<<2, 32>>>();	// This doesn't work.
    cudaDeviceSynchronize();

    return 0;
}