#include <stdio.h>
#include <cuda.h>

__global__ void dynshared()
{
    extern __shared__ int s[];
    s[threadIdx.x] = threadIdx.x;
    __syncthreads();
    if (threadIdx.x % 2)
        printf("%d\n", s[threadIdx.x]);
}

int main()
{
    int n;
    scanf("% d", &n);
    dynshared<<<1, n, n * sizeof(int)>>>();
    cudaDeviceSynchronize();
    return 0;
}
#