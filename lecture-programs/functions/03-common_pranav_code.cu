#include <cuda.h>
#include <stdlib.h>

#define N 32

__host__ __device__ void fun(int *arr, int nn)
{
    for (unsigned ii = 0; ii < nn; ++ii)
        ++arr[ii];
}
__global__ void dfun(int *arr)
{
    fun(arr + threadIdx.x, 1);
    // need to change for more blocks.
}
int main()
{
    int arr[N], *darr;
    cudaMalloc(&darr, N * sizeof(int));
    for (unsigned ii = 0; ii < N; ++ii)
        arr[ii] = ii;
    cudaMemcpy(darr, arr, N * sizeof(int),
               cudaMemcpyHostToDevice);
    fun(arr, N);
    dfun<<<1, N>>>(darr);
    cudaDeviceSynchronize();
    return 0;
}