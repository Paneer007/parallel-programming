#include <stdio.h>
#include <cuda.h>

__host__ __device__ void fun(int *counter)
{
    ++*counter;
}
__global__ void printk(int *counter)
{
    fun(counter);
    printf("printk (after fun): %d\n", *counter);
}
int main()
{
    int *counter;
    cudaHostAlloc(&counter, sizeof(int), 0);
    *counter = 0;
    printf("main: %d\n", *counter);
    printk<<<1, 1>>>(counter);
    cudaDeviceSynchronize();
    fun(counter);
    printf("main (after fun): %d\n", *counter);
    return 0;
}