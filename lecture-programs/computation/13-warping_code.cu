#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dkernel(unsigned *vector, unsigned vectorsize)
{
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    // Thread Divergence
    if (id < vectorsize)
    {
        if (id % 2)
        {
            vector[id] = id;
        }
        else
        {
            vector[id] = vectorsize * vectorsize;
        }
        vector[id]++;
    }

    // Code to avoid thread divergence
    int x = 2;
    int y = vector[id];

    vector[id] =  x^y^vector[id];
}

// Diverges 
__global__ void dkernel(unsigned *vector, unsigned vectorsize)
{

    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned ii = 0; ii < id; ++ii)
        vector[id] += ii;
}


int main()
{
    const unsigned vectorsize = 32; // Example size, you can change it
    const unsigned size_in_bytes = vectorsize * sizeof(unsigned);

    unsigned *h_vector = (unsigned *)malloc(size_in_bytes);

    unsigned *d_vector;
    cudaMalloc(&d_vector, size_in_bytes);

    cudaMemcpy(d_vector, h_vector, size_in_bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 1, 1);
    dim3 gridDim((vectorsize + blockDim.x - 1) / blockDim.x, 1, 1);

    // Launch kernel
    dkernel<<<gridDim, blockDim>>>(d_vector, vectorsize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_vector, d_vector, size_in_bytes, cudaMemcpyDeviceToHost);

    printf("Result vector:\n");
    for (unsigned i = 0; i < vectorsize; i++)
    {
        printf("%u ", h_vector[i]);
    }
    printf("\n");

    return 0;
}
