#include <stdio.h>
#include <cuda.h>

__global__ void dkernel() {
    if (threadIdx.x == 0 ) {
            printf("%d %d %d %d %d %d.\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
        }
}

int main() {
    dim3 grid(2, 3, 4);
    dim3 block(5, 6, 7);
    dkernel<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}