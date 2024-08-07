#include <stdio.h>
#include <cuda.h>

__global__ void initializeArray(int *arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = 0;
    }
}

__global__ void addIndex(int *arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] += idx;
    }
}

int main() {
    int size = 1024;
    int *gpuarr;

    cudaMalloc(&gpuarr, size * sizeof(int));

    initializeArray<<<1, size>>>(gpuarr, size);
    cudaDeviceSynchronize();

    // Add the index value to each element in the array
    addIndex<<<1, size>>>(gpuarr, size);
    cudaDeviceSynchronize();

    int *cpuarr = (int *)malloc(size * sizeof(int));
    cudaMemcpy(cpuarr, gpuarr, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the array
    for (int i = 0; i < size; i++) {
        printf("%d ", cpuarr[i]);
    }
    printf("\n");

    return 0;
}

/*
Output: 
size == 32:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 
size == 1024
0 1 ... 1023
size == 8000
all zeroes
*/