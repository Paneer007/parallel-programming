#include <iostream>
#include <cuda.h>

using namespace std;

__global__ void squareArray(const int *input, int *output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
}

__global__ void cubeArray(const int *input, int *output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int value = input[idx];
        output[idx] = value * value * value;
    }
}

__global__ void sumArrays(const int *squared, const int *cubed, int *result, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        result[idx] = squared[idx] + cubed[idx];
    }
}

int main() {
    const int size = 10;  // Size of the arrays
    const int arraySize = size * sizeof(int);

    // Host arrays
    int h_array1[size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int h_array2[size] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    int h_squared[size];
    int h_cubed[size];
    int h_result[size];

    // Device arrays
    int *d_array1, *d_array2, *d_squared, *d_cubed, *d_result;

    // Allocate device memory
    cudaMalloc(&d_array1, arraySize);
    cudaMalloc(&d_array2, arraySize);
    cudaMalloc(&d_squared, arraySize);
    cudaMalloc(&d_cubed, arraySize);
    cudaMalloc(&d_result, arraySize);

    // Copy host data to device
    cudaMemcpy(d_array1, h_array1, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, h_array2, arraySize, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch kernels
    squareArray<<<numBlocks, blockSize>>>(d_array1, d_squared, size);
    cubeArray<<<numBlocks, blockSize>>>(d_array2, d_cubed, size);
    sumArrays<<<numBlocks, blockSize>>>(d_squared, d_cubed, d_result, size);

    // Copy results back to host
    cudaMemcpy(h_squared, d_squared, arraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cubed, d_cubed, arraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result, d_result, arraySize, cudaMemcpyDeviceToHost);

    // Print results
    cout << "Array 1 (squared): ";
    for (int i = 0; i < size; ++i) {
        cout << h_squared[i] << " ";
    }
    cout << endl;

    cout << "Array 2 (cubed): ";
    for (int i = 0; i < size; ++i) {
        cout << h_cubed[i] << " ";
    }
    cout << endl;

    cout << "Result (sum of squared and cubed arrays): ";
    for (int i = 0; i < size; ++i) {
        cout << h_result[i] << " ";
    }
    cout << endl;

    return 0;
}
