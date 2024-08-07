#include <iostream>
#include <cuda_runtime.h>

const int MATRIX_SIZE = 1024;
const int THREADS_PER_BLOCK_X = 1024; // Number of threads per block in the x-dimension

__global__ void updateMatrix(int *d_M, int numCols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    int idx = row * numCols + col;

    extern __shared__ int sharedRow[];

    if (col < numCols) {
        sharedRow[col] = d_M[idx];
        if (col < numCols - 1) {
            sharedRow[col + 1] = d_M[idx + 1];
        }
    }

    __syncthreads();

    if (col < numCols - 1) {
        d_M[idx] = sharedRow[col] + sharedRow[col + 1];
    }
}

int main() {
    int size = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);

    int *h_M = new int[MATRIX_SIZE * MATRIX_SIZE];

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            h_M[i * MATRIX_SIZE + j] = i * MATRIX_SIZE + j;
        }
    }

    int *d_M;
    cudaMalloc(&d_M, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);

    int blockSize = THREADS_PER_BLOCK_X;
    int numBlocks = MATRIX_SIZE; // One block per row
    size_t sharedMemorySize = blockSize * sizeof(int);

    updateMatrix<<<numBlocks, blockSize, sharedMemorySize>>>(d_M, MATRIX_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_M, d_M, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) { // Print first 10 rows
        for (int j = 0; j < 10; ++j) { // Print first 10 columns
            std::cout << h_M[i * MATRIX_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
