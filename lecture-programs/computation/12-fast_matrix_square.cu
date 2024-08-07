#include <stdio.h>
#include <cuda_runtime.h>

__global__ void square(unsigned *matrix, unsigned *result, unsigned matrixsize) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned ii = id / matrixsize;
    unsigned jj = id % matrixsize;

    if (ii < matrixsize && jj < matrixsize) {
        unsigned sum = 0;
        for (unsigned kk = 0; kk < matrixsize; ++kk) {
            sum += matrix[ii * matrixsize + kk] * matrix[kk * matrixsize + jj];
        }
        result[ii * matrixsize + jj] = sum;
    }
}

int main() {
    const unsigned matrixsize = 3; // Example size, you can change it
    const unsigned matrix_elements = matrixsize * matrixsize;
    const unsigned size_in_bytes = matrix_elements * sizeof(unsigned);

    unsigned *h_matrix = (unsigned *)malloc(size_in_bytes);
    unsigned *h_result = (unsigned *)malloc(size_in_bytes);

    for (unsigned i = 0; i < matrix_elements; ++i) {
        h_matrix[i] = i + 1;
    }

    printf("Source Matrix:\n");
    for (unsigned i = 0; i < matrixsize; ++i) {
        for (unsigned j = 0; j < matrixsize; ++j) {
            printf("%d ", h_matrix[i * matrixsize + j]);
        }
        printf("\n");
    }


    unsigned *d_matrix, *d_result;
    cudaMalloc(&d_matrix, size_in_bytes);
    cudaMalloc(&d_result, size_in_bytes);

    cudaMemcpy(d_matrix, h_matrix, size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, size_in_bytes);

    dim3 blockDim(16, 1, 1);
    dim3 gridDim((matrix_elements + blockDim.x - 1) / blockDim.x, 1, 1);

    square<<<gridDim, blockDim>>>(d_matrix, d_result, matrixsize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, size_in_bytes, cudaMemcpyDeviceToHost);

    printf("Result matrix:\n");
    for (unsigned i = 0; i < matrixsize; ++i) {
        for (unsigned j = 0; j < matrixsize; ++j) {
            printf("%u ", h_result[i * matrixsize + j]);
        }
        printf("\n");
    }

    free(h_matrix);
    free(h_result);
    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}
