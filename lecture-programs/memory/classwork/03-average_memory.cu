#include <iostream>
#include <cuda_runtime.h>

const int N = 1024; // Size of the array (number of points)
const int THREADS_PER_BLOCK = 256; // Number of threads per block

struct Point {
    int x, y;
};

// Kernel to find average and process points
__global__ void processPoints(Point *d_arr, int *d_sum, int *d_count, int numPoints) {
    __shared__ int shared_x[THREADS_PER_BLOCK * 4];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int threadId = threadIdx.x;

    if (tid < numPoints) {
        // Load data into shared memory
        for (int i = 0; i < 4 && tid * 4 + i < numPoints; ++i) {
            shared_x[threadId * 4 + i] = d_arr[tid * 4 + i].x;
        }

        __syncthreads();

        // Compute average of x values
        int local_avg = 0;
        if (tid * 4 < numPoints) {
            for (int i = 0; i < 4 && tid * 4 + i < numPoints; ++i) {
                local_avg += shared_x[threadId * 4 + i];
            }
            local_avg /= min(4, numPoints - tid * 4);
        }

        // Process y values
        for (int i = 0; i < 4 && tid * 4 + i < numPoints; ++i) {
            Point &p = d_arr[tid * 4 + i];
            if (p.y > local_avg) {
                p.y = local_avg;
                atomicAdd(d_count, 1);
            } else {
                atomicAdd(d_sum, p.y);
            }
        }
    }
}

int main() {
    // Allocate and initialize host memory
    Point *h_arr = new Point[N];
    for (int i = 0; i < N; ++i) {
        h_arr[i].x = i % 100;
        h_arr[i].y = i % 100;
    }

    Point *d_arr;
    int *d_sum, *d_count;
    int h_sum = 0, h_count = 0;

    // Allocate device memory
    cudaMalloc(&d_arr, N * sizeof(Point));
    cudaMalloc(&d_sum, sizeof(int));
    cudaMalloc(&d_count, sizeof(int));

    // Initialize global variables on the device
    cudaMemset(d_sum, 0, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_arr, h_arr, N * sizeof(Point), cudaMemcpyHostToDevice);

    // Compute average on host
    int h_avg = 0;
    for (int i = 0; i < N; ++i) {
        h_avg += h_arr[i].x;
    }
    h_avg /= N;

    // Launch the kernel
    int blocks = (N + THREADS_PER_BLOCK * 4 - 1) / (THREADS_PER_BLOCK * 4);
    processPoints<<<blocks, THREADS_PER_BLOCK>>>(d_arr, d_sum, d_count, N);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Number of elements set to AVG: " << h_count << std::endl;
    std::cout << "Sum of y values not replaced: " << h_sum << std::endl;

    // Free memory
    delete[] h_arr;
    cudaFree(d_arr);
    cudaFree(d_sum);
    cudaFree(d_count);

    return 0;
}
