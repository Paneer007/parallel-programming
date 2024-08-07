#include <iostream>
#include <cuda_runtime.h>

struct nodeAOS {
    int a;
    double b;
    char c;
} *allnodesAOS;

struct nodeSOA {
    int *a;
    double *b;
    char *c;
} allnodesSOA;

__global__ void dkernelaos(nodeAOS *allnodesAOS) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    allnodesAOS[id].a = id;
    allnodesAOS[id].b = 0.0;
    allnodesAOS[id].c = 'c';
}

__global__ void dkernelsoa(int *a, double *b, char *c) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    a[id] = id;
    b[id] = 0.0;
    c[id] = 'd';
}

int main() {
    const int N = 10; // Define the number of elements
    const size_t size = N * sizeof(int);

    // Allocate host memory for AOS
    nodeAOS *h_allnodesAOS = new nodeAOS[N];

    // Allocate host memory for SOA
    nodeSOA h_allnodesSOA;
    h_allnodesSOA.a = new int[N];
    h_allnodesSOA.b = new double[N];
    h_allnodesSOA.c = new char[N];

    // Allocate device memory for AOS
    nodeAOS *d_allnodesAOS;
    cudaMalloc(&d_allnodesAOS, N * sizeof(nodeAOS));

    // Allocate device memory for SOA
    int *d_a;
    double *d_b;
    char *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, N * sizeof(double));
    cudaMalloc(&d_c, N * sizeof(char));

    dkernelaos<<<N, 1>>>(d_allnodesAOS);
    cudaDeviceSynchronize();

    dkernelsoa<<<N, 1>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // Copy data from device to host
    cudaMemcpy(h_allnodesAOS, d_allnodesAOS, N * sizeof(nodeAOS), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_allnodesSOA.a, d_a, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_allnodesSOA.b, d_b, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_allnodesSOA.c, d_c, N * sizeof(char), cudaMemcpyDeviceToHost);

    // Print results for AOS
    std::cout << "Array of Structures (AOS) data:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "Node " << i << ": a = " << h_allnodesAOS[i].a << ", b = " << h_allnodesAOS[i].b << ", c = " << h_allnodesAOS[i].c << std::endl;
    }

    // Print results for SOA
    std::cout << "Structure of Arrays (SOA) data:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "Node " << i << ": a = " << h_allnodesSOA.a[i] << ", b = " << h_allnodesSOA.b[i] << ", c = " << h_allnodesSOA.c[i] << std::endl;
    }

    // Free memory
    delete[] h_allnodesAOS;
    delete[] h_allnodesSOA.a;
    delete[] h_allnodesSOA.b;
    delete[] h_allnodesSOA.c;
    cudaFree(d_allnodesAOS);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
