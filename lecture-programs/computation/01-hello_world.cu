#include <stdio.h>
#include <cuda.h>

__global__ void dkernel() {
 printf("Hello World.\n");
}

int main() {
 dkernel<<<1, 1>>>();
 dkernel<<<1, 1>>>();
 dkernel<<<1, 1>>>();
 cudaDeviceSynchronize();
 return 0;
}

/*
Output:
Hello World.
Hello World.
Hello World.
*/