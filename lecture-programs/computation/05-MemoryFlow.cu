#include<stdio.h>
#include<cuda.h>
#define N 10

__global__ void fun(int *a){
    a[threadIdx.x] = threadIdx.x * threadIdx.x;
}

int main() {
 int a[N], *da;
 int i;
 // Allocated memory on cuda device
 cudaMalloc(&da, N * sizeof(int));
 fun<<<1, N>>>(da);
 // Transfer content of memory from cuda device to CPU memory
 cudaMemcpy(a, da, N * sizeof(int),
 cudaMemcpyDeviceToHost);
 for (i = 0; i < N; ++i)
 printf("%d\n", a[i]);
 return 0;
} 

/*
Output:
0
1
4
9
16
25
36
49
64
81
*/

// Note CPU and GPU memory are not associated