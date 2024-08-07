#include<cuda.h>
#include<stdio.h>

#define N 10

__global__ void fun(){
    printf("%d \n", threadIdx.x *threadIdx.x);
}

int main(){
    fun<<<1,N>>>();
    cudaDeviceSynchronize();
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