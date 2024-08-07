#include<stdio.h>
#include<cuda.h>

__global__ void dkernel(){
    printf("Hello World. \n");
}

int main(){
    dkernel<<<1, 1>>>();
    printf("CPU one\n");
    dkernel<<<1, 1>>>();
    printf("CPU two\n");
    dkernel<<<1, 1>>>();
    printf("CPU three\n");
    cudaDeviceSynchronize();
    printf("on CPU\n");
    return 0;
}

/*
Output:
CPU one
CPU two
CPU three
Hello World. 
Hello World. 
Hello World.
*/