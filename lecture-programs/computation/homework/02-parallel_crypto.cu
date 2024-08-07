#include <iostream>
#include <cuda.h>

#define N 1024 
using namespace std;

__global__ void encrypt1(char *input, char *output, int length) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < length) {
        char c = input[idx];
        output[idx] = (c == 'z') ? 'a' : c + 1;
    }
}

__global__ void encrypt2(char *input, char *output, int length) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < length) {
        char c = input[idx];
        char newChar = c + idx;
        // Handle wrap-around
        output[idx] = (newChar > 'z') ? (newChar - 26) : newChar;
    }
}

__global__ void decrypt1(char *input, char *output, int length) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < length) {
        char c = input[idx];
        output[idx] = (c == 'a') ? 'z' : c - 1;
    }
}

__global__ void decrypt2(char *input, char *output, int length) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < length) {
        char c = input[idx];
        char newChar = c - idx;
        // Handle wrap-around
        output[idx] = (newChar < 'a') ? (newChar + 26) : newChar;
    }
}

int main() {
    char h_message[N] = "hello";
    int length = strlen(h_message);
    
    char *d_input, *d_encrypted1, *d_encrypted2, *d_decrypted1, *d_decrypted2;

    cudaMalloc(&d_input, length * sizeof(char));
    cudaMalloc(&d_encrypted1, length * sizeof(char));
    cudaMalloc(&d_encrypted2, length * sizeof(char));
    cudaMalloc(&d_decrypted1, length * sizeof(char));
    cudaMalloc(&d_decrypted2, length * sizeof(char));

    // Copy data to device
    cudaMemcpy(d_input, h_message, length * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;

    // Encrypt and decrypt the message
    encrypt1<<<numBlocks, blockSize>>>(d_input, d_encrypted1, length);
    encrypt2<<<numBlocks, blockSize>>>(d_input, d_encrypted2, length);
    decrypt1<<<numBlocks, blockSize>>>(d_encrypted1, d_decrypted1, length);
    decrypt2<<<numBlocks, blockSize>>>(d_encrypted2, d_decrypted2, length);

    // Copy results back to host
    char h_encrypted1[N], h_encrypted2[N], h_decrypted1[N], h_decrypted2[N];
    cudaMemcpy(h_encrypted1, d_encrypted1, length * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_encrypted2, d_encrypted2, length * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_decrypted1, d_decrypted1, length * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_decrypted2, d_decrypted2, length * sizeof(char), cudaMemcpyDeviceToHost);

    // Print results
    cout << "Original message: " << h_message << endl;
    cout << "Encrypted (method 1): " << h_encrypted1 << endl;
    cout << "Encrypted (method 2): " << h_encrypted2 << endl;
    cout << "Decrypted (method 1): " << h_decrypted1 << endl;
    cout << "Decrypted (method 2): " << h_decrypted2 << endl;


    return 0;
}
