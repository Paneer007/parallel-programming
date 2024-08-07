#include <iostream>
#include <cuda.h>

using namespace std;

const int NUM_STUDENTS = 80;
const int NUM_GRADES = 6; // S, A, B, C, D, E

// CUDA kernel to assign grades based on marks
__global__ void assignGrades(int *marks, char *grades) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NUM_STUDENTS) {
        int mark = marks[idx];
        if (mark >= 90) grades[idx] = 'S';
        else if (mark >= 80) grades[idx] = 'A';
        else if (mark >= 70) grades[idx] = 'B';
        else if (mark >= 60) grades[idx] = 'C';
        else if (mark >= 50) grades[idx] = 'D';
        else if (mark >= 40) grades[idx] = 'E';
        else grades[idx] = 'U';
    }
}

// CUDA kernel to compute the histogram of grades
__global__ void computeHistogram(char *grades, int *histogram) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NUM_STUDENTS) {
        int gradeIndex = 0;
        switch (grades[idx]) {
            case 'S': gradeIndex = 0; break;
            case 'A': gradeIndex = 1; break;
            case 'B': gradeIndex = 2; break;
            case 'C': gradeIndex = 3; break;
            case 'D': gradeIndex = 4; break;
            case 'E': gradeIndex = 5; break;
            default: gradeIndex = -1; 
        }

        if (gradeIndex != -1) {
            atomicAdd(&histogram[gradeIndex], 1);
        }
    }
}

int main() {
    int h_marks[NUM_STUDENTS];
    char h_grades[NUM_STUDENTS];
    int h_histogram[NUM_GRADES] = {0};

    // Initialize student marks
    for (int i = 0; i < NUM_STUDENTS; ++i) {
        h_marks[i] = rand() % 101; // Random marks between 0 and 100
    }

    int *d_marks;
    char *d_grades;
    int *d_histogram;

    // Allocate device memory
    cudaMalloc(&d_marks, NUM_STUDENTS * sizeof(int));
    cudaMalloc(&d_grades, NUM_STUDENTS * sizeof(char));
    cudaMalloc(&d_histogram, NUM_GRADES * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_marks, h_marks, NUM_STUDENTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, NUM_GRADES * sizeof(int));

    // Launch kernels
    int threadsPerBlock = 32;
    int blocksPerGrid = (NUM_STUDENTS + threadsPerBlock - 1) / threadsPerBlock;
    
    assignGrades<<<blocksPerGrid, threadsPerBlock>>>(d_marks, d_grades);
    cudaDeviceSynchronize();

    computeHistogram<<<blocksPerGrid, threadsPerBlock>>>(d_grades, d_histogram);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_grades, d_grades, NUM_STUDENTS * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_histogram, d_histogram, NUM_GRADES * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    cout << "Histogram of Grades:" << endl;
    cout << "S: " << h_histogram[0] << endl;
    cout << "A: " << h_histogram[1] << endl;
    cout << "B: " << h_histogram[2] << endl;
    cout << "C: " << h_histogram[3] << endl;
    cout << "D: " << h_histogram[4] << endl;
    cout << "E: " << h_histogram[5] << endl;

    return 0;
}
