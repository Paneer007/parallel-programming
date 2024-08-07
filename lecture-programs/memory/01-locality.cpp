#include <bits/stdc++.h>

using namespace std;
using namespace chrono;
const int SIZE = 512; // Define matrix size (you can adjust this)

void initializeMatrix(vector<vector<int>> &matrix) {
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      matrix[i][j] = i + j;
    }
  }
}

void multiplyMatricesSpatial(vector<vector<int>> &A, vector<vector<int>> &B,
                             vector<vector<int>> &C) {
  auto start = high_resolution_clock::now();

  // Matrix multiplication with spatial locality
  for (int i = 0; i < SIZE; ++i) {
    for (int k = 0; k < SIZE; ++k) {
      for (int j = 0; j < SIZE; ++j) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  auto end = high_resolution_clock::now();
  auto duration = end - start;
  cout << "Time taken with spatial locality: " << duration.count()
       << " seconds\n";
}

void multiplyMatricesTemporal(const vector<vector<int>> &A,
                              const vector<vector<int>> &B,
                              vector<vector<int>> &C) {
  auto start = high_resolution_clock::now();

  // Matrix multiplication with temporal locality
  vector<int> B_col(SIZE);
  for (int j = 0; j < SIZE; ++j) {
    for (int k = 0; k < SIZE; ++k) {
      B_col[k] = B[k][j];
    }

    for (int i = 0; i < SIZE; ++i) {
      C[i][j] = 0;
      for (int k = 0; k < SIZE; ++k) {
        C[i][j] += A[i][k] * B_col[k];
      }
    }
  }

  auto end = high_resolution_clock::now();
  auto duration = end - start;
  cout << "Time taken with temporal locality: " << duration.count()
       << " seconds\n";
}

int main() {
  // Initialize matrices
  vector<vector<int>> A(SIZE, vector<int>(SIZE));
  vector<vector<int>> B(SIZE, vector<int>(SIZE));
  vector<vector<int>> C(SIZE, vector<int>(SIZE, 0));

  initializeMatrix(A);
  initializeMatrix(B);

  multiplyMatricesSpatial(A, B, C);

  // Reset C matrix for the next test
  fill(C.begin(), C.end(), vector<int>(SIZE, 0));

  multiplyMatricesTemporal(A, B, C);

  return 0;
}
