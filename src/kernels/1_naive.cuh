#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.0;
  for(int i = 0; i < N; i++) {
    tmp += A[x * K + i] * B[i * N + y]; // A[x][i] * B[i][y]
  }
  tmp = alpha * tmp + beta * C[x * N + y]; // C[x][y]

  C[x * N + y] = tmp;
}