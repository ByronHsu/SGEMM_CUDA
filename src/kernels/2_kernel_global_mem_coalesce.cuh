#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  int x = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
  int y = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  float tmp = 0.0;
  for(int i = 0; i < N; i++) {
    tmp += A[x * K + i] * B[i * N + y]; // A[x][i] * B[i][y]
  }
  tmp = alpha * tmp + beta * C[x * N + y]; // C[x][y]

  C[x * N + y] = tmp;
}