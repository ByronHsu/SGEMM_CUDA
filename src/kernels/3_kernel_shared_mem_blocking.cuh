#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  int t_x = threadIdx.x / BLOCKSIZE;
  int t_y = threadIdx.x % BLOCKSIZE;

  int c_x = blockIdx.x * BLOCKSIZE + t_x;
  int c_y = blockIdx.y * BLOCKSIZE + t_y;

  __shared__ float shm_A[BLOCKSIZE * BLOCKSIZE];
  __shared__ float shm_B[BLOCKSIZE * BLOCKSIZE];
  float tmp = 0.0;
  
  for(int k = 0; k < K; k += BLOCKSIZE) {
    // C_blk[blk_x][blk_y] = sum(A_blk[blk_x][blk_k] @ B[blk_k][blk_y])

    // 1. Fill shm A and B
    shm_A[t_x * BLOCKSIZE + t_y] = A[c_x * K + (t_y + k)];
    shm_B[t_x * BLOCKSIZE + t_y] = B[(t_x + k) * N + c_y];

    __syncthreads();

    // 2. matmul shm A and B
    for(int i = 0; i < BLOCKSIZE; i++) {
      tmp += shm_A[t_x * BLOCKSIZE + i] * shm_B[i * BLOCKSIZE + t_y];
    }

    __syncthreads();
  }

  C[c_x * N + c_y] = alpha * tmp + beta * C[c_x * N + c_y];
}