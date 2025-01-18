#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {

  int block_row = blockIdx.x;
  int block_col = blockIdx.y;

  int thread_row = threadIdx.x / BN;
  int thread_col = threadIdx.x % BN;

  int A_row = threadIdx.x / BK;
  int A_col = threadIdx.x % BK;

  int B_row = threadIdx.x / BN;
  int B_col = threadIdx.x % BN;

  // initial advance
  A += block_row * BM * K;
  B += block_col * BN;
  C += block_row * BM * N + block_col * BN;

  __shared__ float shm_A[BM * BK];
  __shared__ float shm_B[BK * BN];

  float tmp[TM] = {0.0f};

  // outer loop: advance on dim k
  for(int k = 0; k < K; k += BK){
    // load data into shm A and B
    shm_A[A_row * BK + A_col] = A[A_row * K + A_col];
    shm_B[B_row * BN + B_col] = B[B_row * N + B_col];

    __syncthreads();

    for(int t = 0; t < TM; t++) { // for each ele in the col tile
      for(int i = 0; i < BK; i++){ // accumulate
        tmp[t] += shm_A[(thread_row * TM + t) * BK + i] * shm_B[i * BN + thread_col];
      }
    }

    // advance A and B
    A += BK;
    B += BK * N;

    __syncthreads();
  }
  
  for(int t = 0; t < TM; t++) {
    C[(thread_row * TM + t) * N + thread_col] = alpha * tmp[t] + beta * C[(thread_row * TM + t) * N + thread_col];
  }
}