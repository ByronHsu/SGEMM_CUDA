#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  int C_block_row = blockIdx.x;
  int C_block_col = blockIdx.y;

  int C_thread_row = threadIdx.x / (BN / TN);
  int C_thread_col = threadIdx.x % (BN / TN);

  int num_threads_per_block = (BM * BN / (TM * TN));

  int local_A_row = threadIdx.x / BK;
  int local_A_col = threadIdx.x % BK;
  int local_A_row_stride = num_threads_per_block / BK;

  int local_B_row = threadIdx.x / BN;
  int local_B_col = threadIdx.x % BN;
  int local_B_row_stride = num_threads_per_block / BN;

  int local_C_row_offset = C_thread_row * TM;
  int local_C_col_offset = C_thread_col * TN;

  __shared__ float shm_A[BM * BK];
  __shared__ float shm_B[BK * BN];

  float tmp[TM * TN] = {0.0f};

  // initial advance
  A += C_block_row * BM * K;
  B += C_block_col * BN;
  C += C_block_row * BM * N + C_block_col * BN;

  for(int k = 0; k < K; k += BK) {
    
    for(int i = 0; i < BM; i += local_A_row_stride) {
      shm_A[(local_A_row + i) * BK + local_A_col] = A[(local_A_row + i) * K + local_A_col];
    }

    for(int i = 0; i < BK; i += local_B_row_stride) {
      shm_B[(local_B_row + i) * BN + local_B_col] = B[(local_B_row + i) * N + local_B_col];
    }

    __syncthreads();


    float A_col[TM]; 
    float B_row[TN];


    // calculate: 
    //  1. load a slice into reg
    //  2. outer product reg A and reg B
    // for(int i = 0; i < TM; i++) {
    //   for(int j = 0; j < TN; j++) {
    //     for(int k = 0; k < BK; k++) {
    //       tmp[i * TN + j] += shm_A[(local_C_row_offset + i) * BK + k] * shm_B[k * BN + (local_C_col_offset + j)];
    //     }
    //   }
    // }

    // out product
    for(int k = 0; k < BK; k++) {
      for(int i = 0; i < TM; i++) {
        A_col[i] = shm_A[(local_C_row_offset + i) * BK + k];
      }
      for(int j = 0; j < TN; j++) {
        B_row[j] = shm_B[k * BN + (local_C_col_offset + j)];
      }

      for(int i = 0; i < TM; i++) {
        for(int j = 0; j < TN; j++) {
          tmp[i * TN + j] += A_col[i] * B_row[j];
        }
      }
    }

    // advance A and B
    A += BK;
    B += BK * N;

    __syncthreads();
  }

  // write
  for(int i = 0; i < TM; i++) {
    for(int j = 0; j < TN; j++) {
      C[(local_C_row_offset + i) * N + (local_C_col_offset + j)] = alpha * tmp[i * TN + j] + beta * C[(local_C_row_offset + i) * N + (local_C_col_offset + j)]; 
    }
  }
}