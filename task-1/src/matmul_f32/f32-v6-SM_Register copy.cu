#include <cuda_runtime.h>
#include "playground/matmul.hpp"
#include "playground/system.hpp"

#define BLOCK_M 64
#define BLOCK_N 64   //64/16=4
#define BLOCK_K 64   //32/16=2
#define STRIDE 4  // 每个线程计算 4x4 的结果
#define PADDING 1

namespace playground
{

__global__ void SharedMemory_v30(float *C, const float *A, const float *B, int M, int K, int N)
{
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float sA[BLOCK_M][BLOCK_K + PADDING]; 
    __shared__ float sB[BLOCK_K][BLOCK_N + PADDING]; 
    float regC[STRIDE][STRIDE] = {0.0f};
    int C_row_start = by * BLOCK_M;
    int C_col_start = bx * BLOCK_N;

    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K)
    {
        int start_m = ty * STRIDE;  // 0, 4, 8, ...
        int start_n = tx * STRIDE;  // 0, 4, 8, ...
        int start_k_a = tx * (BLOCK_K / 16); 
        int start_k_b = ty * (BLOCK_K / 16); 

        #pragma unroll
        for (int i = 0; i < BLOCK_M/16; i++) // M维4次
        {
            #pragma unroll
            for (int j = 0; j < (BLOCK_K/16); j++) // K维2次
            {
                int global_rowA = C_row_start + start_m + i;
                int global_colA = k_tile + start_k_a + j; 
                sA[start_m + i][start_k_a + j] = (global_rowA < M && global_colA < K) ? A[global_rowA * K + global_colA] : 0.0f;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < (BLOCK_K/16); i++) // K
        {
            #pragma unroll
            for (int j = 0; j < BLOCK_N/16; j++) // N
            {
                int global_rowB = k_tile + start_k_b + i;
                int global_colB = C_col_start + start_n + j;
                sB[start_k_b + i][start_n + j] = (global_rowB < K && global_colB < N) ? B[global_rowB * N + global_colB] : 0.0f;
            }
        }
        
        __syncthreads();
        #pragma unroll 
        for (int k = 0; k < BLOCK_K; ++k)
        {
            float regA[STRIDE]; 
            float regB[STRIDE]; 
            
            #pragma unroll
            for (int i = 0; i < STRIDE; ++i)
            {
                regA[i] = sA[ty * STRIDE + i][k]; 
                regB[i] = sB[k][tx * STRIDE + i];
            }
            
            #pragma unroll
            for (int i = 0; i < STRIDE; ++i)
                #pragma unroll
                for (int j = 0; j < STRIDE; ++j)
                    regC[i][j] += regA[i] * regB[j];
        }
        __syncthreads(); 
    }


    int base_row = C_row_start + ty * STRIDE;
    int base_col = C_col_start + tx * STRIDE;
    for (int i = 0; i < STRIDE; ++i)
    {
        for (int j = 0; j < STRIDE; ++j)
            C[(base_row + i) * N + (base_col + j)] = regC[i][j];
    }
}

PLAYGROUND_MATMUL_DEC(float32_t, 60, m, n, k, A, B, C)
{
    dim3 block_dim(16, 16); 
    dim3 grid_dim( n/16/4, m/16/4);
//设置最大96KB动态共享内存
    cudaFuncSetAttribute(SharedMemory_v30, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    // 计算动态共享内存大小
    unsigned int dsmem = 2 * (64 * (64 + 1) + 64 * (64 + 1)) * sizeof(half);
    SharedMemory_v30<<<grid_dim, block_dim,dsmem>>>(C, A, B, m, k, n);
    cudaDeviceSynchronize();
}

} // namespace playground