#include <cuda_runtime.h>
#include "playground/matmul.hpp"
#include "playground/system.hpp"

#define BLOCK_M 64
#define BLOCK_N 64  
#define BLOCK_K 64  
#define STRIDE 4
#define PADDING 1

namespace playground
{

__global__ void Float4_v2(float *C, const float *A, const float *B, int M, int K, int N)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float4 sA[BLOCK_M][BLOCK_K / 4 + PADDING];
    __shared__ float4 sB[BLOCK_K][BLOCK_N / 4 + PADDING];
    float regC[STRIDE][STRIDE] = {0.0f};
    int C_row_start = by * BLOCK_M;
    int C_col_start = bx * BLOCK_N;

    const float4* A_vec = (const float4*)A;
    const float4* B_vec = (const float4*)B;

    const int M_STEP = BLOCK_M / 16;          // 4
    const int K_VEC_STEP = (BLOCK_K / 4) / 16; // 1 
    const int N_VEC_STEP = (BLOCK_N / 4) / 16; // 1
    const int K_STEP = BLOCK_K / 16;          // 4

    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K)
    {      
        #pragma unroll
        for (int i = 0; i < M_STEP; i++) // M维4次
        {
            #pragma unroll
            for (int j = 0; j < K_VEC_STEP; j++) // K维1次
            {
                int sA_row = ty * M_STEP + i;
                int sA_col_vec = tx * K_VEC_STEP + j;
                int global_rowA = C_row_start + sA_row;
                int global_colA_vec = (k_tile / 4) + sA_col_vec;
                
                float4 data_vec;
                if (global_rowA < M && (k_tile + sA_col_vec * 4) < K)
                    data_vec = A_vec[global_rowA * (K / 4) + global_colA_vec];
                else
                    data_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                
                sA[sA_row][sA_col_vec] = data_vec;
            }
        }
       
        #pragma unroll
        for (int i = 0; i < K_STEP; i++) // K维4次
        {
            #pragma unroll
            for (int j = 0; j < N_VEC_STEP; j++) // N维1次
            {
                int sB_row = ty * K_STEP + i;
                int sB_col_vec = tx * N_VEC_STEP + j;
                int global_rowB = k_tile + sB_row;
                int global_colB_vec = (C_col_start / 4) + sB_col_vec;
               
                float4 data_vec;
                if (global_rowB < K && (C_col_start + sB_col_vec * 4) < N)
                    data_vec = B_vec[global_rowB * (N / 4) + global_colB_vec];
                else
                    data_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
               
                sB[sB_row][sB_col_vec] = data_vec;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k)
        {
            float regA[STRIDE];  //4*1的向量
            float4 regB_vec;   //1*4的向量
           
            #pragma unroll
            for (int i = 0; i < STRIDE; ++i)
            {
                int sA_col_vec = k / 4;
                int sA_comp = k % 4;
                regA[i] = ((float*)&sA[ty * STRIDE + i][sA_col_vec])[sA_comp];
            }

            int sB_col_vec = tx * STRIDE / 4;
            regB_vec = sB[k][sB_col_vec];

            #pragma unroll
            for (int i = 0; i < STRIDE; ++i)
            {
                regC[i][0] += regA[i] * regB_vec.x;
                regC[i][1] += regA[i] * regB_vec.y;
                regC[i][2] += regA[i] * regB_vec.z;
                regC[i][3] += regA[i] * regB_vec.w;
            }
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

PLAYGROUND_MATMUL_DEC(float32_t, 8, m, n, k, A, B, C)
{
    dim3 block_dim(16, 16);
    dim3 grid_dim((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);

    Float4_v2<<<grid_dim, block_dim>>>(C, A, B, m, k, n);
    cudaDeviceSynchronize();
}

} // namespace playground