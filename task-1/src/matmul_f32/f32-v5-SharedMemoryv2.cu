#include <cuda_runtime.h>
#include "playground/matmul.hpp"
#include "playground/system.hpp"

#define BLOCK_SIZE 16   
#define STRIDE 2      
#define STEP (BLOCK_SIZE * STRIDE)  

namespace playground
{

__global__ void SharedMemory_v2(float *C, const float *A, const float *B, int hC, int wA, int wC)
{
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    float *A_start = (float*)A + STEP * blockIdx.y * wA;  
    float *B_start = (float*)B + STEP * blockIdx.x;     
    
    __shared__ float sA[STEP][STEP];  
    __shared__ float sB[STEP][STEP];  
    float temp[STRIDE][STRIDE] = {0.0f};

    for (int s = 0; s < wA; s += STEP)
    {
        for (int i = 0; i < STRIDE; i++)
        {
            for (int j = 0; j < STRIDE; j++)
            {
                sA[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = A_start[(ty + BLOCK_SIZE * i) * wA + tx + BLOCK_SIZE * j + s];
                sB[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = B_start[(ty + BLOCK_SIZE * i + s) * wC + tx + BLOCK_SIZE * j];
            }
        }
        __syncthreads();
        for (int i = 0; i < STRIDE; i++)
        {
            for (int j = 0; j < STRIDE; j++)
            {
                for (int k = 0; k < STEP; k++)
                {
                    temp[i][j] += sA[ty + i * BLOCK_SIZE][k] * sB[k][tx + j * BLOCK_SIZE];
                }
            }
        }
        
        __syncthreads();
    }

    float *C_start = C + wC * blockIdx.y * STEP + blockIdx.x * STEP;
    for (int i = 0; i < STRIDE; i++)
    {
        for (int j = 0; j < STRIDE; j++)
        {    
            C_start[wC * (ty + i * BLOCK_SIZE) + tx + j * BLOCK_SIZE] = temp[i][j];
        }
    }
}

PLAYGROUND_MATMUL_DEC(float32_t, 5, m, n, k, A, B, C)
{
    dim3 block_Dim(16, 16);  //256
    dim3 grid_Dim( 4096/16/2, 4096/16/2);  

    SharedMemory_v2<<<grid_Dim, block_Dim>>>(C, A, B, m, k, n);
    cudaDeviceSynchronize();
}

} // namespace playground