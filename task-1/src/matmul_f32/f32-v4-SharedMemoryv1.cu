#include <cuda_runtime.h>
#include "playground/matmul.hpp"
#include "playground/system.hpp"
#define tile_size 16 //分成16块

namespace playground
{

__global__ void SharedMemory_v1(float *C, const float *A, const float *B, int hC, int wA, int wC)
{
    __shared__ float sA[tile_size][tile_size];  //共享内存大小
    __shared__ float sB[tile_size][tile_size];

    int row = blockDim.y * blockIdx.y + threadIdx.y; //->Cx,y轴不是连续的
    int col = blockDim.x * blockIdx.x + threadIdx.x; //->Cy,x轴对应水平方向宽度，所以是列索引，同时C++的行主序让沿着x轴，列索引是连续的，所以x轴对应列索引

    float sum = 0; 
    for (int s = 0; s < wA; s +=tile_size)
    {
        sA[threadIdx.y][threadIdx.x] = A[row * wA + threadIdx.x +s];
        sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + s)* wC + col];
        __syncthreads();
        for (int k = 0; k < tile_size; k++)
        {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * wC + col] = sum;

}

PLAYGROUND_MATMUL_DEC(float32_t, 4, m, n, k, A, B, C)
{
    dim3 block_Dim(16, 16);  //256 threads per block
    dim3 grid_Dim(4096 / 16, 4096 / 16);

    SharedMemory_v1<<<grid_Dim, block_Dim>>>(C, A, B, m, k, n);
    cudaDeviceSynchronize(); //同步
}
} 
