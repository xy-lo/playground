#include <cuda_runtime.h>
#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{

__global__ void naive_2D(float *C, const float *A, const float *B, int hC, int wA, int wC)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y; //->Cx,y轴不是连续的
    int col = blockDim.x * blockIdx.x + threadIdx.x; //->Cy,x轴对应水平方向宽度，所以是列索引，同时C++的行主序让沿着x轴，列索引是连续的，所以x轴对应列索引

    if(row < hC && col < wC)
    {
        float sum = 0;
        for (int i = 0; i < wA; i++)
            sum += A[row * wA + i] * B[i * wC + col];
        C[row * wC +col] = sum;
    }
}

PLAYGROUND_MATMUL_DEC(float32_t, 3, m, n, k, A, B, C)
{
    dim3 block_Dim(16, 16);  //256 threads per block
    dim3 grid_Dim(4096 / 16, 4096 / 16);

    naive_2D<<<grid_Dim, block_Dim>>>(C, A, B, m, n, k);
    cudaDeviceSynchronize(); 
}
} 