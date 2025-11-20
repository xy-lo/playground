#include <cuda_runtime.h>
#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{

__global__ void naive_1D(float *C, const float *A, const float *B, int hC, int wA, int wC)
{
    int totalSize = wC * hC;
    int thID = blockDim.x * blockIdx.x + threadIdx.x; //有多少线程就有多少个ID并行计算
    while(thID < totalSize)
    {
        int Cx = thID / wC; //数据在第几行
        int Cy = thID % wC; //数据在第几列
        float sum = 0;
        for (int i = 0; i < wA; i++)
            sum += A[Cx * wA + i] * B[i * wC + Cy];
        C[Cx * wC +Cy] = sum;
        thID += gridDim.x * blockDim.x; //进行下一个grid的计算
    }
}

PLAYGROUND_MATMUL_DEC(float32_t, 2, m, n, k, A, B, C)
{

    const int block_Dim = 256; 
    int grid_Dim = 4096 * 4096 / block_Dim;
    naive_1D<<<grid_Dim, block_Dim>>>(C, A, B, m, n, k);
    cudaDeviceSynchronize(); 
}
} 