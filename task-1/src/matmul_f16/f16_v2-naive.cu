#include <cuda_runtime.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{

    //每个warp实现16*16的
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 64

using namespace nvcuda;

__global__ void wmmaNaiveKernel(const half *__restrict__ A,
                                const half *__restrict__ B,
                                half *__restrict__ C, size_t M, size_t N, size_t K)
{
    const size_t K_tiles = (K + WMMA_K - 1) / WMMA_K; //每次处理K维度的个数
    const size_t warp_row = blockIdx.y * WMMA_M;  //行数=当前块的行索引*每个块的行数
    const size_t warp_col = blockIdx.x * WMMA_N;  //列数=当前块的列索引*每个块的列数
    if (warp_row >= M || warp_col >= N)
    {
        return;
    }
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0F);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i)
    {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, B + i * WMMA_K * N + warp_col, N);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }
    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major); //起始位置，写入的数据，宽度，存储格式行主序
}

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    dim3 block(WARP_SIZE);
    dim3 grid((n + WMMA_N - 1) / WMMA_N, (m + WMMA_M - 1) / WMMA_M);

    wmmaNaiveKernel<<<grid, block>>>(reinterpret_cast<const half *>(A),
                                     reinterpret_cast<const half *>(B),
                                     reinterpret_cast<half *>(C), m, n, k);
}

}  // namespace playground