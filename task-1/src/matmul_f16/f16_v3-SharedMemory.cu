#include <cuda_runtime.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{

using namespace nvcuda;
// A[i][j] 的行主序偏移 (row * stride + col)
// K 或 N 是矩阵的行跨度（Stride）
#define OFFSET(row, col, stride) ((row) * (stride) + (col))

// FLOAT4 宏用于 128 位加载/存储，优化内存访问
// 确保 nvcuda::half 可以安全地转换为 int4
#define FLOAT4(ptr) (*(int4 *)&(ptr))


__global__ void SharedMemoryv1(
    const half *__restrict__ a,
    const half *__restrict__ b,
    half *__restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid / 32;  //warp id

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    //Thread 在 Block 内的相对地址
    int load_a_smem_m = (tid >> 2) << 1; //256个线程分成256/4=64组，每组4个线程；<<1相当于每组分成两行：0.1.2...63->0.2.4...126
    int load_a_smem_k = (tid &  3) << 3; //每组0.1.2.3共4个线程
    int load_b_smem_k = (tid >> 5) << 2; //256个线程分成256/32=8组，每组32个线程；<<2相当于每组分成4行：0.1.2...63->0.4.8...252
    int load_b_smem_n = (tid & 31) << 3; //每组0.1.2.3...31共32个线程
//s_a[128][32]
// <---------------- K (宽度 32) ---------------->
// ^  (m)
// |  +----------------+----------------+----------------+----------------+
// |0 | tid=0 (0..7)   | tid=1 (8..15)  | tid=2 (16..23) | tid=3 (24..31) |  <-- 组 0 (tid 0-3)
// |1 | tid=0          | tid=1          | tid=2          | tid=3          |      负责行 0, 1
// |  +----------------+----------------+----------------+----------------+
// |2 | tid=4 (0..7)   | tid=5 (8..15)  | tid=6 (16..23) | tid=7 (24..31) |  <-- 组 1 (tid 4-7)
// |3 | tid=4          | tid=5          | tid=6          | tid=7          |      负责行 2, 3
// |  +----------------+----------------+----------------+----------------+
// | ...
// |  +----------------+----------------+----------------+----------------+
// |126| tid=252 (0..7) | tid=253 (8..15)| tid=254 (16..23)| tid=255 (24..31)| <-- 组 63 (tid 252-255)
// |127| tid=252        | tid=253        | tid=254        | tid=255        |     负责行 126, 127
// v  +----------------+----------------+----------------+----------------+

// s_b[32][256]
// <------------------------------ N (宽度 256) ------------------------------>
// ^  (k)
// |  +---------+---------+-----+---------+
// |0 | tid=0   | tid=1   | ... | tid=31  |
// |1 | (0..7)  | (8..15) | ... |(248..255|  <-- 组 0 (tid 0-31)
// |2 |         |         |     |         |      负责行 0, 1, 2, 3
// |3 |         |         |     |         |
// |  +---------+---------+-----+---------+
// |4 | tid=32  | tid=33  | ... | tid=63  |
// |5 | (0..7)  | (8..15) | ... |(248..255|  <-- 组 1 (tid 32-63)
// |6 |         |         |     |         |      负责行 4, 5, 6, 7
// |7 |         |         |     |         |
// |  +---------+---------+-----+---------+
// | ...
// |  +---------+---------+-----+---------+
// |28| tid=224 | tid=225 | ... | tid=255 |
// |29| (0..7)  | (8..15) | ... |(248..255|  <-- 组 7 (tid 224-255)
// |30|         |         |     |         |      负责行 28, 29, 30, 31
// |31|         |         |     |         |
// v  +---------+---------+-----+---------+

    //计算全局地址
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    //二维转换成一维地址
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);
    // 映射到2D坐标
    int comp_c_frag_m = wid &  1; //0.2.4.6=0；1.3.5.7=1
    int comp_c_frag_n = wid >> 1; //0.1=0；2.3=1；4.5=2；6.7=3

    //全局内存放到共享内存
    for (int bk = 0; bk < K / BK; bk++) {
        FLOAT4(s_a[load_a_smem_m    ][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr        ]);
        FLOAT4(s_a[load_a_smem_m + 1][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr +     K]);
        FLOAT4(s_b[load_b_smem_k    ][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr        ]);
        FLOAT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr +     N]);
        FLOAT4(s_b[load_b_smem_k + 2][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 2 * N]);
        FLOAT4(s_b[load_b_smem_k + 3][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 3 * N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64     ][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64     ][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[ 0][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[ 0][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[ 0][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[ 0][comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}


PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    dim3 block(256);
    dim3 grid(n / 256, m / 128, 1);

    SharedMemoryv1<<<grid, block>>>(reinterpret_cast<const half *>(A),
                                     reinterpret_cast<const half *>(B),
                                     reinterpret_cast<half *>(C), m, n, k);
}

}  // namespace playground