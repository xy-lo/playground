#include <cuda_runtime.h>
#include <mma.h>

#include "playground/common.hpp"
#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{

#define BM 256
#define BN 128
#define BK 32 

#define K_STAGE 4

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define A_PAD 0
#define B_PAD 0

// 每个 Warp 计算的 C 块中的 MMA 块数量 (Warp Tile)
#define MMA_M_TILES 4 // 64 / MMA_M
#define MMA_N_TILES 8 // 64 / MMA_N

// 每个 Warp 负责的 C 块大小 (用于验证)
// #define WARP_ROWS (MMA_M * MMA_M_TILES) // 16 * 4 = 64
// #define WARP_COLS (MMA_N * MMA_N_TILES) // 8 * 8 = 64
// 注意：用户代码的 WARP_COLS 是 64，但 BN 是 128。
// 这是因为 blockDim.y=2, 每个 block 有 2 个 warp 负责 N 维度。
// 64 * 2 = 128。同理，blockDim.z=4, 64 * 4 = 256 (BM)。

//#define OFFSET(row, col, stride) ((row) * (stride) + (col))——放到common.hpp里了

__global__ void mma_4stagev1(const float16_t* __restrict__ A, const float16_t* __restrict__ B, float16_t* __restrict__ C, const int M, const int N, const int K)
{

    extern __shared__ float16_t smem[];
    float16_t* sa = smem;
    float16_t* sb = smem + K_STAGE * BM * (BK + A_PAD);
    float16_t* sc = smem; // C 缓冲区复用 sa 的空间

    // 寄存器: 使用新的宏定义
    uint32_t RC[MMA_M_TILES][MMA_N_TILES][2];
    uint32_t RA[2][MMA_M_TILES][4]; // 2 = 寄存器双缓冲
    uint32_t RB[2][MMA_N_TILES][2]; // 2 = 寄存器双缓冲

#pragma unroll
    for (size_t i = 0; i < MMA_M_TILES; i++) {
#pragma unroll
        for (size_t j = 0; j < MMA_N_TILES; j++) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    size_t bx =(blockIdx.y % 2 == 0) ? (blockDim.x - 1 - blockIdx.x) : blockIdx.x;
    size_t by = blockIdx.y;
    size_t tid = threadIdx.x + threadIdx.y * blockDim.x +
                 threadIdx.z * blockDim.x * blockDim.y;  //全局一维线程id：0-255
    size_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    size_t s_a_off = 0;
    size_t s_b_off = 0;

    // buff 0
    // a: global -> smem, 256 * 32 (BM * BK)
#pragma unroll
    for (int i = 0; i < 4; i++) {
        //目标：一个warp加载64*32个元素到smem，.num设置成*4，均匀分配给tid256个线程
        int row = ((tid >> 2) << 2) + i; //M = 256: >>2：将32线程分成0.1...7一共8组；<<2：每组内的线程负责4行
        int col = (tid & 3) << 3;  //K = 32: &3：将32线程分成0.1.2.3一共四组；<<3：每组内的线程负责8列

// <---------------- K 维度 (宽度 32) ------------------>
// ^  (M)
// |  +----------------+----------------+----------------+----------------+
// |0 | tid=0 (0..7)   | tid=1 (8..15)  | tid=2 (16..23) | tid=3 (24..31) |  <-- 组 0 (tid 0-3)
// |1 | tid=0          | tid=1          | tid=2          | tid=3          |      负责行 0-3 (通过 i 循环)
// |2 | tid=0          | tid=1          | tid=2          | tid=3          |
// |3 | tid=0          | tid=1          | tid=2          | tid=3          |
// |  +----------------+----------------+----------------+----------------+
// |4 | tid=4 (0..7)   | tid=5 (8..15)  | tid=6 (16..23) | tid=7 (24..31) |  <-- 组 1 (tid 4-7)
// |5 | tid=4          | tid=5          | tid=6          | tid=7          |      负责行 4-7 (通过 i 循环)
// |6 | tid=4          | tid=5          | tid=6          | tid=7          |
// |7 | tid=4          | tid=5          | tid=6          | tid=7          |
// |  +----------------+----------------+----------------+----------------+
// | ... (共 64 个 4 行块)
// |  +----------------+----------------+----------------+----------------+
// |252| tid=252 (0..7) | tid=253 (8..15)| tid=254 (16..23)| tid=255 (24..31)| <-- 组 63 (tid 252-255)
// |253| tid=252        | tid=253        | tid=254        | tid=255        |     负责行 252-255
// |254| tid=252        | tid=253        | tid=254        | tid=255        |     (通过 i 循环)
// |255| tid=252        | tid=253        | tid=254        | tid=255        |
// v  +----------------+----------------+----------------+----------------+
        int load_a_gmem_m = by * BM + row;
        int load_a_gmem_k = 0 * BK + col; // 0 * 32
        col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
        uint32_t smem_ptr =__cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, BK)); // 使用 OFFSET 和 BK
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K); 
        CP_ASYNC_CG(smem_ptr, &A[load_a_gmem_addr], 16);
    }

    // buff 0
    // b: global -> smem, 32 * 128 (BK * BN)
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int row = ((tid >> 4) << 1) + i;
        int col = (tid & 15) << 3;

        int load_b_gmem_k = 0 * BK + row; // 0 * 32
        int load_b_gmem_n = bx * BN + col;

        col = col ^ ((row & ((1 << 4) - 1)) << 3);
        uint32_t smem_ptr =
            __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, BN)); // 使用 OFFSET 和 BN
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N); // 使用 OFFSET
        CP_ASYNC_CG(smem_ptr, &B[load_b_gmem_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    s_a_off += BM * BK;
    s_b_off += BK * BN;

    // buff 1
    // a: global -> smem, 256 * 32
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = ((tid >> 2) << 2) + i;
        int col = (tid & 3) << 3;
        int load_a_gmem_m = by * BM + row;
        int load_a_gmem_k = 1 * BK + col; // 1 * 32
        col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
        uint32_t smem_ptr =
            __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, BK)); // 使用 OFFSET 和 BK
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K); // 使用 OFFSET
        CP_ASYNC_CG(smem_ptr, &A[load_a_gmem_addr], 16);
    }

    // buff 1
    // b: global -> smem, 32 * 128
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int row = ((tid >> 4) << 1) + i;
        int col = (tid & 15) << 3;
        int load_b_gmem_k = 1 * BK + row; // 1 * 32
        int load_b_gmem_n = bx * BN + col;
        col = col ^ ((row & ((1 << 4) - 1)) << 3);
        uint32_t smem_ptr =
            __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, BN)); // 使用 OFFSET 和 BN
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N); // 使用 OFFSET
        CP_ASYNC_CG(smem_ptr, &B[load_b_gmem_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    s_a_off += BM * BK;
    s_b_off += BK * BN;

    // buff 2
    // a: global -> smem, 256 * 32
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = ((tid >> 2) << 2) + i;
        int col = (tid & 3) << 3;
        int load_a_gmem_m = by * BM + row;
        int load_a_gmem_k = 2 * BK + col; // 2 * 32
        col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
        uint32_t smem_ptr =
            __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, BK)); // 使用 OFFSET 和 BK
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K); // 使用 OFFSET
        CP_ASYNC_CG(smem_ptr, &A[load_a_gmem_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP(); // A 和 B 分开 commit

    // buff 2
    // b: global -> smem, 32 * 128
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int row = ((tid >> 4) << 1) + i;
        int col = (tid & 15) << 3;
        int load_b_gmem_k = 2 * BK + row; // 2 * 32
        int load_b_gmem_n = bx * BN + col;
        col = col ^ ((row & ((1 << 4) - 1)) << 3);
        uint32_t smem_ptr =
            __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, BN)); // 使用 OFFSET 和 BN
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N); // 使用 OFFSET
        CP_ASYNC_CG(smem_ptr, &B[load_b_gmem_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    CP_ASYNC_WAIT_GROUP(2); // 等待 buff 0 和 1
    __syncthreads();

    size_t reg_store_idx = 0;
    size_t reg_load_idx = 1;

    s_a_off = 0;
    s_b_off = 0;

    // 加载 k=0 的数据到寄存器
#pragma unroll
    for (size_t i = 0; i < MMA_M_TILES; i++) { // 使用宏
        int row = tz * 64 + i * 16 + tx % 16;
        int col = 0 * MMA_K + (tx / 16) * 8; // 0 * 16
        col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
        uint32_t smem_base =
            __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, BK)); // 使用 OFFSET 和 BK
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    smem_base);
    }

#pragma unroll
    for (size_t i = 0; i < MMA_N_TILES; i++) { // 使用宏
        int row = 0 * MMA_K + tx % 16; // 0 * 16
        int col = ty * 64 + i * 8;
        col = col ^ ((row & ((1 << 4) - 1)) << 3);
        uint32_t smem_base =
            __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, BN)); // 使用 OFFSET 和 BN
        LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                      smem_base);
    }

    // --- 主流水线循环 ---
    for (int k = 3; k < K / BK; k++) {
        int smem_sel = (k + 1) % K_STAGE;
        int smem_sel_next = k % K_STAGE;
        s_a_off = smem_sel * BM * BK;
        s_b_off = smem_sel * BK * BN;

        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        // 从 smem (k-1) 加载数据到寄存器 (第 k 步的第 1-half)
#pragma unroll
        for (size_t i = 0; i < MMA_M_TILES; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = 1 * MMA_K + (tx / 16) * 8; // 1 * 16
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, BK)); // 使用 OFFSET 和 BK
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        smem_base);
        }

#pragma unroll
        for (size_t i = 0; i < MMA_N_TILES; i++) {
            int row = 1 * MMA_K + tx % 16; // 1 * 16
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, BN)); // 使用 OFFSET 和 BN
            LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                          smem_base);
        }
        
        // 计算 (k-2) (第 2-half)
#pragma unroll
        for (int m = 0; m < MMA_M_TILES; m++) {
#pragma unroll
            for (int n = 0; n < MMA_N_TILES; n++) {
                int n_ = (m & 1) ? (MMA_N_TILES - 1 - n) : n; // 使用宏
                HMMA16816(RC[m][n_][0], RC[m][n_][1], RA[reg_load_idx][m][0],
                          RA[reg_load_idx][m][1], RA[reg_load_idx][m][2],
                          RA[reg_load_idx][m][3], RB[reg_load_idx][n_][0],
                          RB[reg_load_idx][n_][1], RC[m][n_][0], RC[m][n_][1]);
            }
        }

        s_a_off = smem_sel_next * BM * BK;
        s_b_off = smem_sel_next * BK * BN;

        // 预取 k 块数据到 smem
        // a: global -> smem
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int row = ((tid >> 2) << 2) + i;
            int col = (tid & 3) << 3;
            int load_a_gmem_m = by * BM + row;
            int load_a_gmem_k = k * BK + col; // k * 32
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_ptr =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, BK)); // 使用 OFFSET 和 BK
            int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K); // 使用 OFFSET
            CP_ASYNC_CG(smem_ptr, &A[load_a_gmem_addr], 16);
        }

        // b: global -> smem
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int row = ((tid >> 4) << 1) + i;
            int col = (tid & 15) << 3;
            int load_b_gmem_k = k * BK + row; // k * 32
            int load_b_gmem_n = bx * BN + col;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_ptr =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, BN)); // 使用 OFFSET 和 BN
            int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N); // 使用 OFFSET
            CP_ASYNC_CG(smem_ptr, &B[load_b_gmem_addr], 16);
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(2); // 等待 (k-2) 块数据加载完成

        __syncthreads();

        // ---- k 步的第 2-half ----
        
        int smem_sel_k_minus_2 = (k - 2) % K_STAGE;
        s_a_off = smem_sel_k_minus_2 * BM * BK;
        s_b_off = smem_sel_k_minus_2 * BK * BN;

        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        // 从 smem (k-2) 加载数据到寄存器 (第 2-half)
#pragma unroll
        for (size_t i = 0; i < MMA_M_TILES; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = 0 * MMA_K + (tx / 16) * 8; // 0 * 16
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, BK)); // 使用 OFFSET 和 BK
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        smem_base);
        }

#pragma unroll
        for (size_t i = 0; i < MMA_N_TILES; i++) {
            int row = 0 * MMA_K + tx % 16; // 0 * 16
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, BN)); // 使用 OFFSET 和 BN
            LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                          smem_base);
        }

        // 计算 (k-2) (第 1-half)
#pragma unroll
        for (int m = 0; m < MMA_M_TILES; m++) {
#pragma unroll
            for (int n = 0; n < MMA_N_TILES; n++) {
                int n_ = (m & 1) ? (MMA_N_TILES - 1 - n) : n; // 使用宏
                HMMA16816(RC[m][n_][0], RC[m][n_][1], RA[reg_load_idx][m][0],
                          RA[reg_load_idx][m][1], RA[reg_load_idx][m][2],
                          RA[reg_load_idx][m][3], RB[reg_load_idx][n_][0],
                          RB[reg_load_idx][n_][1], RC[m][n_][0], RC[m][n_][1]);
            }
        }
    } // end main loop

    // --- Epilogue (处理流水线中剩余的 k-2, k-1 块) ---

    // 处理 k = K/BK - 3 (已在循环末尾计算)
    
    // 处理 k = K/BK - 2
    int smem_sel = (K / BK - 3) % K_STAGE;
#pragma unroll
    for (int k_half = 1; k_half >= 0; k_half--) {
        s_a_off = smem_sel * BM * BK;
        s_b_off = smem_sel * BK * BN;
        reg_load_idx ^= 1;
        reg_store_idx ^= 1;
#pragma unroll
        for (size_t i = 0; i < MMA_M_TILES; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = k_half * MMA_K + (tx / 16) * 8; // k_half * 16
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, BK)); // 使用 OFFSET 和 BK
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        smem_base);
        }

#pragma unroll
        for (size_t i = 0; i < MMA_N_TILES; i++) {
            int row = k_half * MMA_K + tx % 16; // k_half * 16
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, BN)); // 使用 OFFSET 和 BN
            LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                          smem_base);
        }

#pragma unroll
        for (int m = 0; m < MMA_M_TILES; m++) {
#pragma unroll
            for (int n = 0; n < MMA_N_TILES; n++) {
                int n_ = (m & 1) ? (MMA_N_TILES - 1 - n) : n; // 使用宏
                HMMA16816(RC[m][n_][0], RC[m][n_][1], RA[reg_load_idx][m][0],
                          RA[reg_load_idx][m][1], RA[reg_load_idx][m][2],
                          RA[reg_load_idx][m][3], RB[reg_load_idx][n_][0],
                          RB[reg_load_idx][n_][1], RC[m][n_][0], RC[m][n_][1]);
            }
        }
        if (k_half == 1) {
            smem_sel = (smem_sel + 1) % K_STAGE;
            CP_ASYNC_WAIT_GROUP(1); // 等待 K/BK - 2 的数据
            __syncthreads();
        }
    }

    // 处理 k = K/BK - 1
    smem_sel = (K / BK - 2) % K_STAGE; // (K/BK - 3 + 1) % 4
#pragma unroll
    for (int k_half = 1; k_half >= 0; k_half--) {
        s_a_off = smem_sel * BM * BK;
        s_b_off = smem_sel * BK * BN;
        reg_load_idx ^= 1;
        reg_store_idx ^= 1;
#pragma unroll
        for (size_t i = 0; i < MMA_M_TILES; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = k_half * MMA_K + (tx / 16) * 8; // k_half * 16
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, BK)); // 使用 OFFSET 和 BK
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        smem_base);
        }

#pragma unroll
        for (size_t i = 0; i < MMA_N_TILES; i++) {
            int row = k_half * MMA_K + tx % 16; // k_half * 16
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, BN)); // 使用 OFFSET 和 BN
            LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                          smem_base);
        }

#pragma unroll
        for (int m = 0; m < MMA_M_TILES; m++) {
#pragma unroll
            for (int n = 0; n < MMA_N_TILES; n++) {
                int n_ = (m & 1) ? (MMA_N_TILES - 1 - n) : n; // 使用宏
                HMMA16816(RC[m][n_][0], RC[m][n_][1], RA[reg_load_idx][m][0],
                          RA[reg_load_idx][m][1], RA[reg_load_idx][m][2],
                          RA[reg_load_idx][m][3], RB[reg_load_idx][n_][0],
                          RB[reg_load_idx][n_][1], RC[m][n_][0], RC[m][n_][1]);
            }
        }
        if (k_half == 1) {
            smem_sel = (smem_sel + 1) % K_STAGE;
            CP_ASYNC_WAIT_GROUP(0); // 等待 K/BK - 1 的数据
            __syncthreads();
        }
    }

    // 处理 k = K/BK (最后一块)
    smem_sel = (K / BK - 1) % K_STAGE; // (K/BK - 2 + 1) % 4
#pragma unroll
    for (int k_half = 1; k_half >= 0; k_half--) {
        s_a_off = smem_sel * BM * BK;
        s_b_off = smem_sel * BK * BN;
        reg_load_idx ^= 1;
        reg_store_idx ^= 1;
#pragma unroll
        for (size_t i = 0; i < MMA_M_TILES; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = k_half * MMA_K + (tx / 16) * 8; // k_half * 16
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, BK)); // 使用 OFFSET 和 BK
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        smem_base);
        }

#pragma unroll
        for (size_t i = 0; i < MMA_N_TILES; i++) {
            int row = k_half * MMA_K + tx % 16; // k_half * 16
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, BN)); // 使用 OFFSET 和 BN
            LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                          smem_base);
        }

#pragma unroll
        for (int m = 0; m < MMA_M_TILES; m++) {
#pragma unroll
            for (int n = 0; n < MMA_N_TILES; n++) {
                int n_ = (m & 1) ? (MMA_N_TILES - 1 - n) : n; // 使用宏
                HMMA16816(RC[m][n_][0], RC[m][n_][1], RA[reg_load_idx][m][0],
                          RA[reg_load_idx][m][1], RA[reg_load_idx][m][2],
                          RA[reg_load_idx][m][3], RB[reg_load_idx][n_][0],
                          RB[reg_load_idx][n_][1], RC[m][n_][0], RC[m][n_][1]);
            }
        }
    }

    __syncthreads();

    // --- 结果写回 ---
    // Reg -> Smem
#pragma unroll
    for (size_t i = 0; i < MMA_M_TILES; i++) {
#pragma unroll
        for (size_t j = 0; j < MMA_N_TILES; j++) {
            int row = tz * 64 + i * 16 + tx / 4;
            int col = ty * 64 + j * 8 + (tx % 4) * 2;
            int col1 = col ^ ((row & ((1 << 4) - 1)) << 3);
            int col2 = col ^ (((row + 8) & ((1 << 4) - 1)) << 3);

            (reinterpret_cast<uint32_t*>(sc + OFFSET(row, col1, BN)))[0] = // 使用 OFFSET 和 BN
                RC[i][j][0];
            (reinterpret_cast<uint32_t*>(sc + OFFSET(row + 8, col2, BN)))[0] = // 使用 OFFSET 和 BN
                RC[i][j][1];
        }
    }
    __syncthreads();

    // Smem -> Gmem
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int row = i * 16 + tid / 16;
        int col1 = (tid % 16) * 8;
        int col2 = col1 ^ ((row & ((1 << 4) - 1)) << 3);
        INT4(C + OFFSET(by * BM + row, bx * BN + col1, N)) = // 使用 OFFSET
            INT4(sc + OFFSET(row, col2, BN)); // 使用 OFFSET 和 BN
    }
}

PLAYGROUND_MATMUL_DEC(float16_t, 7, M, N, K, A, B, C)
{

    dim3 blockDim(32, 2, 4); // 256 threads
    dim3 gridDim(div_ceil(N, BN), div_ceil(M, BM));

    cudaFuncSetAttribute(mma_4stagev1, cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);
    size_t sharedMemSize = std::max(
        (size_t)K_STAGE * (BM * (BK + A_PAD) + BK * (BN + B_PAD)) * sizeof(float16_t),
        (size_t)BM * BN * sizeof(float16_t)); 
    
    mma_4stagev1<<<gridDim, blockDim, sharedMemSize>>>(A, B, C, M, N, K);
        
}

} // namespace playground