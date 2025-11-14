#include <cuda_runtime.h>
#include <mma.h> // 使用 mma.h 替换 wmma.h

#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include "playground/common.hpp"

namespace playground
{

// WMMA 相关的宏被替换或移除
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32

#define BM 128
#define BN 256
#define BK 32 // 注意: BK 是 MMA_K 的两倍

#define APAD 8
#define BPAD 8

// 新增：定义每个 Warp 的 MMA 块数量
#define MMA_M_TILES 4 // 保持 Warp Tile M 维度 = 4 * 16 = 64
#define MMA_N_TILES 8 // Warp Tile N 维度 = 8 * 8 = 64 (wmma是 4 * 16 = 64)

//#define OFFSET(row, col, stride) ((row) * (stride) + (col))——放到common.hpp里了

// using namespace nvcuda; // mma.h 不需要这个

__global__ void DoubleBuffer_MMA_Version (const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M, size_t N, size_t K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x; // 0-255
    
    // Warp 和 Lane 索引（用于 LDMATRIX 和 Reg->Smem）
    int wid = tid >> 5;     // warp id: 0-7
    int lane_id = tid & 31; // lane id: 0-31
    
    // Warp 在 2x4 网格中的坐标
    int warp_m_idx = wid & 1;   // 0 or 1
    int warp_n_idx = wid >> 1;  // 0, 1, 2, or 3

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK + APAD);
    half *sc = (half*)smem; // C 缓冲区复用 s_a 的空间
    
    const int s_a_db_offset = BM * (BK + APAD);
    const int s_b_db_offset = BK * (BN + BPAD);

    // --- 寄存器定义 (MMA 版本) ---
    // [2] = 存储 BK=32 的两个 k_half
    // [2] = 寄存器双缓冲 (此代码不使用)
    uint32_t RA[2][MMA_M_TILES][4];
    uint32_t RB[2][MMA_N_TILES][2];
    uint32_t RC[MMA_M_TILES][MMA_N_TILES][2];

    // 初始化 RC 累加器
    #pragma unroll
    for (int i = 0; i < MMA_M_TILES; i++) {
        #pragma unroll
        for (int j = 0; j < MMA_N_TILES; j++) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    // --- Gmem -> Smem (cp.async) 逻辑 (与原版相同) ---
    // Thread 在 Block 内的相对地址
    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    //计算全局内存加载地址
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    //边界检查
    bool load_a_valid = load_a_gmem_m < M && load_a_gmem_m + 1 < M;
    bool load_b_valid = load_b_gmem_n < N;

    //计算共享内存基地址
    unsigned int s_a_base_addr = __cvta_generic_to_shared(s_a);
    unsigned int s_b_base_addr = __cvta_generic_to_shared(s_b);
    unsigned int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    unsigned int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    unsigned int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    unsigned int load_b_smem_addr_1 = load_b_smem_addr_0 + (BN + BPAD) * sizeof(half);
    unsigned int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    unsigned int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    int num_tiles = (K + BK - 1) / BK;

    //buffer 0异步加载 (与原版相同)
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_b_gmem_k = load_b_smem_k;
        const half* load_a_ptr = A + OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        const half* load_b_ptr = B + OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        if (load_a_valid && load_a_gmem_k < K) {
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                 : "r"(load_a_smem_addr_0), "l"(load_a_ptr));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                 : "r"(load_a_smem_addr_1), "l"(load_a_ptr + K));
        }
        
        if (load_b_valid && load_b_gmem_k < K) {
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                 : "r"(load_b_smem_addr_0), "l"(load_b_ptr));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                 : "r"(load_b_smem_addr_1), "l"(load_b_ptr + N));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                 : "r"(load_b_smem_addr_2), "l"(load_b_ptr + 2 * N));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                 : "r"(load_b_smem_addr_3), "l"(load_b_ptr + 3 * N));
        }

        asm ("cp.async.commit_group;\n" ::);
    }

    // --- 主循环 (K_STAGE=2 逻辑不变) ---
    for (int tile = 0; tile < num_tiles; tile++) {
        int smem_sel = tile & 1;  //当前缓冲区
        int smem_sel_next = (tile + 1) & 1; //下一个缓冲区

        asm ("cp.async.wait_group 0;\n" ::); // 等待数据加载
        __syncthreads();

        //加载下一个tile到下一个缓冲区 (与原版相同)
        if (tile + 1 < num_tiles) {
            int load_a_gmem_k = (tile + 1) * BK + load_a_smem_k;
            int load_b_gmem_k = (tile + 1) * BK + load_b_smem_k;
            const half* load_a_ptr = A + OFFSET(load_a_gmem_m, load_a_gmem_k, K);
            const half* load_b_ptr = B + OFFSET(load_b_gmem_k, load_b_gmem_n, N);

            unsigned int dst_a_0 = load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * sizeof(half);
            unsigned int dst_a_1 = load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * sizeof(half);
            unsigned int dst_b_0 = load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * sizeof(half);
            unsigned int dst_b_1 = load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * sizeof(half);
            unsigned int dst_b_2 = load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * sizeof(half);
            unsigned int dst_b_3 = load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * sizeof(half);

            if (load_a_valid && load_a_gmem_k < K) {
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                     : "r"(dst_a_0), "l"(load_a_ptr));
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                     : "r"(dst_a_1), "l"(load_a_ptr + K));
            }
            if (load_b_valid && load_b_gmem_k < K) {
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                     : "r"(dst_b_0), "l"(load_b_ptr));
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                     : "r"(dst_b_1), "l"(load_b_ptr + N));
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                     : "r"(dst_b_2), "l"(load_b_ptr + 2 * N));
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
                     : "r"(dst_b_3), "l"(load_b_ptr + 3 * N));
            }
            asm ("cp.async.commit_group;\n" ::);
        }

        // --- Smem -> Reg (LDMATRIX) ---
        // k_half = 0 (加载 BK 的前 16)
        #pragma unroll
        for (size_t i = 0; i < MMA_M_TILES; i++) {
            int row = warp_m_idx * 64 + i * 16 + lane_id % 16;
            int col = 0 * MMA_K + (lane_id / 16) * 8; // k_half = 0
            uint32_t smem_base = __cvta_generic_to_shared(
                s_a + smem_sel * s_a_db_offset + OFFSET(row, col, BK + APAD));
            LDMATRIX_X4(RA[0][i][0], RA[0][i][1], RA[0][i][2], RA[0][i][3], smem_base);
        }
        #pragma unroll
        for (size_t i = 0; i < MMA_N_TILES; i++) {
            int row = 0 * MMA_K + lane_id % 16; // k_half = 0
            int col = warp_n_idx * 64 + i * 8;
            uint32_t smem_base = __cvta_generic_to_shared(
                s_b + smem_sel * s_b_db_offset + OFFSET(row, col, BN + BPAD));
            LDMATRIX_X2_T(RB[0][i][0], RB[0][i][1], smem_base);
        }

        // k_half = 1 (加载 BK 的后 16)
        #pragma unroll
        for (size_t i = 0; i < MMA_M_TILES; i++) {
            int row = warp_m_idx * 64 + i * 16 + lane_id % 16;
            int col = 1 * MMA_K + (lane_id / 16) * 8; // k_half = 1
            uint32_t smem_base = __cvta_generic_to_shared(
                s_a + smem_sel * s_a_db_offset + OFFSET(row, col, BK + APAD));
            LDMATRIX_X4(RA[1][i][0], RA[1][i][1], RA[1][i][2], RA[1][i][3], smem_base);
        }
        #pragma unroll
        for (size_t i = 0; i < MMA_N_TILES; i++) {
            int row = 1 * MMA_K + lane_id % 16; // k_half = 1
            int col = warp_n_idx * 64 + i * 8;
            uint32_t smem_base = __cvta_generic_to_shared(
                s_b + smem_sel * s_b_db_offset + OFFSET(row, col, BN + BPAD));
            LDMATRIX_X2_T(RB[1][i][0], RB[1][i][1], smem_base);
        }


        // --- Compute (HMMA) ---
        // k_half = 0
        #pragma unroll
        for (int m = 0; m < MMA_M_TILES; m++) {
            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; n++) {
                HMMA16816(RC[m][n][0], RC[m][n][1], 
                          RA[0][m][0], RA[0][m][1], RA[0][m][2], RA[0][m][3], 
                          RB[0][n][0], RB[0][n][1], 
                          RC[m][n][0], RC[m][n][1]);
            }
        }
        // k_half = 1
        #pragma unroll
        for (int m = 0; m < MMA_M_TILES; m++) {
            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; n++) {
                HMMA16816(RC[m][n][0], RC[m][n][1], 
                          RA[1][m][0], RA[1][m][1], RA[1][m][2], RA[1][m][3], 
                          RB[1][n][0], RB[1][n][1], 
                          RC[m][n][0], RC[m][n][1]);
            }
        }
    } // end main loop

    // --- 结果写回 (Reg -> Smem -> Gmem) ---
    __syncthreads(); // 确保所有K维度的计算都已完成

    // 1. Reg -> Smem (参考 mma_4stagev1)
    #pragma unroll
    for (size_t i = 0; i < MMA_M_TILES; i++) { // 4
        #pragma unroll
        for (size_t j = 0; j < MMA_N_TILES; j++) { // 8
            // 找到此线程在 16x8 C 块 (fragment) 中的位置
            int row_in_frag = lane_id / 4; // 0-7
            int col_in_frag = (lane_id % 4) * 2; // 0, 2, 4, 6
            
            // 找到此 fragment 在 128x256 Smem 块中的基地址
            int row_base = warp_m_idx * 64 + i * 16;
            int col_base = warp_n_idx * 64 + j * 8;
            
            // 计算最终的 smem 写入地址
            int row = row_base + row_in_frag;
            int col = col_base + col_in_frag;
            
            // 注意: mma 16x8x16 的 RC fragment 包含 8x4=32 个 f16
            // 它们存储在 [2] 个 uint32_t (RC[...][0], RC[...][1])
            // RC[...][0] 存储 fragment 的 0-3 行
            // RC[...][1] 存储 fragment 的 8-11 行
            // (这是对 mma_4stagev1 写回的简化和适配)
            
            (reinterpret_cast<uint32_t*>(sc + OFFSET(row, col, BN)))[0] = RC[i][j][0];
            (reinterpret_cast<uint32_t*>(sc + OFFSET(row + 8, col, BN)))[0] = RC[i][j][1]; // 8x4 (f16) = 4x (u32)
        }
    }
    __syncthreads(); // 确保所有 Reg -> Smem 写回完成

    // 2. Smem -> Gmem (参考 mma_4stagev1, 适配 BM=128, BN=256)
    // 256 个线程 写入 128 * 256 = 32768 个元素 (f16)
    // 每个线程写入 32768 / 256 = 128 个 f16 = 64 个 f32 = 16 个 INT4
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        // 计算此线程的 smem 和 gmem 读写基地址
        // 256 线程 -> 16x16 线程格
        int row = (tid / 16) + (i * 8); // (0..15) + (0..15 * 8) -> 跨越 128 行
        int col = (tid % 16) * 16;      // (0..15) * 16 = 0..240 -> 跨越 256 列

        // 每次写入 16 * f16 (8 * f32 = 2 * INT4)
        INT4(C + OFFSET(by * BM + row, bx * BN + col, N)) = 
            INT4(sc + OFFSET(row, col, BN));
        INT4(C + OFFSET(by * BM + row, bx * BN + col + 8, N)) = 
            INT4(sc + OFFSET(row, col + 8, BN));
    }
}

PLAYGROUND_MATMUL_DEC(float16_t, 9, m, n, k, A, B, C)
{
    dim3 blockDim(256);  // 8 warps
    int BX = (n + BN - 1) / BN;
    int BY = (m + BM - 1) / BM;
    dim3 gridDim(BX, BY);

    //设置最大动态共享内存
    size_t dsmem_ab = 2 * (BM * (BK + APAD) + BK * (BN + BPAD)) * sizeof(half);
    // 增加 C 写回所需的 smem (BM * BN)
    size_t dsmem_c = BM * BN * sizeof(half);
    unsigned int dsmem = (unsigned int)std::max(dsmem_ab, dsmem_c);
    
    // 确保 dsmem 不超过 96KB
    if (dsmem > 98304) dsmem = 98304; 

    cudaFuncSetAttribute(DoubleBuffer_MMA_Version, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    DoubleBuffer_MMA_Version<<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<const half *>(A),
        reinterpret_cast<const half *>(B),
        reinterpret_cast<half *>(C),
        m, n, k);
}

}  // namespace playground