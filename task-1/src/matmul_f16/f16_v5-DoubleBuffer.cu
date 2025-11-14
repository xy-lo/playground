#include <cuda_runtime.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

#define BM 128
#define BN 256
#define BK 32

#define APAD 8
#define BPAD 8

#define OFFSET(row, col, stride) ((row) * (stride) + (col))

using namespace nvcuda;

__global__ void DoubleBuffer (const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M, size_t N, size_t K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;  // warp id: tid / 32

    extern __shared__ half smem[];
    half *s_a = smem;                         //A的共享内存起始地址
    half *s_b = smem + 2 * BM * (BK + APAD);  //B的共享内存起始地址
    const int s_a_db_offset = BM * (BK + APAD);  //缓冲偏移
    const int s_b_db_offset = BK * (BN + BPAD);

    // WMMA fragments寄存器内存
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    //Thread 在 Block 内的相对地址
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

    //计算C片段寄存器对应的行列
    int comp_c_frag_m = wid & 1;   // wid % 2
    int comp_c_frag_n = wid >> 1;  // wid / 2
    int num_tiles = (K + BK - 1) / BK;

    //buffer 0异步加载
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

    // 主循环
    for (int tile = 0; tile < num_tiles; tile++) {
        int smem_sel = tile & 1;  //当前缓冲区
        int smem_sel_next = (tile + 1) & 1; //下一个缓冲区

        asm ("cp.async.wait_group 0;\n" ::); // 等待数据加载
        __syncthreads();

        //加载下一个tile到下一个缓冲区
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

        //计算当前缓冲区的基地址
        int base_a = smem_sel * s_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD);
        int base_b = smem_sel * s_b_db_offset + comp_c_frag_n * 64;

        //共享内存加载到寄存器
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(frag_a[0][i], 
                &s_a[base_a + i * 16 * (BK + APAD)], BK + APAD);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::load_matrix_sync(frag_b[0][j], 
                &s_b[base_b + j * 16], BN + BPAD);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(frag_a[1][i], 
                &s_a[base_a + i * 16 * (BK + APAD) + 16], BK + APAD);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::load_matrix_sync(frag_b[1][j], 
                &s_b[base_b + 16 * (BN + BPAD) + j * 16], BN + BPAD);
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }
    }

    //存储结果到全局内存
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    
    if (store_c_gmem_m < M && store_c_gmem_n < N) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int row = store_c_gmem_m + i * 16;
                int col = store_c_gmem_n + j * 16;
                if (row < M && col < N) {
                    wmma::store_matrix_sync(&C[OFFSET(row, col, N)],
                                           frag_c[i][j], N, wmma::mem_row_major);
                }
            }
        }
    }
}

PLAYGROUND_MATMUL_DEC(float16_t, 5, m, n, k, A, B, C)
{
    dim3 blockDim(256);  // 8 warps
    int BX = (n + BN - 1) / BN;
    int BY = (m + BM - 1) / BM;
    dim3 gridDim(BX, BY);

    //设置最大96KB动态共享内存
    cudaFuncSetAttribute(DoubleBuffer, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    // 计算动态共享内存大小
    unsigned int dsmem = 2 * (BM * (BK + APAD) + BK * (BN + BPAD)) * sizeof(half);

    DoubleBuffer<<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<const half *>(A),
        reinterpret_cast<const half *>(B),
        reinterpret_cast<half *>(C),
        m, n, k);
}

}  // namespace playground