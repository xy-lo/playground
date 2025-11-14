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

// Block and tile sizes
#define BM 128
#define BN 256
#define BK 32

// Padding to avoid bank conflicts
#define APAD 8
#define BPAD 8

// Offset macro for 2D indexing
#define OFFSET(row, col, stride) ((row) * (stride) + (col))

using namespace nvcuda;

__global__ void wmmaDoubleBufferKernele(
    const half *__restrict__ A,
    const half *__restrict__ B,
    half *__restrict__ C,
    size_t M, size_t N, size_t K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;  // warp id: tid / 32

    // Shared memory with double buffering
    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + 2 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    // WMMA fragments
    // frag_a[2]: double buffer, [4]: 4 tiles in M dimension
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    // Initialize accumulator fragments
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    // Calculate loading indices for shared memory
    // Each thread loads 2 rows of A (2x8 elements) and 4 rows of B (4x8 elements)
    int load_a_smem_m = (tid >> 2) << 1;  // (tid / 4) * 2
    int load_a_smem_k = (tid & 3) << 3;   // (tid % 4) * 8
    int load_b_smem_k = (tid >> 5) << 2;  // (tid / 32) * 4
    int load_b_smem_n = (tid & 31) << 3;  // (tid % 32) * 8

    // Convert generic addresses to shared memory addresses
    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    // Calculate shared memory addresses for async copy
    int load_a_smem_addr_0 = s_a_base_addr + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    int load_b_smem_addr_0 = s_b_base_addr + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 + (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 = load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 = load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    // Calculate global memory addresses
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    // Calculate which fragment this warp computes
    int comp_c_frag_m = wid & 1;   // wid % 2
    int comp_c_frag_n = wid >> 1;  // wid / 2

    // Load first tile (for double buffering initialization)
    {
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0), "l"(&A[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1), "l"(&A[load_a_gmem_addr + K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0), "l"(&B[load_b_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1), "l"(&B[load_b_gmem_addr + N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2), "l"(&B[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3), "l"(&B[load_b_gmem_addr + 3 * N]));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    // Main loop with double buffering
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk & 1) ^ 1;           // Current buffer
        int smem_sel_next = ((bk - 1) & 1) ^ 1; // Next buffer

        // Update global memory addresses
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // Async copy next tile to next buffer
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_0 + smem_sel_next * s_a_db_offset * (int)sizeof(half)),
              "l"(&A[load_a_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_a_smem_addr_1 + smem_sel_next * s_a_db_offset * (int)sizeof(half)),
              "l"(&A[load_a_gmem_addr + K]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_0 + smem_sel_next * s_b_db_offset * (int)sizeof(half)),
              "l"(&B[load_b_gmem_addr]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_1 + smem_sel_next * s_b_db_offset * (int)sizeof(half)),
              "l"(&B[load_b_gmem_addr + N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_2 + smem_sel_next * s_b_db_offset * (int)sizeof(half)),
              "l"(&B[load_b_gmem_addr + 2 * N]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "r"(load_b_smem_addr_3 + smem_sel_next * s_b_db_offset * (int)sizeof(half)),
              "l"(&B[load_b_gmem_addr + 3 * N]));

        // Load current tile from shared memory to fragments
        // Load A fragments (4 tiles, 2 K-slices)
        wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

        // Load B fragments (4 tiles, 2 K-slices)
        wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

        // Perform matrix multiplication
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        // Wait for async copy to complete
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    // Process last tile
    int smem_sel = (((K + BK - 1) / BK) & 1) ^ 1;

    wmma::load_matrix_sync(frag_a[0][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16], BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 48], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 16], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 32], BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64 + 48], BN + BPAD);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }
    }

    // Store results to global memory
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&C[store_c_gmem_addr + i * 16 * N + j * 16],
                                   frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}

PLAYGROUND_MATMUL_DEC(float16_t, 50, m, n, k, A, B, C)
{
    dim3 blockDim(256);  // 8 warps per block
    int BX = (n + BN - 1) / BN;
    int BY = (m + BM - 1) / BM;
    dim3 gridDim(BX, BY);

    // Set maximum dynamic shared memory size
    cudaFuncSetAttribute(wmmaDoubleBufferKernele,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        98304);

    // Calculate dynamic shared memory size
    unsigned int dsmem = 2 * (BM * (BK + APAD) + BK * (BN + BPAD)) * sizeof(half);

    wmmaDoubleBufferKernele<<<gridDim, blockDim, dsmem>>>(
        reinterpret_cast<const half *>(A),
        reinterpret_cast<const half *>(B),
        reinterpret_cast<half *>(C),
        m, n, k);
}

}  // namespace playground