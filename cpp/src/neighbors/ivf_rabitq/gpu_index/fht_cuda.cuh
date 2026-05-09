/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Fast Hadamard Transform — CUDA kernels for power-of-2 dimensions.
 *
 * Implementation using the standard butterfly algorithm:
 *   1. Thread-level butterfly on kNElts elements per thread (in registers)
 *   2. Warp-level butterfly via __shfl_xor_sync
 *   3. Cross-warp butterfly via shared memory exchange
 *   4. (If kNChunks > 1) chunk-level butterfly for large dimensions
 *
 * Also provides a fused 4-round rotation kernel (sign flip + FHT × 4) that
 * keeps data in registers between rounds, minimizing global memory traffic.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <raft/util/cuda_rt_essentials.hpp>

namespace fht {

// ============================================================================
// Compile-time helpers
// ============================================================================

constexpr __host__ __device__ int clog2(int val) { return val > 1 ? 1 + clog2(val >> 1) : 0; }

/// Generic compile-time dispatch: converts a runtime log_N (3..15) into a
/// template parameter by recursive if-constexpr.
/// Usage: dispatch_log_n(log_N, [&](auto K) { launch<K.value>(...); });
template<int kCur = 3, int kMax = 15, typename Func>
inline void dispatch_log_n(int log_n, Func&& func) {
    if (log_n == kCur) {
        func(std::integral_constant<int, kCur>{});
    } else if constexpr (kCur < kMax) {
        dispatch_log_n<kCur + 1, kMax>(log_n, std::forward<Func>(func));
    } else {
        std::cerr << "fht::dispatch_log_n: unsupported log_N=" << log_n << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// Thread-level Hadamard butterfly (in-register)
// ============================================================================

template<int kLogN, int kNChunks>
__device__ __forceinline__ void butterfly_thread(float x[kNChunks][1 << kLogN]) {
    constexpr int N = 1 << kLogN;
    #pragma unroll
    for (int level = 0; level < kLogN; ++level) {
        const int stride = 1 << level;
        #pragma unroll
        for (int j = 0; j < N / 2; ++j) {
            const int lo  = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;
            #pragma unroll
            for (int c = 0; c < kNChunks; ++c) {
                float a = x[c][idx];
                float b = x[c][idx + stride];
                x[c][idx]          = a + b;
                x[c][idx + stride] = a - b;
            }
        }
    }
}

// ============================================================================
// Warp-level Hadamard butterfly (via shuffle)
// ============================================================================

template<int kLogWarpSize, int kStartLevel, int kNChunks, int kNItems>
__device__ __forceinline__ void butterfly_warp(float x[kNChunks][kNItems]) {
    constexpr int N = 1 << kLogWarpSize;
    const int lane_id = threadIdx.x % N;
    #pragma unroll
    for (int level = kStartLevel; level < kLogWarpSize; ++level) {
        const int mask = 1 << level;
        const float sign = (lane_id & mask) ? -1.f : 1.f;
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float other = __shfl_xor_sync(0xffffffff, x[c][i], mask);
                x[c][i] = sign * x[c][i] + other;
            }
        }
    }
}

// ============================================================================
// Cross-warp shared memory exchange
// ============================================================================

/// Layout: smem[i * kNThreads + idx] with XOR swizzle (col ^ row).
template<int kNChunks, int kNElts, int kWarpSize, int kNWarps, bool Pre>
__device__ __forceinline__ void exchange_via_smem(float x[kNChunks][kNElts], float* smem) {
    static_assert(kNElts == 4);
    constexpr int kNThreads = kWarpSize * kNWarps;
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    const int row_t = threadIdx.x % kNWarps;
    const int col_t = threadIdx.x / kNWarps;

    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        __syncthreads();
        int write_idx = Pre
            ? (warp_id * kWarpSize + (lane_id ^ warp_id))
            : (row_t   * kWarpSize + (col_t  ^ row_t));
        #pragma unroll
        for (int i = 0; i < kNElts; ++i)
            smem[i * kNThreads + write_idx] = x[c][i];
        __syncthreads();
        int read_idx = Pre
            ? (row_t   * kWarpSize + (col_t  ^ row_t))
            : (warp_id * kWarpSize + (lane_id ^ warp_id));
        #pragma unroll
        for (int i = 0; i < kNElts; ++i)
            x[c][i] = smem[i * kNThreads + read_idx];
    }
}

// ============================================================================
// Vectorized load / store (float4 = 16 bytes per thread)
// ============================================================================

template<int kNChunks, int kNElts>
__device__ __forceinline__ void vec_load(const float* src, float x[kNChunks][kNElts], int dim) {
    static_assert(kNElts == 4);
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        int offset = (c * blockDim.x + threadIdx.x) * kNElts;
        if (offset < dim) {
            float4 v = reinterpret_cast<const float4*>(src)[c * blockDim.x + threadIdx.x];
            x[c][0] = v.x;  x[c][1] = v.y;  x[c][2] = v.z;  x[c][3] = v.w;
        } else {
            x[c][0] = 0.f;  x[c][1] = 0.f;  x[c][2] = 0.f;  x[c][3] = 0.f;
        }
    }
}

template<int kNChunks, int kNElts>
__device__ __forceinline__ void vec_store(float* dst, float x[kNChunks][kNElts], int dim, float scale) {
    static_assert(kNElts == 4);
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        int offset = (c * blockDim.x + threadIdx.x) * kNElts;
        if (offset < dim) {
            float4 v;
            v.x = x[c][0] * scale;  v.y = x[c][1] * scale;
            v.z = x[c][2] * scale;  v.w = x[c][3] * scale;
            reinterpret_cast<float4*>(dst)[c * blockDim.x + threadIdx.x] = v;
        }
    }
}

// ============================================================================
// Sign flip in registers (using flip bits from shared memory)
// ============================================================================

template<int kNChunks, int kNElts>
__device__ __forceinline__ void apply_sign_flip(float x[kNChunks][kNElts],
                                                 const uint8_t* flip_smem) {
    static_assert(kNElts == 4);
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        int base_dim = (c * blockDim.x + threadIdx.x) * kNElts;
        int byte_idx = base_dim / 8;
        int bit_off  = base_dim % 8;
        uint8_t bits = flip_smem[byte_idx];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            if (bits & (1u << (bit_off + i)))
                x[c][i] = -x[c][i];
        }
    }
}

// ============================================================================
// Kernel traits
// ============================================================================

template<int kLogN_>
struct FhtTraits {
    static constexpr int kLogN    = kLogN_;
    static constexpr int kDim     = 1 << kLogN;
    static constexpr int kNElts   = 4;
    static constexpr int kNThreads = (kDim / kNElts <= 256) ? (kDim / kNElts) : 256;
    static constexpr int kNChunks  = kDim / (kNElts * kNThreads);
    static constexpr int kWarpSize = (kNThreads < 32) ? kNThreads : 32;
    static constexpr int kNWarps   = kNThreads / kWarpSize;

    static constexpr int kSmemExchange = kNElts * kNThreads * sizeof(float);
    static constexpr int kSmemFlip     = 4 * (kDim / 8);
    static constexpr int kSmemTotal    = kSmemExchange + kSmemFlip;
};

// ============================================================================
// FHT body — thread → warp → cross-warp → chunk butterfly
// ============================================================================

template<int kLogN, int kNChunks, int kNElts, int kWarpSize, int kNWarps>
__device__ __forceinline__ void fht_body(float x[kNChunks][kNElts], float* smem_exchange) {
    constexpr int kLogWarpSize = clog2(kWarpSize);
    constexpr int kLogNWarps   = clog2(kNWarps);

    butterfly_thread<clog2(kNElts), kNChunks>(x);
    butterfly_warp<kLogWarpSize, 0, kNChunks, kNElts>(x);

    if constexpr (kNWarps > 1) {
        exchange_via_smem<kNChunks, kNElts, kWarpSize, kNWarps, true>(x, smem_exchange);
        butterfly_warp<kLogNWarps, 0, kNChunks, kNElts>(x);
        exchange_via_smem<kNChunks, kNElts, kWarpSize, kNWarps, false>(x, smem_exchange);
    }

    if constexpr (kNChunks > 1) {
        float xt[kNElts][kNChunks];
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c)
            #pragma unroll
            for (int i = 0; i < kNElts; ++i)
                xt[i][c] = x[c][i];
        butterfly_thread<clog2(kNChunks), kNElts>(xt);
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c)
            #pragma unroll
            for (int i = 0; i < kNElts; ++i)
                x[c][i] = xt[i][c];
    }
}

// ============================================================================
// Fused 4-round rotation kernel (power-of-2 path)
// ============================================================================

// NOTE: `input` and `output` are intentionally NOT marked __restrict__ — this
// kernel supports in-place rotation (output may alias input). Each block reads
// its row fully into registers before any writeback, so aliasing is safe.
template<int kLogN>
__global__ void fht_kac_rotate_fused_kernel(
    const float* input,
    float* output,
    const uint8_t* __restrict__ flip,
    int N, float total_scale)
{
    using Tr = FhtTraits<kLogN>;
    constexpr int kNElts = Tr::kNElts, kNChunks = Tr::kNChunks, kDim = Tr::kDim;

    extern __shared__ char smem_raw_[];
    float*   smem_exchange = reinterpret_cast<float*>(smem_raw_);
    uint8_t* smem_flip     = reinterpret_cast<uint8_t*>(smem_raw_ + Tr::kSmemExchange);

    const int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    constexpr int kFlipBytes  = 4 * (kDim / 8);
    constexpr int kFlipStride = kDim / 8;
    for (int i = threadIdx.x; i < kFlipBytes; i += blockDim.x)
        smem_flip[i] = flip[i];
    __syncthreads();

    const float* vec_in = input + (size_t)vec_id * kDim;
    float x[kNChunks][kNElts];
    vec_load<kNChunks, kNElts>(vec_in, x, kDim);

    #pragma unroll
    for (int round = 0; round < 4; ++round) {
        apply_sign_flip<kNChunks, kNElts>(x, smem_flip + round * kFlipStride);
        fht_body<kLogN, kNChunks, kNElts, Tr::kWarpSize, Tr::kNWarps>(x, smem_exchange);
    }

    float* vec_out = output + (size_t)vec_id * kDim;
    vec_store<kNChunks, kNElts>(vec_out, x, kDim, total_scale);
}

// ============================================================================
// Fused 4-round rotation kernel (non-power-of-2 path)
// ============================================================================

// NOTE: `input`/`output` intentionally not __restrict__ — supports in-place
// rotation. Each block stages its row into shared memory before any writeback.
template<int kLogTrunc>
__global__ void fht_kac_rotate_fused_nonpow2_kernel(
    const float* input,
    float* output,
    const uint8_t* __restrict__ flip,
    int N, int padded_dim, float fac, float final_scale)
{
    using Tr = FhtTraits<kLogTrunc>;
    constexpr int kNElts    = Tr::kNElts;
    constexpr int kNChunks  = Tr::kNChunks;
    constexpr int kTruncDim = Tr::kDim;
    constexpr int kNThreads = Tr::kNThreads;

    extern __shared__ char smem_raw_[];
    float*   smem_data     = reinterpret_cast<float*>(smem_raw_);
    float*   smem_exchange = smem_data + padded_dim;
    uint8_t* smem_flip     = reinterpret_cast<uint8_t*>(smem_exchange + kNElts * kNThreads);

    const int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    const int start = padded_dim - kTruncDim;
    const int flip_bytes_per_round = padded_dim / 8;
    const int half_P = padded_dim / 2;

    for (int i = threadIdx.x; i < 4 * flip_bytes_per_round; i += kNThreads)
        smem_flip[i] = flip[i];
    const float* vec_in = input + (size_t)vec_id * padded_dim;
    for (int i = threadIdx.x; i < padded_dim; i += kNThreads)
        smem_data[i] = vec_in[i];
    __syncthreads();

    const int offsets[4] = {0, start, 0, start};

    for (int round = 0; round < 4; ++round) {
        // Sign flip on full padded_dim
        const uint8_t* round_flip = smem_flip + round * flip_bytes_per_round;
        for (int i = threadIdx.x; i < padded_dim; i += kNThreads) {
            if (round_flip[i / 8] & (1u << (i % 8)))
                smem_data[i] = -smem_data[i];
        }
        __syncthreads();

        // FHT on trunc_dim elements at offsets[round]
        int fht_off = offsets[round];
        float x[kNChunks][kNElts];
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            int base = fht_off + (c * kNThreads + threadIdx.x) * kNElts;
            x[c][0] = smem_data[base];     x[c][1] = smem_data[base + 1];
            x[c][2] = smem_data[base + 2]; x[c][3] = smem_data[base + 3];
        }

        fht_body<kLogTrunc, kNChunks, kNElts, Tr::kWarpSize, Tr::kNWarps>(x, smem_exchange);

        // Write back with per-round fac (can't defer — kacs_walk mixes scales)
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            int base = fht_off + (c * kNThreads + threadIdx.x) * kNElts;
            smem_data[base]     = x[c][0] * fac; smem_data[base + 1] = x[c][1] * fac;
            smem_data[base + 2] = x[c][2] * fac; smem_data[base + 3] = x[c][3] * fac;
        }
        __syncthreads();

        // Kacs walk
        for (int i = threadIdx.x; i < half_P; i += kNThreads) {
            float a = smem_data[i], b = smem_data[i + half_P];
            smem_data[i] = a + b;  smem_data[i + half_P] = a - b;
        }
        __syncthreads();
    }

    float* vec_out = output + (size_t)vec_id * padded_dim;
    for (int i = threadIdx.x; i < padded_dim; i += kNThreads)
        vec_out[i] = smem_data[i] * final_scale;
}

// ============================================================================
// Host-side launch helpers
// ============================================================================

template<int kLogN>
inline void launch_fused_rotate(const float* input, float* output,
                                 const uint8_t* flip, int N,
                                 float total_scale, cudaStream_t stream) {
    using Tr = FhtTraits<kLogN>;
    constexpr int smem = Tr::kSmemTotal;
    auto kernel = &fht_kac_rotate_fused_kernel<kLogN>;
    if (smem >= 48 * 1024)
        RAFT_CUDA_TRY(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<N, Tr::kNThreads, smem, stream>>>(input, output, flip, N, total_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
}

template<int kLogTrunc>
inline void launch_fused_rotate_nonpow2(const float* input, float* output,
                                         const uint8_t* flip, int N,
                                         int padded_dim, float fac,
                                         float final_scale, cudaStream_t stream) {
    using Tr = FhtTraits<kLogTrunc>;
    int smem = padded_dim * (int)sizeof(float)
             + Tr::kNElts * Tr::kNThreads * (int)sizeof(float)
             + 4 * (padded_dim / 8);
    auto kernel = &fht_kac_rotate_fused_nonpow2_kernel<kLogTrunc>;
    if (smem >= 48 * 1024)
        RAFT_CUDA_TRY(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<N, Tr::kNThreads, smem, stream>>>(input, output, flip, N, padded_dim, fac, final_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
}

// ============================================================================
// Runtime dispatch
// ============================================================================

inline void dispatch_fused_rotate(const float* input, float* output,
                                   const uint8_t* flip, int N, int log_N,
                                   float total_scale, cudaStream_t stream) {
    dispatch_log_n(log_N, [&](auto K) {
        launch_fused_rotate<K.value>(input, output, flip, N, total_scale, stream);
    });
}

inline void dispatch_fused_rotate_nonpow2(const float* input, float* output,
                                           const uint8_t* flip, int N,
                                           int log_trunc, int padded_dim,
                                           float fac, float final_scale,
                                           cudaStream_t stream) {
    dispatch_log_n(log_trunc, [&](auto K) {
        launch_fused_rotate_nonpow2<K.value>(input, output, flip, N,
                                              padded_dim, fac, final_scale, stream);
    });
}

}  // namespace fht
