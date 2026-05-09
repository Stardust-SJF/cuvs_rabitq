/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

#include "searcher_gpu.cuh"

#include <raft/util/cuda_dev_essentials.cuh>

#include <cstdint>
#include <cuda_runtime.h>

namespace cuvs::neighbors::ivf_rabitq::detail {
namespace {

static constexpr int BITS_PER_CHUNK = 4;
static constexpr int LUT_SIZE       = (1 << BITS_PER_CHUNK);  // 16

// --- Tunables ---
using T    = float;
using IdxT = uint32_t;

using lut_dtype = __half;  // FP16

// POD struct consolidating parameters for all computeInnerProducts* kernels
struct ComputeInnerProductsKernelParams {
  const ClusterQueryPair* d_sorted_pairs       = nullptr;
  const float* d_query                         = nullptr;
  const uint32_t* d_short_data                 = nullptr;
  const IVFGPU::GPUClusterMeta* d_cluster_meta = nullptr;
  float* d_lut_for_queries_float               = nullptr;
  lut_dtype* d_lut_for_queries_half            = nullptr;
  const uint32_t* d_packed_queries             = nullptr;  // Packed query bit planes
  const float* d_widths                        = nullptr;  // Query scaling factors
  const float* d_short_factors                 = nullptr;
  const float* d_G_k1xSumq                     = nullptr;
  const float* d_G_kbxSumq                     = nullptr;
  const float* d_centroid_distances            = nullptr;
  uint32_t topk                                = 0;
  uint32_t num_queries                         = 0;
  uint32_t nprobe                              = 0;
  uint32_t num_pairs                           = 0;
  uint32_t num_centroids                       = 0;
  uint32_t D                                   = 0;
  const float* d_threshold                     = nullptr;  // threshold for each query
  uint32_t max_candidates_per_pair             = 0;        // max storage per pair, 1000 suggested
  uint32_t max_candidates_per_query =
    0;  // max number of vectors in probed clusters for any particular query
  uint32_t ex_bits            = 0;        // bits per dimension in ex codes
  const uint8_t* d_long_code  = nullptr;  // long codes for all vectors
  const float* d_ex_factor    = nullptr;  // ex factors for distance computation
  const PID* d_pids           = nullptr;  // PIDs for all vectors
  float* d_topk_dists         = nullptr;  // output top-k distances
  PID* d_topk_pids            = nullptr;  // output top-k PIDs
  int* d_query_write_counters = nullptr;
  uint32_t num_bits           = 0;  // number of bits (8 for int8)
  uint32_t num_words          = 0;  // approx. D/32
  // Per-block granularity for the candidate-rerank stages. 0=auto (hybrid),
  // 1=force Path A, 2=force Path B. See cuvs::neighbors::ivf_rabitq::ip_variant_kind.
  uint8_t ip_variant          = 0;
};

// function to extract long codes
__device__ inline uint32_t extract_code(const uint8_t* codes, size_t d, size_t EX_BITS)
{
  size_t bitPos    = d * EX_BITS;
  size_t byteIdx   = bitPos >> 3;
  size_t bitOffset = bitPos & 7;
  uint32_t v       = codes[byteIdx] << 8;
  if (bitOffset + EX_BITS > 8) { v |= codes[byteIdx + 1]; }
  int shift = 16 - (bitOffset + EX_BITS);
  return (v >> shift) & ((1u << EX_BITS) - 1);
}

// Build d_sorted_pairs query-major directly from raft::matrix::select_k's
// output, skipping the cluster-major radix sort. Used when coresidency
// (avg pairs per cluster) is too low to amortise the sort over L2 reuse.
__global__ inline void build_query_major_pairs_kernel(const int* d_raft_idx,
                                                      ClusterQueryPair* d_sorted_pairs,
                                                      int batch_size,
                                                      int nprobe)
{
  int tid         = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = batch_size * nprobe;
  if (tid < total_pairs) {
    d_sorted_pairs[tid].cluster_idx = d_raft_idx[tid];
    d_sorted_pairs[tid].query_idx   = tid / nprobe;
  }
}

// Floor below which the search kernel's tuned shared-memory layout, MAX_TOP_K
// warpsort, and candidate-scan grid-stride loop assumptions degrade.
// Empirically (GBitQ bench_dynblock_coresidency.csv) shrinking below 256 caused
// 30-70% QPS regressions at NQ >= 100.
static constexpr uint32_t kSearchKernelMinBlockDim = 256;

// Choose a search-kernel blockDim based on device occupancy and total work.
// Reproduces cuvs's IVF-PQ compute_similarity heuristic with three nested
// while-loops plus a Loop D bump for small nprobe + plentiful total work.
//
// Loop A: ensure max-occupancy blocks could fill an SM threadwise.
// Loop B: ensure total threads (num_pairs * n_threads) cover the whole device.
// Loop C: at small num_queries, fill one SM with one query's threads (better
//         L1 hit rate for that query's per-cluster bulk reads).
// Loop D: at small nprobe (<=10) and Loops A/B/C settled at the floor, bump
//         to 512 to give the ex-code re-rank stage more warp-per-candidate
//         parallelism.
//
// Returns a power of two in [kSearchKernelMinBlockDim, kernel_max_threads_per_block].
static inline uint32_t compute_dynamic_block_dim(size_t num_queries,
                                                 size_t num_pairs,
                                                 const cudaDeviceProp& dev_props,
                                                 int kernel_max_threads_per_block)
{
  const uint32_t cap = (kernel_max_threads_per_block > 0)
                         ? static_cast<uint32_t>(kernel_max_threads_per_block)
                         : static_cast<uint32_t>(dev_props.maxThreadsPerBlock);
  uint32_t n_threads = static_cast<uint32_t>(raft::WarpSize);
  // Loop A
  while (static_cast<size_t>(dev_props.maxBlocksPerMultiProcessor) * n_threads <
           static_cast<size_t>(dev_props.maxThreadsPerMultiProcessor) &&
         n_threads < cap) {
    n_threads *= 2;
  }
  // Loop B
  while (num_pairs * n_threads < static_cast<size_t>(dev_props.multiProcessorCount) *
                                   dev_props.maxThreadsPerMultiProcessor &&
         n_threads < cap) {
    n_threads *= 2;
  }
  // Loop C
  while (num_queries * n_threads <
           static_cast<size_t>(dev_props.maxThreadsPerMultiProcessor) &&
         n_threads < cap) {
    n_threads *= 2;
  }
  // Loop D
  const size_t nprobe = (num_queries > 0) ? (num_pairs / num_queries) : 0;
  if (nprobe <= 10 && n_threads < 512u && cap >= 512u) { n_threads = 512u; }
  if (n_threads > cap) n_threads = cap;
  if (n_threads < kSearchKernelMinBlockDim) n_threads = kSearchKernelMinBlockDim;
  if (n_threads > cap) n_threads = cap;  // floor may exceed cap on a tiny kernel
  return n_threads;
}

// Threshold-seeding kernel for the CENTROID_REORDER strategy.
//
// For each query, picks the topk-th nearest cluster (rank = topk-1, clamped to
// nprobe-1) from raft::matrix::select_k's query-major output and seeds the
// per-query topk threshold to scale * dist(query, that cluster). The first
// cluster scanned by the main search kernel can then prune candidates whose
// lower-bound exceeds this seed.
//
// IMPORTANT: indexes into d_raft_idx (RAFT select_k output, query-major,
// distance-ascending) and NOT d_sorted_pairs (cluster-major after the radix
// sort). With NQ > 1 the cluster-major layout would mix queries together.
__global__ inline void seed_threshold_from_centroid_kernel(const int* d_raft_idx,
                                                           const float* d_centroid_distances,
                                                           float* d_threshold_batch,
                                                           size_t num_queries,
                                                           size_t num_centroids,
                                                           size_t nprobe,
                                                           size_t topk,
                                                           float scale)
{
  size_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;
  size_t rank      = (topk > 0 && topk - 1 < nprobe) ? (topk - 1) : (nprobe - 1);
  int cluster_idx  = d_raft_idx[q * nprobe + rank];
  float q_g_add    = d_centroid_distances[q * num_centroids + cluster_idx];
  d_threshold_batch[q] = q_g_add * scale;
}

}  // namespace
}  // namespace cuvs::neighbors::ivf_rabitq::detail
