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
  // Loop D — bump to 512 at small nprobe when the grid is small enough that
  // even the floor block size (kSearchKernelMinBlockDim = 256) would
  // underfill the device. The bigger block recruits otherwise-idle warps
  // for the ex-code re-rank stage's warp-per-candidate parallelism.
  //
  // The underutilization check is critical: at large batch_size the grid is
  // already huge (~100k blocks across ~142 SMs), and bumping blockDim from
  // 256 → 512 only reduces per-SM occupancy without finding any idle warps
  // to recruit. Empirically this caused a 10% perf cliff at np=10 vs np=12
  // on openai_1536_5M / bs=10000 / quant4 / bits_per_dim=8 (where the
  // kernel is shmem-heavy, so larger blocks drop occupancy hard).
  const size_t nprobe = (num_queries > 0) ? (num_pairs / num_queries) : 0;
  const size_t device_thread_capacity =
    static_cast<size_t>(dev_props.multiProcessorCount) * dev_props.maxThreadsPerMultiProcessor;
  const bool device_underfilled_at_floor =
    num_pairs * static_cast<size_t>(kSearchKernelMinBlockDim) < device_thread_capacity;
  if (nprobe <= 10 && n_threads < 512u && cap >= 512u && device_underfilled_at_floor) {
    n_threads = 512u;
  }
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

// Representative-based seed threshold: provably-correct upper bound on the
// K-th-best distance.
//
// For each query q, we have N_reps representatives r_1..r_N from the top-N
// nearest clusters (per centroid distance). Their distances to q form a set
// {d_1,...,d_N}; the K-th smallest of this set is an upper bound on the true
// K-th-best (because we have N ≥ K real database vectors with these
// distances). We pass that as the initial threshold.
//
// Launch: one block per query. Each block loads the query into shared, then
// each warp computes one (q, r_i) squared-distance via a strided dot product.
// Thread 0 picks the K-th smallest from N_reps and writes the threshold.
//
// Shared mem: D_padded * sizeof(float) for query + N_reps * sizeof(float) for
// the per-rep distances.
template <int BlockSize>
__global__ void seed_threshold_from_representative_kernel(const int* __restrict__ d_raft_idx,
                                                          const float* __restrict__ d_query,
                                                          const float* __restrict__ d_representatives,
                                                          const float* __restrict__ d_centroid_distances,
                                                          float* __restrict__ d_threshold_batch,
                                                          size_t num_centroids,
                                                          size_t nprobe,
                                                          int N_reps,
                                                          int topk,
                                                          size_t D_padded,
                                                          float fallback_scale)
{
  const int q = blockIdx.x;
  extern __shared__ float smem_seed[];
  float* shared_query = smem_seed;
  float* shared_dists = smem_seed + D_padded;

  const float* q_ptr = d_query + static_cast<size_t>(q) * D_padded;
  for (size_t i = threadIdx.x; i < D_padded; i += BlockSize) {
    shared_query[i] = q_ptr[i];
  }
  __syncthreads();

  const int lane      = threadIdx.x % 32;
  const int warp_id   = threadIdx.x / 32;
  const int num_warps = BlockSize / 32;

  for (int rep_idx = warp_id; rep_idx < N_reps; rep_idx += num_warps) {
    int cluster_id            = d_raft_idx[static_cast<size_t>(q) * nprobe + rep_idx];
    const float* rep_ptr      = d_representatives + static_cast<size_t>(cluster_id) * D_padded;
    float diff_sq             = 0.0f;
    for (size_t d = lane; d < D_padded; d += 32) {
      float diff = shared_query[d] - rep_ptr[d];
      diff_sq += diff * diff;
    }
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
      diff_sq += __shfl_down_sync(0xFFFFFFFF, diff_sq, off);
    }
    if (lane == 0) shared_dists[rep_idx] = diff_sq;
  }
  __syncthreads();

  // Thread 0 picks the K-th smallest of N_reps using a length-K running max.
  // O(N*K); fine for K≤topk_limit (~32) and N≤2*K.
  if (threadIdx.x == 0) {
    constexpr int kMaxTopK = 64;
    float topk_dists[kMaxTopK];
    int count       = 0;
    int k           = topk < kMaxTopK ? topk : kMaxTopK;
    int n           = N_reps < k ? N_reps : N_reps;
    for (int i = 0; i < N_reps; i++) {
      float d = shared_dists[i];
      if (count < k) {
        topk_dists[count++] = d;
      } else {
        int max_i = 0;
        for (int j = 1; j < k; j++) {
          if (topk_dists[j] > topk_dists[max_i]) max_i = j;
        }
        if (d < topk_dists[max_i]) topk_dists[max_i] = d;
      }
    }
    float kth = topk_dists[0];
    for (int j = 1; j < count; j++) {
      if (topk_dists[j] > kth) kth = topk_dists[j];
    }
    // Fallback safety: take the min with the centroid-based heuristic bound.
    // Both are provable upper bounds on the K-th-best distance, so min is
    // still valid — guaranteed to be ≤ either alone.
    int rank_kth =
      (topk > 0 && static_cast<size_t>(topk - 1) < nprobe) ? (topk - 1) : (int)(nprobe - 1);
    int kth_cluster      = d_raft_idx[static_cast<size_t>(q) * nprobe + rank_kth];
    float centroid_bound = d_centroid_distances[static_cast<size_t>(q) * num_centroids + kth_cluster] *
                           fallback_scale;
    d_threshold_batch[q] = kth < centroid_bound ? kth : centroid_bound;
  }
}

// Fused CENTROID_REORDER setup kernel.
//
// One launch produces all three setup outputs needed by the reorder pipeline:
//   - d_warmup_pairs: query-major (cluster, query) pairs for the warmup pass.
//     For each query q, the first `warmup_clusters` entries are q's nearest
//     clusters (sourced from d_raft_idx[q, 0..warmup_clusters-1]).
//   - d_warmup_per_query: same cluster IDs flattened to int[], used by the
//     mark-keep kernel to test pair membership.
//   - d_threshold_batch (seed): seeds each query's topk threshold to
//     `scale * dist(q, topk-th nearest centroid)`. Same semantics as the
//     standalone seed_threshold_from_centroid_kernel above; this kernel
//     replaces it when the reorder pipeline is active.
//
// Each output is independently nullable; passing nullptr skips that output.
// The grid launches `num_queries * max(warmup_clusters, 1)` threads.
__global__ inline void fused_seed_warmup_kernel(const int* d_raft_idx,
                                                const float* d_centroid_distances,
                                                ClusterQueryPair* d_warmup_pairs,
                                                int* d_warmup_per_query,
                                                float* d_threshold_batch,
                                                size_t num_queries,
                                                size_t num_centroids,
                                                size_t nprobe,
                                                size_t topk,
                                                size_t warmup_clusters,
                                                float scale,
                                                bool emit_seed)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Warmup outputs: one thread per (q, k) where k ∈ [0, warmup_clusters).
  if (warmup_clusters > 0) {
    size_t total_warmup = num_queries * warmup_clusters;
    if (tid < total_warmup) {
      size_t q        = tid / warmup_clusters;
      size_t k        = tid % warmup_clusters;
      int cluster_idx = d_raft_idx[q * nprobe + k];
      if (d_warmup_pairs != nullptr) {
        d_warmup_pairs[tid].cluster_idx = cluster_idx;
        d_warmup_pairs[tid].query_idx   = static_cast<int>(q);
      }
      if (d_warmup_per_query != nullptr) { d_warmup_per_query[tid] = cluster_idx; }
    }
  }

  // Threshold seed: emitted by the k=0 thread of each query (or by tid==q
  // when warmup_clusters == 0).
  if (emit_seed && d_threshold_batch != nullptr) {
    bool emit;
    size_t q_for_seed;
    if (warmup_clusters > 0) {
      emit       = (tid < num_queries * warmup_clusters) && ((tid % warmup_clusters) == 0);
      q_for_seed = tid / warmup_clusters;
    } else {
      emit       = (tid < num_queries);
      q_for_seed = tid;
    }
    if (emit) {
      size_t seed_rank     = (topk > 0 && topk - 1 < nprobe) ? (topk - 1) : (nprobe - 1);
      int seed_cluster_idx = d_raft_idx[q_for_seed * nprobe + seed_rank];
      float q_g_add        = d_centroid_distances[q_for_seed * num_centroids + seed_cluster_idx];
      d_threshold_batch[q_for_seed] = q_g_add * scale;
    }
  }
}

// Mark which (cluster, query) pairs in d_sorted_pairs are NOT in the warmup
// set. Output is a byte flag array consumable by cub::DeviceSelect::Flagged
// to compact the rest pairs (cluster-major order is preserved). `warmup_clusters`
// is tiny (1-4) so the linear scan over d_warmup_per_query is cheap.
__global__ inline void fused_mark_keep_kernel(const ClusterQueryPair* d_sorted_pairs,
                                              const int* d_warmup_per_query,
                                              uint8_t* d_keep_flags,
                                              size_t total_pairs,
                                              size_t warmup_clusters)
{
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= total_pairs) return;
  int q          = d_sorted_pairs[k].query_idx;
  int c          = d_sorted_pairs[k].cluster_idx;
  bool is_warmup = false;
  for (size_t i = 0; i < warmup_clusters; ++i) {
    if (d_warmup_per_query[q * warmup_clusters + i] == c) {
      is_warmup = true;
      break;
    }
  }
  d_keep_flags[k] = is_warmup ? 0u : 1u;
}

// Concatenate warmup_pairs ++ rest_pairs into a single buffer. Warmup pairs
// land at low blockIdx so they fire in the first wave of the search kernel,
// tightening per-query thresholds before the rest pairs run.
__global__ inline void concat_pairs_kernel(const ClusterQueryPair* warmup_pairs,
                                           const ClusterQueryPair* rest_pairs,
                                           ClusterQueryPair* dst,
                                           size_t warmup_n,
                                           size_t rest_n)
{
  size_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = warmup_n + rest_n;
  if (tid >= total) return;
  if (tid < warmup_n) {
    dst[tid] = warmup_pairs[tid];
  } else {
    dst[tid] = rest_pairs[tid - warmup_n];
  }
}

}  // namespace
}  // namespace cuvs::neighbors::ivf_rabitq::detail
