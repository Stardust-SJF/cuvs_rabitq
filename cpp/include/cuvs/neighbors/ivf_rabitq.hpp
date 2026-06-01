/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_fp16.h>

#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq {

// forward declaration
namespace detail {
class IVFGPU;
}

/**
 * @defgroup ivf_rabitq_cpp_index_params IVF-RaBitQ index build parameters
 * @{
 */

/**
 * Random rotation implementation used at index build time.
 *
 * The chosen rotator is persisted in the saved index and the same rotation is
 * applied to queries at search time, so this is a build-time decision only.
 */
enum class rotator_kind : uint8_t {
  /** Full D×D rotation matrix via cuBLAS sgemm. O(N·D²) compute / O(D²) memory. */
  matmul = 0,
  /** Fast Hadamard Transform + Kac's walk. O(N·D·log D) compute / O(D) memory.
   *  No cuBLAS dependency. Recommended for high-dimensional datasets. */
  fht_kac = 1,
};

struct index_params : cuvs::neighbors::index_params {
  /**
   * The number of inverted lists (clusters)
   *
   * Hint: Increasing this parameter may alleviate shared memory pressure.
   */
  uint32_t n_lists = 1024;
  /**
   * The total number of bits per dimension (single bit required for the binary RaBitQ algorithm +
   * additional bits for extended RaBitQ).
   *
   * Supported values: [1, 2, 3, 4, 5, 6, 7, 8, 9].
   *
   * Hint: the smaller the 'bits_per_dim', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t bits_per_dim = 3;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The number of data vectors (per cluster) to use during iterative kmeans building. */
  uint32_t max_train_points_per_cluster = 256;
  /** Flag for using the fast quantize method */
  bool fast_quantize_flag = true;
  /**
   * Maximum number of vectors per batch when using streaming construction.
   *
   * This parameter controls the batch size during streaming construction from host memory.
   * Batches contain complete clusters only (no partial clusters across batch boundaries).
   *
   * Note: Streaming construction is automatically used when the dataset doesn't fit
   * comfortably in GPU memory (determined by available workspace and kTolerableRatio).
   */
  size_t streaming_batch_size = 100000;
  /**
   * Force streaming construction regardless of dataset size.
   *
   * When set to true, streaming construction will be used even if the dataset would fit
   * in GPU memory. This is useful for testing or when you want explicit control over
   * the construction method.
   *
   * Note: This parameter only applies when the input dataset is in host memory. If the
   * dataset is already in device memory, streaming construction is not applicable and
   * this parameter has no effect.
   *
   * Default: false (auto-detect based on available memory)
   */
  bool force_streaming = false;
  /**
   * Random rotation implementation. See `rotator_kind` for trade-offs.
   *
   * Default: `rotator_kind::matmul` (preserves existing behavior).
   */
  rotator_kind rotator = rotator_kind::matmul;
};
/**
 * @}
 */

/**
 * @defgroup ivf_rabitq_cpp_search_params IVF-RaBitQ index search parameters
 * @{
 */
/** A type for specifying the mode for searching the RaBitQ index. */
enum class search_mode {
  LUT16  = 0,
  LUT32  = 1,
  QUANT4 = 2,
  QUANT8 = 3,
};

/**
 * Per-block granularity for the search kernel's candidate-rerank stages.
 *
 * The exact-IP and IP2 stages run after the initial filter has admitted some
 * subset of vectors per block. With `auto` the kernel dispatches based on
 * the number of admitted candidates: dense buffers (high `num_candidates`)
 * use the "thread-per-cand / 1-warp-per-cand" Path A — fast for high ncand
 * because of coalesced HBM access; sparse buffers (low `num_candidates`) use
 * the "warp-per-cand / multi-warp-per-cand" Path B — recruits otherwise-idle
 * warps to share the per-candidate D-dim work. The forced variants are
 * mostly for ablation.
 */
enum class ip_variant_kind : uint8_t {
  auto_           = 0,  // hybrid dispatch (production default)
  thread_per_cand = 1,  // force Path A in both stages
  warp_per_cand   = 2,  // force Path B in both stages
};

/**
 * Threshold-seeding strategy for the in-kernel topk pruning during search.
 *
 * The search kernel maintains a per-query running max of the topk distances
 * found so far; new candidates are admitted only if their lower-bound estimate
 * is below this threshold. The initial value of that threshold determines how
 * aggressively the first cluster scanned can prune.
 */
enum class threshold_strategy : uint8_t {
  /** Threshold initialised to +infinity. The first cluster's main kernel
   *  admits every candidate; pruning only kicks in for subsequent clusters
   *  after the first cluster's topk has been computed. */
  none = 0,
  /** Threshold seeded as `centroid_reorder_scale * (distance from query to
   *  topk-th nearest centroid)`. The first cluster's main kernel can already
   *  prune candidates whose lower-bound exceeds this seed, materially cutting
   *  the candidate set on the first cluster scan. Production default. */
  centroid_reorder = 1,
};

/**
 * Algorithm used by the centroid-distance top-K (`raft::matrix::select_k`
 * call that picks the `n_probes` nearest clusters per query).
 *
 * The default `auto_policy` defers to raft's `kAuto` heuristic; the choice
 * is essentially neutral at our typical shape. The forced-algorithm values
 * are retained for ablation only.
 */
enum class centroid_select_kind : uint8_t {
  /** Defer to `raft::matrix::SelectAlgo::kAuto`. Production default. */
  auto_policy = 0,
  /** Alias for `auto_policy`; both pass raft's `kAuto`. */
  kauto = 1,
  /** Always use `raft::matrix::SelectAlgo::kWarpDistributedShm`. Errors at
   *  runtime if `n_probes > 256` (raft warpsort's kMaxCapacity). Ablation. */
  warp_distributed_shm = 2,
  /** Always use `raft::matrix::SelectAlgo::kRadix11bits`. Ablation. */
  radix11bits = 3,
};

struct search_params : cuvs::neighbors::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
  /** The search mode to be used. */
  search_mode mode = search_mode::QUANT4;
  /** Threshold-seeding strategy. See `threshold_strategy`. */
  threshold_strategy strategy = threshold_strategy::centroid_reorder;
  /** Scale factor for the CENTROID_REORDER seed threshold. Default 1.45.
   *  Ignored when `strategy == none`. */
  float centroid_reorder_scale = 1.45f;
  /** Number of nearest clusters per query that the CENTROID_REORDER pipeline
   *  promotes to a warmup pass, so they fire before the rest of the pairs
   *  in cluster-major order. The warmup wave tightens each query's topk
   *  threshold against its own nearest clusters' actual top-k before the
   *  bulk of the cluster-major work starts, which prunes more candidates
   *  during the rest pass. Set to 0 to skip the reorder and keep only the
   *  threshold seed. Ignored when `strategy == none`. */
  uint32_t warmup_clusters = 1;
  /** When true, the search kernel's blockDim is chosen at runtime based on
   *  device occupancy and total work (3 nested loops: max-blocks-per-SM,
   *  total-threads-cover-device, single-query-fills-SM, plus a Loop D bump
   *  to 512 at small nprobe). Floored at 256, capped at the kernel's
   *  maxThreadsPerBlock. When false, blockDim is the tuned default (256). */
  bool enable_dynamic_block = true;
  /** Pair-count threshold for the cluster-major sort. When `num_queries *
   *  n_probes < min_sort_pairs`, the sort is replaced by a single fused
   *  kernel that emits pairs in query-major order.
   *
   *  The cluster-major sort gives L2 reuse on the per-cluster bulk reads
   *  (multiple blocks scanning the same cluster share its data in L2);
   *  this benefit grows with the total number of (cluster, query) pairs.
   *  At small pair counts the sort overhead exceeds the reuse benefit and
   *  the simpler query-major fused build is faster.
   *
   *  Default 2000, picked around the empirical crossover where the L2-reuse
   *  benefit of sorting begins to outweigh the sort overhead.
   *
   *  Independently of this threshold, the sort is always skipped when
   *  `num_queries == 1` — at NQ=1 each cluster is visited at most once so
   *  there is no co-residency to amortise the sort against, and the sort
   *  is pure overhead regardless of pair count.
   *
   *  Set to 0 to disable the pair-count gate (still skips at NQ=1). */
  uint32_t min_sort_pairs = 2000;
  /** See `ip_variant_kind`. Default `auto_` runs the per-block hybrid
   *  dispatch; the other values force one path for ablation. */
  ip_variant_kind ip_variant = ip_variant_kind::auto_;
  /** See `centroid_select_kind`. Default `auto_policy` defers to raft's
   *  `kAuto` heuristic. */
  centroid_select_kind centroid_select = centroid_select_kind::auto_policy;
};
/**
 * @}
 */

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

/**
 * @defgroup ivf_rabitq_cpp_index IVF-RaBitQ index
 * @{
 */
/**
 * @brief IVF-RaBitQ index.
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename IdxT>
struct index : cuvs::neighbors::index {
  using index_params_type  = ivf_rabitq::index_params;
  using search_params_type = ivf_rabitq::search_params;
  using index_type         = IdxT;
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  index(const index&) = delete;
  index(
    index&&);  // declaration only; definition in impl to allow member unique_ptr of incomplete type
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&;  // declaration only; definition in impl to allow member
                                      // unique_ptr of incomplete type
  ~index();  // declaration only; definition in impl to allow member unique_ptr of incomplete type

  /**
   * @brief Construct an empty index yet to be populated.
   *
   */
  index(raft::resources const& handle);

  /** Construct an empty index yet to be populated. */
  index(raft::resources const& handle,
        size_t n_rows,
        uint32_t dim,
        uint32_t n_lists,
        uint32_t bits_per_dim,
        rotator_kind rotator = rotator_kind::matmul);

  /** Dimensionality of the input data. */
  uint32_t dim() const noexcept;

  /** Total length of the index. */
  IdxT size() const noexcept;

  /** Accessor for underlying RaBitQ index */
  detail::IVFGPU& rabitq_index() noexcept;

 private:
  std::unique_ptr<detail::IVFGPU> rabitq_index_;
};
/**
 * @}
 */

/**
 * @defgroup ivf_rabitq_cpp_index_build IVF-RaBitQ index build
 * @{
 */
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_rabitq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_rabitq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device_matrix_view to a row-major matrix [n_rows, dim]
 * @return the constructed ivf-rabitq index
 *
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_rabitq::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_rabitq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_rabitq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_rabitq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host_matrix_view to a row-major matrix [n_rows, dim]
 * @return the constructed ivf-rabitq index
 *
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_rabitq::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_rabitq::index<int64_t>;
/**
 * @}
 */

/**
 * @defgroup ivf_rabitq_cpp_index_search IVF-RaBitQ index search
 * @{
 */
/**
 * @brief Search ANN using the constructed index.
 *
 * @param[in] handle
 * @param[in] search_params configure the search
 * @param[in] index ivf-rabitq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_rabitq::search_params& search_params,
            cuvs::neighbors::ivf_rabitq::index<int64_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @}
 */

/**
 * @defgroup ivf_rabitq_cpp_serialize IVF-RaBitQ index serialize
 * @{
 */
/**
 * Save the index to file.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_rabitq::build(...);`
 * cuvs::neighbors::ivf_rabitq::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-RaBitQ index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               cuvs::neighbors::ivf_rabitq::index<int64_t>& index);

/**
 * Load index from file.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using IdxT = int64_t; // type of the index
 * // create an empty index
 * ivf_rabitq::index<IdxT> index(handle);
 *
 * cuvs::neighbors::ivf_rabitq::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index IVF-PQ index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_rabitq::index<int64_t>* index);
/**
 * @}
 */

}  // namespace cuvs::neighbors::ivf_rabitq
