/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/24/25.
//

#pragma once

#include "../defines.hpp"

#include <cuvs/neighbors/ivf_rabitq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <fstream>

namespace cuvs::neighbors::ivf_rabitq::detail {

/// Unified GPU rotator supporting both matrix multiplication and FHT+Kac.
///
/// The rotator kind is chosen at construction time and persisted on save/load.
/// All callers use the same rotate()/save()/load() interface regardless of the
/// underlying implementation.
class RotatorGPU {
 public:
  /**
   * @brief Construct a new RotatorGPU.
   * @param handle  raft resources handle.
   * @param dim     Original (unpadded) vector dimension. Padded dimension D is
   *                round_up_to_multiple_of(dim, 32).
   * @param kind    Rotator implementation to use.
   */
  explicit RotatorGPU(raft::resources const& handle,
                      uint32_t dim,
                      rotator_kind kind = rotator_kind::matmul);

  // Disable copy assignment
  RotatorGPU& operator=(const RotatorGPU& other) = delete;

  /// @return Padded dimension.
  size_t size() const;

  /// @return The rotator kind.
  rotator_kind kind() const { return kind_; }

  /**
   * @brief Load rotator from file.
   *
   * Reads a one-byte kind tag, then dispatches to the matching loader. If the
   * file's kind differs from this instance's current kind, the instance is
   * reinitialised to match.
   */
  void load(std::ifstream& input);

  /**
   * @brief Save rotator to file.
   *
   * Format: [uint8_t kind_tag] [kind-specific data].
   */
  void save(std::ofstream& output) const;

  /**
   * @brief Rotate N vectors of D floats.
   * @param d_A      Input:  N × D matrix on device (row-major).
   * @param d_RAND_A Output: N × D matrix on device (row-major).
   * @param N        Number of vectors.
   *
   * In-place aliasing (`d_A == d_RAND_A`) is allowed only when
   * supports_inplace_rotate() returns true (currently fht_kac only).
   */
  void rotate(const float* d_A, float* d_RAND_A, size_t N) const;

  /// @return True if rotate() may be called with d_A == d_RAND_A.
  bool supports_inplace_rotate() const { return kind_ == rotator_kind::fht_kac; }

 private:
  raft::resources const& handle_;  // reusable resource handle
  rmm::cuda_stream_view stream_ =
    raft::resource::get_cuda_stream(handle_);  // CUDA stream obtained from handle_
  rotator_kind kind_;
  size_t D = 0;  // Padded dimension

  // ---- matmul members (used when kind_ == matmul) ----
  raft::device_matrix<float, int64_t, raft::row_major> rotation_matrix_ =
    raft::make_device_matrix<float, int64_t, raft::row_major>(handle_, 0, 0);

  // ---- fht_kac members (used when kind_ == fht_kac) ----
  size_t trunc_dim_ = 0;  // 1 << floor_log2(D), largest power-of-2 <= padded dimension
  float fac_        = 0;  // 1 / sqrt(trunc_dim)
  int log_N_        = 0;  // log2(trunc_dim), for FHT kernel dispatch
  raft::device_vector<uint8_t, int64_t> flip_bits_ =
    raft::make_device_vector<uint8_t, int64_t>(handle_, 0);  // 4 * D / 8 bytes of random sign bits

  void init_matmul(uint32_t dim);
  void init_fht_kac(uint32_t dim);
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
