/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/24/25.
//

#include "rotator_gpu.cuh"
#include "fht_cuda.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/detail/qr.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/integer_utils.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

namespace {

inline size_t floor_log2_size(size_t x)
{
  size_t r = 0;
  while (x >>= 1) {
    ++r;
  }
  return r;
}

}  // namespace

RotatorGPU::RotatorGPU(raft::resources const& handle, uint32_t dim, rotator_kind kind)
  : handle_(handle), kind_(kind)
{
  if (kind_ == rotator_kind::matmul) {
    init_matmul(dim);
  } else {
    init_fht_kac(dim);
  }
}

void RotatorGPU::init_matmul(uint32_t dim)
{
  D                = raft::round_up_safe<uint32_t>(dim, 64u);
  rotation_matrix_ = raft::make_device_matrix<float, int64_t, raft::row_major>(handle_, D, D);
  raft::random::RngState rng(7ULL);
  raft::random::normal(handle_, rng, rotation_matrix_.data_handle(), D * D, 0.0f, 1.0f);
  raft::linalg::detail::qrGetQ_inplace(handle_, rotation_matrix_.data_handle(), D, D, stream_);
}

void RotatorGPU::init_fht_kac(uint32_t dim)
{
  D = raft::round_up_safe<uint32_t>(dim, 64u);

  size_t bottom_log = floor_log2_size(D);
  trunc_dim_        = 1ULL << bottom_log;
  log_N_            = static_cast<int>(bottom_log);
  fac_              = 1.0f / std::sqrt(static_cast<float>(trunc_dim_));

  // 4 rounds × (padded_dim bits per round) / (8 bits per byte)
  size_t flip_bytes = 4 * D / 8;
  flip_bits_        = raft::make_device_vector<uint8_t, int64_t>(handle_, flip_bytes);

  // Random sign bits generated on host with std::random_device → mt19937,
  // matching the GBitQ reference. Non-deterministic across builds by design.
  std::vector<uint8_t> h_flip(flip_bytes);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto& b : h_flip) {
    b = static_cast<uint8_t>(dist(gen));
  }
  raft::copy(flip_bits_.data_handle(), h_flip.data(), flip_bytes, stream_);
  raft::resource::sync_stream(handle_);
}

size_t RotatorGPU::size() const { return D; }

// ============================================================================
// Save / Load — file format: [uint8_t kind_tag] [kind-specific data]
// ============================================================================

void RotatorGPU::save(std::ofstream& output) const
{
  uint8_t tag = static_cast<uint8_t>(kind_);
  output.write(reinterpret_cast<const char*>(&tag), sizeof(tag));

  if (kind_ == rotator_kind::matmul) {
    auto host_buf = raft::make_host_vector<float, int64_t>(D * D);
    raft::copy(host_buf.data_handle(), rotation_matrix_.data_handle(), D * D, stream_);
    raft::resource::sync_stream(handle_);
    for (size_t i = 0; i < D * D; ++i) {
      output.write(reinterpret_cast<char*>(&host_buf(i)), sizeof(float));
    }
  } else {
    size_t flip_bytes = 4 * D / 8;
    std::vector<uint8_t> h_flip(flip_bytes);
    raft::copy(h_flip.data(), flip_bits_.data_handle(), flip_bytes, stream_);
    raft::resource::sync_stream(handle_);
    output.write(reinterpret_cast<const char*>(h_flip.data()), static_cast<long>(flip_bytes));
  }
}

void RotatorGPU::load(std::ifstream& input)
{
  uint8_t tag;
  input.read(reinterpret_cast<char*>(&tag), sizeof(tag));
  rotator_kind file_kind = static_cast<rotator_kind>(tag);

  // If the on-disk kind differs from the current instance's kind, reinitialise
  // the instance to match. D is already set from the constructor (which was
  // called with the right dim).
  if (file_kind != kind_) {
    kind_ = file_kind;
    if (kind_ == rotator_kind::matmul) {
      rotation_matrix_ = raft::make_device_matrix<float, int64_t, raft::row_major>(handle_, D, D);
    } else {
      // Recompute FHT-Kac parameters in the padded vector space used by RaBitQ.
      size_t bottom_log = floor_log2_size(D);
      trunc_dim_        = 1ULL << bottom_log;
      log_N_            = static_cast<int>(bottom_log);
      fac_              = 1.0f / std::sqrt(static_cast<float>(trunc_dim_));
      size_t flip_bytes = 4 * D / 8;
      flip_bits_        = raft::make_device_vector<uint8_t, int64_t>(handle_, flip_bytes);
    }
  }

  if (kind_ == rotator_kind::matmul) {
    auto host_buf = raft::make_host_vector<float, int64_t>(D * D);
    for (size_t i = 0; i < D * D; ++i) {
      input.read(reinterpret_cast<char*>(&host_buf(i)), sizeof(float));
    }
    raft::copy(rotation_matrix_.data_handle(), host_buf.data_handle(), D * D, stream_);
    raft::resource::sync_stream(handle_);
  } else {
    size_t flip_bytes = 4 * D / 8;
    std::vector<uint8_t> h_flip(flip_bytes);
    input.read(reinterpret_cast<char*>(h_flip.data()), static_cast<long>(flip_bytes));
    raft::copy(flip_bits_.data_handle(), h_flip.data(), flip_bytes, stream_);
    raft::resource::sync_stream(handle_);
  }
}

// ============================================================================
// Rotate
// ============================================================================
//
// Rotate matrix A and store the result in RAND_A on the GPU. Both A and RAND_A
// are row-major N×D matrices on device. matmul: cuBLAS sgemm via raft wrapper
// (no aliasing). fht_kac: fused 4-round (sign flip + FHT [+ Kac's walk]) kernels
// that support input/output aliasing.
void RotatorGPU::rotate(const float* d_A, float* d_RAND_A, size_t N) const
{
  auto stream_view = raft::resource::get_cuda_stream(handle_);
  if (kind_ == rotator_kind::matmul) {
    // cuBLAS assumes column-major. Our matrices are row-major, so we compute
    //   RAND_A^T = P^T * A^T
    // which in row-major gives RAND_A = A * P.
    raft::linalg::gemm(
      handle_,
      raft::make_device_matrix_view<float, int64_t, raft::col_major>(
        const_cast<float*>(rotation_matrix_.data_handle()), D, D),
      raft::make_device_matrix_view<float, int64_t, raft::col_major>(
        const_cast<float*>(d_A), D, N),
      raft::make_device_matrix_view<float, int64_t, raft::col_major>(d_RAND_A, D, N));
  } else {
    cudaStream_t stream = stream_view.value();
    if (trunc_dim_ == D) {
      // Power-of-2 path: total scale fac^4 deferred to the end.
      float total_scale = fac_ * fac_ * fac_ * fac_;
      fht::dispatch_fused_rotate(d_A,
                                 d_RAND_A,
                                 flip_bits_.data_handle(),
                                 static_cast<int>(N),
                                 log_N_,
                                 total_scale,
                                 stream);
    } else {
      // Non-power-of-2 path: fac per round, final 0.25 deferred.
      fht::dispatch_fused_rotate_nonpow2(d_A,
                                         d_RAND_A,
                                         flip_bits_.data_handle(),
                                         static_cast<int>(N),
                                         log_N_,
                                         static_cast<int>(D),
                                         fac_,
                                         0.25f,
                                         stream);
    }
  }
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
