// Copyright 2026 The Spyre-Inference Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Portions adapted from vLLM (https://github.com/vllm-project/vllm),
// Copyright contributors to the vLLM project, Apache-2.0.

#include "cpu_types.hpp"

#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

// Fused Gumbel-max kernel: argmax_i(logit_i + g_i) with g_i ~ Gumbel(0, 1) is
// distributed as softmax(logits). The noise is generated per element from a
// counter-based hash of (seed, i), so every row gets its own stream and seeded
// requests reproduce. (A shared precomputed table, as in vLLM, restricts each
// row to 2^20 noise windows and leaves much of a large vocab unreachable.)

constexpr uint64_t SPLITMIX_GAMMA = 0x9E3779B97F4A7C15ULL;

static inline uint64_t splitmix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// u in (0, 1) from 53 random bits, so -log(-log u) <= 54 ln 2 < GUMBEL_MAX.
constexpr double GUMBEL_MAX = 37.5;

static void fused_gumbel_argmax_kernel(int64_t* __restrict__ output,
                                       const float* __restrict__ logits,
                                       const int64_t* __restrict__ seeds,
                                       const int64_t batch_size,
                                       const int64_t vocab_size) {
#pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < batch_size; ++b) {
    const float* row = logits + b * vocab_size;
    const uint64_t key = splitmix64(static_cast<uint64_t>(seeds[b]));

    double best_score = -std::numeric_limits<double>::infinity();
    int64_t best_idx = 0;
    for (int64_t i = 0; i < vocab_size; ++i) {
      const double x = row[i];
      // Exact pruning: the noise is bounded, so a token this far below the
      // running best cannot win. Skips every -inf left by top-k/top-p.
      if (x + GUMBEL_MAX <= best_score) continue;
      const uint64_t bits =
          splitmix64(key + static_cast<uint64_t>(i + 1) * SPLITMIX_GAMMA) >> 11;
      // Wins iff -log u < exp(x - best); -log u >= 1 - u rejects most tokens
      // with one exp instead of two logs.
      const double one_minus_u =
          (static_cast<double>((1ULL << 53) - 1 - bits) + 0.5) * 0x1.0p-53;
      if (one_minus_u >= std::exp(x - best_score)) continue;
      const double u = (static_cast<double>(bits) + 0.5) * 0x1.0p-53;
      const double score = x - std::log(-std::log(u));
      if (score > best_score) {
        best_score = score;
        best_idx = i;
      }
    }
    output[b] = best_idx;
  }
}

static void greedy_argmax_kernel(int64_t* __restrict__ output,
                                 const float* __restrict__ logits,
                                 const int64_t batch_size,
                                 const int64_t vocab_size) {
  constexpr int VEC_ELEM_NUM = vec_op::FP32Vec16::VEC_ELEM_NUM;
  const int64_t vec_end = vocab_size - (vocab_size % VEC_ELEM_NUM);

#pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < batch_size; ++b) {
    const float* row = logits + b * vocab_size;

    // Vector max has no consistent NaN rule, so NaNs are caught via the sum:
    // any NaN (or inf + -inf, a harmless false positive) makes it NaN.
    vec_op::FP32Vec16 vmax(-std::numeric_limits<float>::infinity());
    vec_op::FP32Vec16 vsum(0.0f);
    for (int64_t i = 0; i < vec_end; i += VEC_ELEM_NUM) {
      vec_op::FP32Vec16 v(row + i);
      vmax = vmax.max(v);
      vsum = vsum + v;
    }
    float best_val = vmax.reduce_max();
    bool maybe_nan = std::isnan(vsum.reduce_sum());
    for (int64_t i = vec_end; i < vocab_size; ++i) {
      maybe_nan |= std::isnan(row[i]);
      if (row[i] > best_val) {
        best_val = row[i];
      }
    }

    int64_t best_idx = -1;
    if (maybe_nan) {
      // torch.argmax treats NaN as the maximum: return the first NaN.
      for (int64_t i = 0; i < vocab_size; ++i) {
        if (std::isnan(row[i])) {
          best_idx = i;
          break;
        }
      }
    }
    if (best_idx < 0) {
      best_idx = 0;
      for (int64_t i = 0; i < vocab_size; ++i) {
        if (row[i] == best_val) {
          best_idx = i;
          break;
        }
      }
    }
    output[b] = best_idx;
  }
}

}  // namespace

torch::Tensor fused_gumbel_argmax(const torch::Tensor& logits,
                                  const torch::Tensor& seeds) {
  TORCH_CHECK(logits.device().is_cpu(), "logits must be a CPU tensor");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [batch, vocab]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");
  TORCH_CHECK(seeds.device().is_cpu(), "seeds must be a CPU tensor");
  TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
  TORCH_CHECK(seeds.dim() == 1 && seeds.size(0) == logits.size(0),
              "seeds must be 1-D with batch_size elements");

  auto logits_contig = logits.contiguous();
  auto seeds_contig = seeds.contiguous();
  auto output = torch::empty({logits_contig.size(0)}, torch::kInt64);
  fused_gumbel_argmax_kernel(output.data_ptr<int64_t>(),
                             logits_contig.data_ptr<float>(),
                             seeds_contig.data_ptr<int64_t>(),
                             logits_contig.size(0), logits_contig.size(1));
  return output;
}

torch::Tensor greedy_argmax(const torch::Tensor& logits) {
  TORCH_CHECK(logits.device().is_cpu(), "logits must be a CPU tensor");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [batch, vocab]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");

  // A single row is faster in ATen: OpenMP fork/join dominates the scan
  // (~1 ms vs ~50 µs on s390x at vocab=32k).
  if (logits.size(0) <= 1) {
    return logits.argmax(/*dim=*/-1);
  }

  auto logits_contig = logits.contiguous();
  auto output = torch::empty({logits_contig.size(0)}, torch::kInt64);
  greedy_argmax_kernel(output.data_ptr<int64_t>(),
                       logits_contig.data_ptr<float>(), logits_contig.size(0),
                       logits_contig.size(1));
  return output;
}
