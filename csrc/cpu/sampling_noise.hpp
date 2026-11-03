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
// Vendored from vLLM (https://github.com/vllm-project/vllm),
// csrc/cpu/sampling/common.h at 040abb262c, Copyright contributors to the vLLM
// project, Apache-2.0. Only the noise: the counter-based Gumbel draws keyed on
// (seed, position, token), unchanged. Re-sync verbatim; Spyre's argmax loop,
// with its exact pruning, stays in sampling_kernels.cpp.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cpu_sampling {

// Draft proposals and target verification must draw from disjoint noise
// streams even when they use the same request seed and token position.
inline constexpr int64_t DRAFT_NOISE_SALT = 1 << 30;

inline uint32_t murmur3_rotl32(uint32_t value, int shift) {
  return (value << shift) | (value >> (32 - shift));
}

inline uint32_t murmur3_mix(uint32_t hash, uint32_t key) {
  key *= 0xcc9e2d51;
  key = murmur3_rotl32(key, 15);
  key *= 0x1b873593;
  hash ^= key;
  hash = murmur3_rotl32(hash, 13);
  return hash * 5 + 0xe6546b64;
}

inline uint32_t murmur3_fmix32(uint32_t hash) {
  hash ^= hash >> 16;
  hash *= 0x85ebca6b;
  hash ^= hash >> 13;
  hash *= 0xc2b2ae35;
  return hash ^ (hash >> 16);
}

inline uint32_t murmur3_hash32_prefix(int64_t seed, int64_t position,
                                      uint32_t domain) {
  // Seed and position are shared by every vocabulary entry in a row. Mix this
  // prefix once so the inner loop only needs to mix the token ID and finalize.
  const uint64_t seed_bits = static_cast<uint64_t>(seed);
  uint32_t hash = domain;
  hash = murmur3_mix(hash, static_cast<uint32_t>(seed_bits));
  hash = murmur3_mix(hash, static_cast<uint32_t>(seed_bits >> 32));
  hash = murmur3_mix(hash, static_cast<uint32_t>(position));
  return hash;
}

inline uint32_t murmur3_hash32(int64_t seed, int64_t position,
                               uint32_t token_id, uint32_t domain = 0) {
  uint32_t hash = murmur3_hash32_prefix(seed, position, domain);
  hash = murmur3_mix(hash, token_id);
  return murmur3_fmix32(hash ^ 16);
}

inline float murmur3_uniform32(int64_t seed, int64_t position,
                               uint32_t token_id) {
  const uint32_t random = murmur3_hash32(seed, position, token_id);
  // Splitting the hash makes both integer-to-float conversions exact:
  //   u = round_fp32((random + 0.5) / 2^32).
  // Converting random directly to FP32 could discard up to eight low bits.
  const int32_t high = static_cast<int32_t>(random >> 16);
  const int32_t low = static_cast<int32_t>(random & 0xffff);
  return static_cast<float>(high) * 1.52587890625e-05f +
         (static_cast<float>(low) + 0.5f) * 2.3283064365386963e-10f;
}

inline double murmur3_uniform64(int64_t seed, int64_t position,
                                uint32_t token_id) {
  const uint64_t low = murmur3_hash32(seed, position, token_id);
  const uint64_t high = murmur3_hash32(seed, position, token_id, 0x9e3779b9);
  // Combine two hashes and retain the 53 bits representable by an FP64
  // significand: u = (random53 + 0.5) / 2^53.
  const uint64_t random53 = ((high << 32) | low) >> 11;
  double uniform = (static_cast<double>(random53) + 0.5) * 0x1p-53;
  if (uniform == 1.0) uniform -= 0x1p-53;
  return uniform;
}

inline float log1p_neg_stable(float value) {
  // This is inherited from the triton impl:
  // For small x, evaluate the degree-8 expansion
  //   log(1-x) = -x * (1 + x/2 + x^2/3 + ... + x^7/8)
  // instead of forming 1-x, which loses the low bits of x.
  float polynomial = 1.0f / 8.0f;
  polynomial = 1.0f / 7.0f + value * polynomial;
  polynomial = 1.0f / 6.0f + value * polynomial;
  polynomial = 1.0f / 5.0f + value * polynomial;
  polynomial = 1.0f / 4.0f + value * polynomial;
  polynomial = 1.0f / 3.0f + value * polynomial;
  polynomial = 1.0f / 2.0f + value * polynomial;
  polynomial = 1.0f + value * polynomial;
  const float series = -value * polynomial;
  const float direct = std::log(std::max(1.0f - value, 5.960464477539063e-08f));
  return value < 0.25f ? series : direct;
}

inline float gumbel_noise32(int64_t seed, int64_t position, uint32_t token_id) {
  const float uniform = murmur3_uniform32(seed, position, token_id);
  return -std::log(-log1p_neg_stable(uniform));
}

inline double gumbel_noise64(int64_t seed, int64_t position,
                             uint32_t token_id) {
  const double uniform = murmur3_uniform64(seed, position, token_id);
  return -std::log(-std::log(uniform));
}

}  // namespace cpu_sampling
