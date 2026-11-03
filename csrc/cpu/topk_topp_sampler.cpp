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
// Vendored from vLLM's CPU top-k/top-p kernels (micyan01/vllm#1 at 73f15af9a8,
// following vllm#43452), Copyright contributors to the vLLM project,
// Apache-2.0. Changed in four ways, each marked: the AArch64 paths are
// dropped; an x86 path is added (AVX2 intrinsics, so both the AVX512 and
// AVX2 builds vectorize; POWER keeps the scalar paths); cpu_arch_macros.h's
// s390x DEFINE_FAST_EXP is inlined below; and top-k keeps every token tied
// with the k-th value, as vLLM's PyTorch top-k does, where upstream keeps
// exactly k.
#include "cpu_types.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#if defined(__AVX2__)
  #include <immintrin.h>
#endif

#if defined(__s390x__)
// cpu_arch_macros.h's s390x DEFINE_FAST_EXP: FP32Vec16::exp() in
// cpu_types_vxe.hpp is a vectorised 5-term minimax polynomial.
  #define DEFINE_FAST_EXP                             \
    auto fast_exp = [&](const vec_op::FP32Vec16& vec) \
                        __attribute__((always_inline)) { return vec.exp(); };
#endif

namespace {
constexpr float kPadSentinel = -1e30f;      // logits below this are PAD
constexpr float kBoundaryTol = 1e-5f;       // tie tolerance for top-p cut
constexpr int kMeanStdSampleN = 8192;       // mean/std sample cap
constexpr int kBsearchMaxIters = 32;        // binary-search iteration cap
constexpr int kTernarySearchMaxIters = 18;  // matches Triton
constexpr float kKDuplicateTol = 1e-9f;     // tie tolerance for top-k (Triton)
}  // namespace

// Matches _NORMAL_CDF_TO_SIGMA_TABLE in topk_topp_triton.py (200 entries).
static const float NORMAL_CDF_TO_SIGMA_TABLE[200] = {
    3.656f,  3.650f,  3.650f, 3.650f, 3.626f, 3.626f, 3.626f, 3.514f, 3.514f,
    3.503f,  3.503f,  3.434f, 3.434f, 3.428f, 3.428f, 3.387f, 3.380f, 3.380f,
    3.376f,  3.373f,  3.373f, 3.356f, 3.354f, 3.354f, 3.291f, 3.249f, 3.234f,
    3.214f,  3.198f,  3.198f, 3.185f, 3.177f, 3.177f, 3.165f, 3.164f, 3.161f,
    3.138f,  3.120f,  3.115f, 3.113f, 3.093f, 3.066f, 3.054f, 3.043f, 3.037f,
    3.023f,  2.993f,  2.991f, 2.976f, 2.970f, 2.952f, 2.946f, 2.932f, 2.908f,
    2.902f,  2.895f,  2.886f, 2.874f, 2.861f, 2.844f, 2.836f, 2.810f, 2.801f,
    2.790f,  2.784f,  2.779f, 2.767f, 2.757f, 2.745f, 2.733f, 2.723f, 2.716f,
    2.693f,  2.678f,  2.671f, 2.656f, 2.649f, 2.629f, 2.611f, 2.595f, 2.592f,
    2.585f,  2.574f,  2.550f, 2.543f, 2.534f, 2.521f, 2.518f, 2.497f, 2.485f,
    2.468f,  2.450f,  2.441f, 2.430f, 2.412f, 2.402f, 2.389f, 2.383f, 2.377f,
    2.364f,  2.349f,  2.338f, 2.332f, 2.319f, 2.310f, 2.301f, 2.282f, 2.274f,
    2.266f,  2.250f,  2.242f, 2.236f, 2.226f, 2.215f, 2.207f, 2.196f, 2.179f,
    2.171f,  2.162f,  2.147f, 2.135f, 2.121f, 2.109f, 2.095f, 2.085f, 2.073f,
    2.063f,  2.045f,  2.030f, 2.016f, 2.003f, 1.992f, 1.983f, 1.972f, 1.960f,
    1.949f,  1.940f,  1.928f, 1.912f, 1.897f, 1.881f, 1.869f, 1.854f, 1.838f,
    1.824f,  1.807f,  1.792f, 1.779f, 1.764f, 1.751f, 1.739f, 1.726f, 1.711f,
    1.697f,  1.685f,  1.668f, 1.652f, 1.636f, 1.622f, 1.603f, 1.585f, 1.568f,
    1.551f,  1.534f,  1.513f, 1.499f, 1.480f, 1.464f, 1.441f, 1.422f, 1.394f,
    1.373f,  1.347f,  1.320f, 1.296f, 1.270f, 1.246f, 1.219f, 1.190f, 1.163f,
    1.135f,  1.104f,  1.073f, 1.041f, 1.006f, 0.969f, 0.931f, 0.894f, 0.851f,
    0.806f,  0.757f,  0.702f, 0.643f, 0.574f, 0.498f, 0.405f, 0.288f, 0.134f,
    -0.110f, -3.813f,
};

// Matches _PERCENTILE_TO_STD_TABLE in topk_topp_triton.py (200 entries).
static const float PERCENTILE_TO_STD_TABLE[200] = {
    2.576f,  2.319f,  2.178f,  2.064f,  1.968f,  1.892f,  1.819f,  1.757f,
    1.708f,  1.659f,  1.616f,  1.568f,  1.526f,  1.492f,  1.456f,  1.420f,
    1.382f,  1.342f,  1.309f,  1.280f,  1.249f,  1.221f,  1.193f,  1.169f,
    1.145f,  1.121f,  1.095f,  1.073f,  1.050f,  1.030f,  1.008f,  0.987f,
    0.966f,  0.945f,  0.926f,  0.910f,  0.891f,  0.871f,  0.854f,  0.837f,
    0.819f,  0.803f,  0.784f,  0.767f,  0.753f,  0.734f,  0.719f,  0.702f,
    0.690f,  0.675f,  0.658f,  0.640f,  0.625f,  0.609f,  0.595f,  0.578f,
    0.564f,  0.550f,  0.537f,  0.521f,  0.509f,  0.495f,  0.481f,  0.466f,
    0.453f,  0.439f,  0.424f,  0.410f,  0.397f,  0.383f,  0.370f,  0.356f,
    0.343f,  0.330f,  0.316f,  0.302f,  0.289f,  0.274f,  0.261f,  0.247f,
    0.235f,  0.223f,  0.209f,  0.196f,  0.184f,  0.172f,  0.159f,  0.149f,
    0.137f,  0.124f,  0.112f,  0.100f,  0.086f,  0.074f,  0.062f,  0.050f,
    0.035f,  0.023f,  0.009f,  -0.003f, -0.015f, -0.027f, -0.039f, -0.052f,
    -0.063f, -0.074f, -0.085f, -0.097f, -0.109f, -0.122f, -0.134f, -0.147f,
    -0.158f, -0.171f, -0.184f, -0.196f, -0.210f, -0.223f, -0.235f, -0.248f,
    -0.261f, -0.275f, -0.289f, -0.302f, -0.317f, -0.328f, -0.341f, -0.353f,
    -0.368f, -0.382f, -0.396f, -0.410f, -0.426f, -0.439f, -0.452f, -0.465f,
    -0.480f, -0.493f, -0.507f, -0.521f, -0.537f, -0.551f, -0.568f, -0.582f,
    -0.597f, -0.614f, -0.628f, -0.643f, -0.658f, -0.673f, -0.691f, -0.706f,
    -0.721f, -0.738f, -0.754f, -0.769f, -0.789f, -0.808f, -0.824f, -0.838f,
    -0.857f, -0.877f, -0.893f, -0.912f, -0.929f, -0.947f, -0.965f, -0.983f,
    -1.003f, -1.027f, -1.050f, -1.070f, -1.092f, -1.117f, -1.139f, -1.162f,
    -1.189f, -1.216f, -1.241f, -1.272f, -1.300f, -1.330f, -1.367f, -1.404f,
    -1.441f, -1.485f, -1.523f, -1.564f, -1.607f, -1.658f, -1.710f, -1.778f,
    -1.832f, -1.901f, -1.978f, -2.068f, -2.174f, -2.325f, -2.577f, -3.813f,
};

namespace {

// FP64 accumulator for softmax sum over FP32Vec16 chunks.
// FP32 accumulation over 128K elements loses ~3 ULP; FP64 avoids this.
#if defined(__s390x__)
using vf32 = __vector float;
using vi32 = __vector signed int;

static inline int hsum_i32(vi32 v) { return v[0] + v[1] + v[2] + v[3]; }

static inline float hmin_f32(vf32 v) {
  return std::min(std::min(v[0], v[1]), std::min(v[2], v[3]));
}

struct Fp64Acc16 {
  __vector double acc[8] = {};
  inline void add(const vec_op::FP32Vec16& v) {
    for (int i = 0; i < 4; ++i) {
      const vf32 x = v.reg.val[i];
      // vec_doublee widens lanes 0 and 2; rotate by one lane for 1 and 3.
      acc[2 * i] += vec_doublee(x);
      acc[2 * i + 1] += vec_doublee(vec_sld(x, x, 4));
    }
  }
  inline double reduce() const {
    __vector double sum = acc[0];
    for (int i = 1; i < 8; ++i) sum += acc[i];
    return sum[0] + sum[1];
  }
};

// VXE exp() clamps its input, so exp(-inf) yields ~1e-38 instead of 0.
// Zero PAD lanes to match the scalar and NEON paths.
static inline vec_op::FP32Vec16 zero_pad_lanes(const vec_op::FP32Vec16& v,
                                               const vec_op::FP32Vec16& e) {
  const vf32 pad = vec_splats(kPadSentinel);
  const vf32 zero = vec_splats(0.0f);
  vec_op::f32x4x4_t out;
  for (int r = 0; r < 4; ++r)
    out.val[r] = vec_sel(zero, e.reg.val[r], vec_cmpgt(v.reg.val[r], pad));
  return vec_op::FP32Vec16(out);
}
#elif defined(__AVX2__)
// Spyre addition: an x86 path in AVX2 intrinsics, which the AVX512 build
// compiles too. No FMA: the AVX2 build is -mavx2 only.
static inline int count_mask(__m256 m) {
  return __builtin_popcount(_mm256_movemask_ps(m));
}

static inline float hmin256(__m256 v) {
  __m128 x = _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  x = _mm_min_ps(x, _mm_movehl_ps(x, x));
  x = _mm_min_ss(x, _mm_shuffle_ps(x, x, 1));
  return _mm_cvtss_f32(x);
}

static inline float hmax256(__m256 v) {
  __m128 x = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  x = _mm_max_ps(x, _mm_movehl_ps(x, x));
  x = _mm_max_ss(x, _mm_shuffle_ps(x, x, 1));
  return _mm_cvtss_f32(x);
}

struct Fp64Acc8 {
  __m256d lo = _mm256_setzero_pd(), hi = _mm256_setzero_pd();
  inline void add(__m256 v) {
    lo = _mm256_add_pd(lo, _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
    hi = _mm256_add_pd(hi, _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
  }
  inline double reduce() const {
    alignas(32) double t[4];
    _mm256_store_pd(t, _mm256_add_pd(lo, hi));
    return (t[0] + t[1]) + (t[2] + t[3]);
  }
};

// expf to about 1 ULP: Cephes' range reduction and degree-6 polynomial. Inputs
// below -87.3 (PAD lanes included) clamp to ~1e-38; callers zero PAD lanes.
static inline __m256 exp256(__m256 x) {
  x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
  x = _mm256_max_ps(x, _mm256_set1_ps(-87.3365478515625f));
  __m256 fx =
      _mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                    _mm256_set1_ps(0.5f));
  fx = _mm256_floor_ps(fx);
  x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f)));
  x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f)));
  const __m256 z = _mm256_mul_ps(x, x);
  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y, z), x),
                    _mm256_set1_ps(1.0f));
  const __m256i pow2n = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(0x7f)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(pow2n));
}

// exp(x - base), zero on PAD lanes, as the scalar path's expf(-inf) is. A
// chunk with no finite lane skips the polynomial: after top-k, most are.
static inline __m256 exp_shifted_256(__m256 x, __m256 base) {
  const __m256 finite =
      _mm256_cmp_ps(x, _mm256_set1_ps(kPadSentinel), _CMP_GT_OQ);
  if (!_mm256_movemask_ps(finite)) return _mm256_setzero_ps();
  return _mm256_and_ps(exp256(_mm256_sub_ps(x, base)), finite);
}
#endif

__attribute__((always_inline)) static inline double sum_gt_to_double(
    const float* buf, int n, float threshold) {
  double s = 0.0;
  int j = 0;
#if defined(__s390x__)
  Fp64Acc16 acc;
  const vf32 thr = vec_splats(threshold);
  const vf32 zero = vec_splats(0.0f);
  for (; j + 16 <= n; j += 16) {
    vec_op::f32x4x4_t masked;
    for (int r = 0; r < 4; ++r) {
      const vf32 v = vec_xl(0, buf + j + 4 * r);
      masked.val[r] = vec_sel(zero, v, vec_cmpgt(v, thr));
    }
    acc.add(vec_op::FP32Vec16(masked));
  }
  s = acc.reduce();
#elif defined(__AVX2__)
  Fp64Acc8 acc;
  const __m256 thr = _mm256_set1_ps(threshold);
  for (; j + 8 <= n; j += 8) {
    const __m256 v = _mm256_loadu_ps(buf + j);
    acc.add(_mm256_and_ps(v, _mm256_cmp_ps(v, thr, _CMP_GT_OQ)));
  }
  s = acc.reduce();
#endif
  for (; j < n; ++j)
    if (buf[j] > threshold) s += (double)buf[j];
  return s;
}

__attribute__((always_inline)) static inline int count_within_tol(
    const float* row, int V, float center, float tol) {
  int n = 0;
  int i = 0;
#if defined(__s390x__)
  const vf32 c = vec_splats(center);
  const vf32 t = vec_splats(tol);
  // Mask lanes are all-ones (-1), so subtracting counts matches.
  vi32 cnt = vec_splats(0);
  for (; i + 16 <= V; i += 16) {
    for (int r = 0; r < 4; ++r) {
      const vf32 v = vec_xl(0, row + i + 4 * r);
      cnt -= (vi32)vec_cmplt(vec_abs(v - c), t);
    }
  }
  n += hsum_i32(cnt);
#elif defined(__AVX2__)
  const __m256 c = _mm256_set1_ps(center);
  const __m256 t = _mm256_set1_ps(tol);
  const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  for (; i + 8 <= V; i += 8) {
    const __m256 d =
        _mm256_and_ps(_mm256_sub_ps(_mm256_loadu_ps(row + i), c), abs_mask);
    n += count_mask(_mm256_cmp_ps(d, t, _CMP_LT_OQ));
  }
#endif
  for (; i < V; ++i)
    if (fabsf(row[i] - center) < tol) ++n;
  return n;
}

__attribute__((always_inline)) static inline void mask_write_below(
    float* row, int V, float threshold, float fill) {
  int i = 0;
#if defined(__s390x__)
  const vf32 thr = vec_splats(threshold);
  const vf32 fv = vec_splats(fill);
  for (; i + 16 <= V; i += 16) {
    for (int r = 0; r < 4; ++r) {
      const vf32 v = vec_xl(0, row + i + 4 * r);
      vec_xst(vec_sel(fv, v, vec_cmpgt(v, thr)), 0, row + i + 4 * r);
    }
  }
#elif defined(__AVX2__)
  const __m256 thr = _mm256_set1_ps(threshold);
  const __m256 fv = _mm256_set1_ps(fill);
  for (; i + 8 <= V; i += 8) {
    const __m256 v = _mm256_loadu_ps(row + i);
    _mm256_storeu_ps(
        row + i, _mm256_blendv_ps(fv, v, _mm256_cmp_ps(v, thr, _CMP_GT_OQ)));
  }
#endif
  for (; i < V; ++i)
    if (!(row[i] > threshold)) row[i] = fill;
}

// Vectorised max/min/finite-count over row[0..V). PAD lanes (<= pad_sentinel)
// are blended to +inf so they don't corrupt reduce_min. Returns SIMD tail
// start in *tail_i_out; caller handles remaining elements and clamps min<=max.
__attribute__((always_inline)) static inline void vec_max_min_with_pad_blend(
    const float* row, int V, float pad_sentinel, float* max_out, float* min_out,
    int* n_finite_out, int* tail_i_out) {
  int i = 0;
  float max_l = -1e38f, min_l = 1e38f;
  int n_finite = 0;
#if defined(__s390x__)
  {
    const vf32 pad = vec_splats(pad_sentinel);
    const vf32 pos_inf = vec_splats(std::numeric_limits<float>::infinity());
    vec_op::FP32Vec16 maxv(row[0]);
    vec_op::FP32Vec16 minv(std::numeric_limits<float>::infinity());
    vi32 cnt = vec_splats(0);
    for (; i + 16 <= V; i += 16) {
      vec_op::FP32Vec16 v16(row + i);
      maxv = maxv.max(v16);
      vec_op::f32x4x4_t safe_raw;
      for (int r = 0; r < 4; ++r) {
        const vf32 x = v16.reg.val[r];
        const auto fin = vec_cmpgt(x, pad);
        cnt -= (vi32)fin;
        safe_raw.val[r] = vec_sel(pos_inf, x, fin);
      }
      minv = minv.min(vec_op::FP32Vec16(safe_raw));
    }
    n_finite = hsum_i32(cnt);
    max_l = maxv.reduce_max();
    min_l = minv.reduce_min();
  }
#elif defined(__AVX2__)
  {
    const __m256 pad = _mm256_set1_ps(pad_sentinel);
    const __m256 pos_inf =
        _mm256_set1_ps(std::numeric_limits<float>::infinity());
    __m256 maxv = _mm256_set1_ps(row[0]);
    __m256 minv = pos_inf;
    for (; i + 8 <= V; i += 8) {
      const __m256 x = _mm256_loadu_ps(row + i);
      maxv = _mm256_max_ps(maxv, x);
      const __m256 fin = _mm256_cmp_ps(x, pad, _CMP_GT_OQ);
      n_finite += count_mask(fin);
      minv = _mm256_min_ps(minv, _mm256_blendv_ps(pos_inf, x, fin));
    }
    if (i > 0) {
      max_l = hmax256(maxv);
      min_l = hmin256(minv);
    }
  }
#endif
  *max_out = max_l;
  *min_out = min_l;
  *n_finite_out = n_finite;
  *tail_i_out = i;
}

__attribute__((always_inline)) static inline int gather_outliers(
    const float* buf, int n, float threshold, float* out, double* sum_out) {
  int m = 0;
  double s = 0.0;
  for (int j = 0; j < n; ++j) {
    float v = buf[j];
    if (v > threshold) {
      out[m++] = v;
      if (sum_out) s += (double)v;
    }
  }
  if (sum_out) *sum_out = s;
  return m;
}

static inline void compute_mean_std(const float* row, int V, float* avg_out,
                                    float* std_out) {
  float sum_s = 0.f, sum_sq = 0.f;
  int nf = 0;
  for (int i = 0; i < V && nf < kMeanStdSampleN; ++i) {
    float v = row[i];
    if (v > kPadSentinel) {
      sum_s += v;
      sum_sq += v * v;
      ++nf;
    }
  }
  float avg = nf > 0 ? sum_s / nf : 0.f;
  // E[X²] - (E[X])² is numerically unstable for large logit magnitudes, but
  // this is only a heuristic pivot initializer; the ternary/binary search
  // converges correctly regardless of pivot quality.
  float var = nf > 0 ? sum_sq / nf - avg * avg : 1.f;
  *avg_out = avg;
  *std_out = sqrtf(var > 0.f ? var : 0.f);
}

// Apply boundary-tie resolution in row order. Keeps all v > pivot, plus up to
// n_keep_at_boundary values within tol of dup_value. Rare path: called only
// when binary/ternary search lands exactly on a duplicate boundary value.
static inline void apply_boundary_tie_loop(float* row, int V, float pivot,
                                           float dup_value, float tol,
                                           int n_keep_at_boundary) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  int kept_boundary = 0;
  for (int i = 0; i < V; ++i) {
    float v = row[i];
    bool keep = v > pivot;
    if (!keep && fabsf(v - dup_value) < tol &&
        kept_boundary < n_keep_at_boundary) {
      keep = true;
      ++kept_boundary;
    }
    if (!keep) row[i] = neg_inf;
  }
}

}  // namespace

// Pre-computed per-row statistics shared between top-k and top-p passes.
struct RowStats {
  float avg, std_v, max_l, min_l;
  int n_finite;
};

// Compute RowStats for row[0..V). min_l is clamped to max_l if no finite min
// was found (degenerate all-PAD row).
static inline void compute_row_stats(const float* row, int V, RowStats* st) {
  compute_mean_std(row, V, &st->avg, &st->std_v);
  int tail_i;
  vec_max_min_with_pad_blend(row, V, kPadSentinel, &st->max_l, &st->min_l,
                             &st->n_finite, &tail_i);
  for (int i = tail_i; i < V; ++i) {
    float v = row[i];
    if (v > st->max_l) st->max_l = v;
    if (v > kPadSentinel) {
      ++st->n_finite;
      if (v < st->min_l) st->min_l = v;
    }
  }
  if (st->min_l > st->max_l) st->min_l = st->max_l;
}

// Entry-point validation helpers shared by all three public functions.
static void check_logits(const torch::Tensor& logits) {
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(logits.is_cpu(), "logits must be CPU");
  TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
}

static void check_batch_float(const torch::Tensor& t,
                              const torch::Tensor& logits, const char* name) {
  TORCH_CHECK(t.is_cpu(), name, " must be CPU");
  TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(t.dim() == 1 && t.size(0) == logits.size(0), name,
              " must be 1D with size == batch size");
}

static void check_batch_int(const torch::Tensor& t, const torch::Tensor& logits,
                            const char* name) {
  TORCH_CHECK(t.is_cpu(), name, " must be CPU");
  TORCH_CHECK(t.dtype() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(t.dim() == 1 && t.size(0) == logits.size(0), name,
              " must be 1D with size == batch size");
}

static float binary_search_buffer(const float* buf, int n, float p_val,
                                  float lo, float hi, double* sum_above_out) {
  double s_hi = 0.0;
  for (int iter = 0; iter < kBsearchMaxIters; ++iter) {
    float mid = lo + (hi - lo) * 0.5f;
    if (mid == lo || mid == hi) break;
    double s = sum_gt_to_double(buf, n, mid);
    if (s >= (double)p_val) {
      lo = mid;
    } else {
      hi = mid;
      s_hi = s;
    }
  }
  *sum_above_out = s_hi;
  return hi;
}

static void scan_pivot_stats(const float* buf, int n, float pivot, float tol,
                             int* num_above, float* min_above,
                             int* num_at_min) {
  float ma = std::numeric_limits<float>::infinity();
  int na = 0;
  int i = 0;
#if defined(__s390x__)
  {
    const vf32 vpivot = vec_splats(pivot);
    const vf32 pos_inf = vec_splats(std::numeric_limits<float>::infinity());
    vf32 vmin = pos_inf;
    vi32 cnt = vec_splats(0);
    for (; i + 16 <= n; i += 16) {
      for (int r = 0; r < 4; ++r) {
        const vf32 values = vec_xl(0, buf + i + 4 * r);
        const auto above = vec_cmpgt(values, vpivot);
        cnt -= (vi32)above;
        vmin = vec_min(vmin, vec_sel(pos_inf, values, above));
      }
    }
    na = hsum_i32(cnt);
    ma = hmin_f32(vmin);
  }
#elif defined(__AVX2__)
  {
    const __m256 vpivot = _mm256_set1_ps(pivot);
    const __m256 pos_inf =
        _mm256_set1_ps(std::numeric_limits<float>::infinity());
    __m256 vmin = pos_inf;
    for (; i + 8 <= n; i += 8) {
      const __m256 values = _mm256_loadu_ps(buf + i);
      const __m256 above = _mm256_cmp_ps(values, vpivot, _CMP_GT_OQ);
      na += count_mask(above);
      vmin = _mm256_min_ps(vmin, _mm256_blendv_ps(pos_inf, values, above));
    }
    ma = hmin256(vmin);
  }
#endif
  for (; i < n; ++i) {
    if (buf[i] > pivot) {
      ++na;
      if (buf[i] < ma) ma = buf[i];
    }
  }
  *num_above = na;
  *min_above = ma;
  // count_within_tol runs over the full buf; elements below pivot are all
  // < min_above, so only ULP-equal values fall within kKDuplicateTol and
  // are legitimately counted as boundary duplicates.
  *num_at_min = count_within_tol(buf, n, ma, tol);
}

static float ternary_search_topk(const float* buf, int n, int k, float lo,
                                 float hi, float* dup_value_out,
                                 int* num_dup_out, int* num_above_out) {
  float final_pivot = (lo + hi) * 0.5f;
  float dup_value = std::numeric_limits<float>::infinity();
  int num_dup = 0, num_above = 0;
  for (int iter = 0; iter < kTernarySearchMaxIters; ++iter) {
    float p0 = lo + (hi - lo) * (1.0f / 3.0f);
    float p1 = lo + (hi - lo) * (2.0f / 3.0f);
    int na0, na1, nd0, nd1;
    float mn0, mn1;
    scan_pivot_stats(buf, n, p0, kKDuplicateTol, &na0, &mn0, &nd0);
    scan_pivot_stats(buf, n, p1, kKDuplicateTol, &na1, &mn1, &nd1);

    bool found0 = (na0 >= k) && (na0 - nd0 < k);
    bool found1 = (na1 >= k) && (na1 - nd1 < k);
    if (found1) {
      final_pivot = p1;
      dup_value = mn1;
      num_dup = nd1;
      num_above = na1;
      break;
    }
    if (found0) {
      final_pivot = p0;
      dup_value = mn0;
      num_dup = nd0;
      num_above = na0;
      break;
    }
    if (na1 > k)
      lo = p1;
    else if (na0 > k)
      lo = p0;
    if (na0 < k)
      hi = p0;
    else if (na1 < k)
      hi = p1;

    if (hi - lo < kKDuplicateTol) {
      final_pivot = (lo + hi) * 0.5f;
      scan_pivot_stats(buf, n, final_pivot, kKDuplicateTol, &num_above,
                       &dup_value, &num_dup);
      break;
    }
  }
  *dup_value_out = dup_value;
  *num_dup_out = num_dup;
  *num_above_out = num_above;
  return final_pivot;
}

// Core top-k logic. Receives pre-computed RowStats; does not re-scan the row
// for mean/std or max/min.
static void _top_k_row_core(float* __restrict__ row, int V, int k_val,
                            const RowStats& st,
                            float* __restrict__ outlier_buf) {
  int tidx = (int)((double)k_val / V * 200);
  if (tidx < 0) tidx = 0;
  if (tidx > 199) tidx = 199;
  float sigma = PERCENTILE_TO_STD_TABLE[tidx];
  sigma = sigma + fabsf(sigma) * -0.15f;
  float outlier_logit = st.avg + st.std_v * sigma;

  int n_outliers = gather_outliers(row, V, outlier_logit, outlier_buf, nullptr);

  if (st.n_finite <= k_val) return;

  const float neg_inf = -std::numeric_limits<float>::infinity();

  // Degenerate-flat: every finite value ties with the k-th, and vLLM's top-k
  // keeps ties (Spyre change; upstream kept the first k).
  if (st.max_l - st.min_l < kKDuplicateTol) return;

  float final_pivot, dup_value;
  int n_dup, n_above;
  if (n_outliers > k_val) {
    final_pivot =
        ternary_search_topk(outlier_buf, n_outliers, k_val, outlier_logit,
                            st.max_l, &dup_value, &n_dup, &n_above);
  } else {
    final_pivot = ternary_search_topk(row, V, k_val, st.min_l, st.max_l,
                                      &dup_value, &n_dup, &n_above);
  }

  int n_keep_at_boundary = n_dup - (n_above - k_val);
  if (n_keep_at_boundary < 0) n_keep_at_boundary = 0;
  // Spyre change: when the k-th value is among the duplicates, keep all of
  // them, as vLLM's top-k (logits >= the k-th value) does; upstream keeps only
  // enough to reach exactly k.
  if (n_keep_at_boundary > 0) n_keep_at_boundary = n_dup;

  if (n_keep_at_boundary == 0) {
    mask_write_below(row, V, final_pivot, neg_inf);
  } else {
    apply_boundary_tie_loop(row, V, dup_value, dup_value, kKDuplicateTol,
                            n_keep_at_boundary);
  }
}

// Core top-p logic. Receives pre-computed RowStats; does not re-scan the row.
static void _top_p_row_core(float* __restrict__ row, int V, float p_val,
                            const RowStats& st, float* __restrict__ scratch,
                            float* __restrict__ outlier_buf) {
  if (p_val >= 1.0f) return;

  const float neg_inf = -std::numeric_limits<float>::infinity();
  if (p_val <= 0.0f) {
    mask_write_below(row, V, st.max_l - kBoundaryTol, neg_inf);
    return;
  }

  int tidx = (int)(p_val * 200);
  if (tidx < 0) tidx = 0;
  if (tidx > 199) tidx = 199;
  float sigma = NORMAL_CDF_TO_SIGMA_TABLE[tidx];
  sigma = sigma + fabsf(sigma) * -0.25f;
  float outlier_logit = st.avg + st.std_v * sigma;

#if defined(__s390x__)
  DEFINE_FAST_EXP
  vec_op::FP32Vec16 base16(st.max_l);
#elif defined(__AVX2__)
  const __m256 base8 = _mm256_set1_ps(st.max_l);
#endif
  double sum_exp_d = 0.0;
  {
    int i = 0;
#if defined(__s390x__)
    Fp64Acc16 acc;
    for (; i + 16 <= V; i += 16) {
      vec_op::FP32Vec16 v16(row + i);
      vec_op::FP32Vec16 e16 = fast_exp(v16 - base16);
  #if defined(__s390x__)
      e16 = zero_pad_lanes(v16, e16);
  #endif
      acc.add(e16);
    }
    sum_exp_d += acc.reduce();
#elif defined(__AVX2__)
    Fp64Acc8 acc;
    for (; i + 8 <= V; i += 8)
      acc.add(exp_shifted_256(_mm256_loadu_ps(row + i), base8));
    sum_exp_d += acc.reduce();
#endif
    for (; i < V; ++i) sum_exp_d += (double)expf(row[i] - st.max_l);
  }
  float sum_exp = (float)sum_exp_d;

  float outlier_prob = expf(outlier_logit - st.max_l) / sum_exp;
  float max_prob = 1.0f / sum_exp;
  float min_prob = expf(st.min_l - st.max_l) / sum_exp;

  double sum_out_d = 0.0;
  {
    const float inv_sum = 1.0f / sum_exp;
    int i = 0;
#if defined(__s390x__)
    vec_op::FP32Vec16 isum16(inv_sum);
    for (; i + 16 <= V; i += 16) {
      vec_op::FP32Vec16 v16(row + i);
      vec_op::FP32Vec16 p16 = fast_exp(v16 - base16) * isum16;
  #if defined(__s390x__)
      p16 = zero_pad_lanes(v16, p16);
  #endif
      p16.save(scratch + i);
    }
#elif defined(__AVX2__)
    const __m256 isum8 = _mm256_set1_ps(inv_sum);
    for (; i + 8 <= V; i += 8)
      _mm256_storeu_ps(
          scratch + i,
          _mm256_mul_ps(exp_shifted_256(_mm256_loadu_ps(row + i), base8),
                        isum8));
#endif
    for (; i < V; ++i) scratch[i] = expf(row[i] - st.max_l) * inv_sum;
  }
  int n_out =
      gather_outliers(scratch, V, outlier_prob, outlier_buf, &sum_out_d);

  float* sbuf;
  int sn;
  float buf_lo, buf_hi;
  if (sum_out_d >= (double)p_val) {
    sbuf = outlier_buf;
    sn = n_out;
    buf_hi = max_prob;
    buf_lo = outlier_prob;
  } else {
    sbuf = scratch;
    sn = V;
    buf_hi = max_prob;
    buf_lo = min_prob;
  }

  double sum_above;
  float boundary_prob =
      binary_search_buffer(sbuf, sn, p_val, buf_lo, buf_hi, &sum_above);

  float dup_logit = logf(boundary_prob * sum_exp) + st.max_l;

  // NaN or overflow above max_l means binary search degenerated; skip masking.
  if (!(dup_logit <= st.max_l)) {
    return;
  }

  int n_at_boundary = count_within_tol(row, V, dup_logit, kBoundaryTol);

  int n_keep_at_boundary = 0;
  if (n_at_boundary > 0) {
    double remaining = (double)p_val - sum_above;
    if (remaining > 0.0 && (double)boundary_prob > 0.0) {
      double needed_d = ceil(remaining / (double)boundary_prob);
      if (needed_d >= (double)n_at_boundary)
        n_keep_at_boundary = n_at_boundary;
      else if (needed_d > 0.0)
        n_keep_at_boundary = (int)needed_d;
    }
  }

  if (n_keep_at_boundary == 0) {
    mask_write_below(row, V, dup_logit, neg_inf);
  } else {
    apply_boundary_tie_loop(row, V, dup_logit, dup_logit, kBoundaryTol,
                            n_keep_at_boundary);
  }
}

static void top_k_row(float* __restrict__ row, int V, int k_val,
                      float* __restrict__ outlier_buf) {
  RowStats st;
  compute_row_stats(row, V, &st);
  _top_k_row_core(row, V, k_val, st, outlier_buf);
}

static void top_p_row(float* __restrict__ row, int V, float p_val,
                      float* __restrict__ scratch,
                      float* __restrict__ outlier_buf) {
  RowStats st;
  compute_row_stats(row, V, &st);
  _top_p_row_core(row, V, p_val, st, scratch, outlier_buf);
}

// Fused top-k + top-p: shared mean/std and max/min passes save ~2 row reads
// vs sequential top_k_row + top_p_row.
static void top_k_p_row(float* __restrict__ row, int V, int k_val, float p_val,
                        float* __restrict__ scratch,
                        float* __restrict__ outlier_buf) {
  RowStats st;
  compute_row_stats(row, V, &st);
  // k_val > 0 && k_val < V is pre-checked by the caller.
  _top_k_row_core(row, V, k_val, st, outlier_buf);
  // Top-k filtered lanes hold neg_inf; exp(-inf)=0 so they don't affect sum.
  _top_p_row_core(row, V, p_val, st, scratch, outlier_buf);
}

void cpu_topp_sampling(torch::Tensor& logits, const torch::Tensor& p) {
  check_logits(logits);
  check_batch_float(p, logits, "p");

  const int64_t B = logits.size(0);
  const int64_t V = logits.size(1);
  const int V_int = static_cast<int>(V);
  if (V == 0) return;
  float* lp = logits.data_ptr<float>();
  const float* pp = p.data_ptr<float>();

  // thread_local avoids reallocation across calls; resize skips zero-fill.
#pragma omp parallel
  {
    thread_local std::vector<float> scratch_tls;
    thread_local std::vector<float> outlier_tls;
    if ((int64_t)scratch_tls.size() < V) scratch_tls.resize(V);
    if ((int64_t)outlier_tls.size() < V) outlier_tls.resize(V);
#pragma omp for schedule(dynamic, 1)
    for (int64_t b = 0; b < B; ++b) {
      top_p_row(lp + b * V, V_int, pp[b], scratch_tls.data(),
                outlier_tls.data());
    }
  }
}

void cpu_topk_sampling(torch::Tensor& logits, const torch::Tensor& k) {
  check_logits(logits);
  check_batch_int(k, logits, "k");

  const int64_t B = logits.size(0);
  const int64_t V = logits.size(1);
  const int V_int = static_cast<int>(V);
  if (V == 0) return;
  float* lp = logits.data_ptr<float>();
  const int* kp = k.data_ptr<int>();

#pragma omp parallel
  {
    thread_local std::vector<float> outlier_tls;
    if ((int64_t)outlier_tls.size() < V) outlier_tls.resize(V);
#pragma omp for schedule(dynamic, 1)
    for (int64_t b = 0; b < B; ++b) {
      int kv = kp[b];
      if (kv > 0 && kv < V_int)
        top_k_row(lp + b * V, V_int, kv, outlier_tls.data());
    }
  }
}

void cpu_topk_topp_sampling(torch::Tensor& logits, const torch::Tensor& k,
                            const torch::Tensor& p) {
  check_logits(logits);
  check_batch_int(k, logits, "k");
  check_batch_float(p, logits, "p");

  const int64_t B = logits.size(0);
  const int64_t V = logits.size(1);
  const int V_int = static_cast<int>(V);
  if (V == 0) return;
  float* lp = logits.data_ptr<float>();
  const int* kp = k.data_ptr<int>();
  const float* pp = p.data_ptr<float>();

#pragma omp parallel
  {
    thread_local std::vector<float> scratch_tls;
    thread_local std::vector<float> outlier_tls;
    if ((int64_t)scratch_tls.size() < V) scratch_tls.resize(V);
    if ((int64_t)outlier_tls.size() < V) outlier_tls.resize(V);
#pragma omp for schedule(dynamic, 1)
    for (int64_t b = 0; b < B; ++b) {
      float* row = lp + b * V;
      int kv = kp[b];
      float pv = pp[b];
      if (kv > 0 && kv < V_int) {
        top_k_p_row(row, V_int, kv, pv, scratch_tls.data(), outlier_tls.data());
      } else {
        top_p_row(row, V_int, pv, scratch_tls.data(), outlier_tls.data());
      }
    }
  }
}
