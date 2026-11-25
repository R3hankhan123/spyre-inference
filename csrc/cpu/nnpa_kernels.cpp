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

// Host matmuls on the IBM Z Integrated Accelerator for AI (NNPA) through zDNN.
// s390x only. libzdnn is dlopen'ed rather than linked, so _C still imports on a host
// without it, and a host without NNPA (z15) reports a zero size and keeps every
// op on the card.

#include <dlfcn.h>
#include <zdnn.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>

#include <torch/all.h>
#include <torch/custom_class.h>
#include <torch/library.h>

namespace {

struct Zdnn {
  decltype(&zdnn_is_nnpa_installed) is_nnpa_installed;
  decltype(&zdnn_get_nnpa_max_dim_idx_size) max_dim_idx_size;
  decltype(&zdnn_init_pre_transformed_desc) init_pre_transformed_desc;
  decltype(&zdnn_generate_transformed_desc) generate_transformed_desc;
  decltype(&zdnn_init_ztensor_with_malloc) init_ztensor_with_malloc;
  decltype(&zdnn_transform_ztensor) transform_ztensor;
  decltype(&zdnn_transform_origtensor) transform_origtensor;
  decltype(&zdnn_free_ztensor_buffer) free_ztensor_buffer;
  decltype(&zdnn_matmul_op) matmul_op;
  decltype(&zdnn_get_status_message) get_status_message;
};

// Null when libzdnn is missing or the CPU has no NNPA.
const Zdnn* load_zdnn() {
  static const Zdnn* const lib = []() -> const Zdnn* {
    void* handle = dlopen("libzdnn.so.0", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) handle = dlopen("libzdnn.so", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) return nullptr;
    static Zdnn z;
    bool resolved = true;
    auto bind = [&](auto& fn, const char* name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
          dlsym(handle, name));
      resolved = resolved && fn != nullptr;
    };
    bind(z.is_nnpa_installed, "zdnn_is_nnpa_installed");
    bind(z.max_dim_idx_size, "zdnn_get_nnpa_max_dim_idx_size");
    bind(z.init_pre_transformed_desc, "zdnn_init_pre_transformed_desc");
    bind(z.generate_transformed_desc, "zdnn_generate_transformed_desc");
    bind(z.init_ztensor_with_malloc, "zdnn_init_ztensor_with_malloc");
    bind(z.transform_ztensor, "zdnn_transform_ztensor");
    bind(z.transform_origtensor, "zdnn_transform_origtensor");
    bind(z.free_ztensor_buffer, "zdnn_free_ztensor_buffer");
    bind(z.matmul_op, "zdnn_matmul_op");
    bind(z.get_status_message, "zdnn_get_status_message");
    return resolved && z.is_nnpa_installed() ? &z : nullptr;
  }();
  return lib;
}

const Zdnn& zdnn() {
  const Zdnn* z = load_zdnn();
  TORCH_CHECK(z != nullptr, "NNPA is not available on this host");
  return *z;
}

void check(zdnn_status status, const char* what) {
  TORCH_CHECK(status == ZDNN_OK, what,
              " failed: ", zdnn().get_status_message(status));
}

// A zDNN tensor in its stick layout. Pinned in place: the ztensor points at the
// descriptors next to it.
class ZTensor {
 public:
  ZTensor(zdnn_data_types type, uint32_t rows, uint32_t cols) {
    zdnn().init_pre_transformed_desc(ZDNN_2D, type, &pre_, rows, cols);
    init();
  }
  ZTensor(zdnn_data_types type, uint32_t len) {
    zdnn().init_pre_transformed_desc(ZDNN_1D, type, &pre_, len);
    init();
  }
  ZTensor(const ZTensor&) = delete;
  ZTensor& operator=(const ZTensor&) = delete;
  ~ZTensor() { zdnn().free_ztensor_buffer(&zt_); }

  void load(const void* host) {
    check(zdnn().transform_ztensor(&zt_, host), "zdnn_transform_ztensor");
  }
  void store(void* host) const {
    check(zdnn().transform_origtensor(&zt_, host), "zdnn_transform_origtensor");
  }
  zdnn_ztensor* get() { return &zt_; }

 private:
  void init() {
    check(zdnn().generate_transformed_desc(&pre_, &tfrmd_),
          "zdnn_generate_transformed_desc");
    check(zdnn().init_ztensor_with_malloc(&pre_, &tfrmd_, &zt_),
          "zdnn_init_ztensor_with_malloc");
  }

  zdnn_tensor_desc pre_{}, tfrmd_{};
  zdnn_ztensor zt_{};
};

int64_t nnpa_max_dim_idx_size() {
  const Zdnn* z = load_zdnn();
  return z == nullptr ? 0 : z->max_dim_idx_size();
}

// Gemma 4's per-layer projection: `x @ weight_t` on NNPA (the projection scale
// already folded into the weight, converted to the stick layout once), then the
// RMSNorm over each layer's slice and the combine with the looked-up rows in one
// fp32 pass per chunk on the host. fp16 in and out.
struct NnpaPerLayerProjection : torch::CustomClassHolder {
  NnpaPerLayerProjection(torch::Tensor weight_t, torch::Tensor norm_weight,
                         double eps, double embed_scale, double input_scale)
      : in_(weight_t.size(0)),
        out_(weight_t.size(1)),
        width_(norm_weight.numel()),
        norm_weight_(norm_weight.to(torch::kFloat).contiguous()),
        eps_(static_cast<float>(eps)),
        embed_scale_(static_cast<float>(embed_scale)),
        input_scale_(static_cast<float>(input_scale)) {
    TORCH_CHECK(weight_t.dim() == 2 && weight_t.scalar_type() == at::kHalf &&
                    weight_t.device().is_cpu(),
                "NnpaPerLayerProjection takes a 2-D fp16 host weight [in, out]");
    TORCH_CHECK(width_ > 0 && out_ % width_ == 0, "projection width ", out_,
                " is not a multiple of the norm width ", width_);
    const int64_t limit = nnpa_max_dim_idx_size();
    TORCH_CHECK(in_ <= limit && out_ <= limit, "NnpaPerLayerProjection weight ",
                weight_t.sizes(), " exceeds the NNPA dimension limit ", limit);
    weight_ = std::make_unique<ZTensor>(FP16, in_, out_);
    weight_->load(weight_t.contiguous().data_ptr());
    bias_ = std::make_unique<ZTensor>(FP32, out_);
    bias_->load(torch::zeros({out_}, torch::kFloat).data_ptr());
  }

  torch::Tensor project(torch::Tensor x, const std::optional<torch::Tensor>& rows) {
    TORCH_CHECK(x.dim() == 2 && x.size(1) == in_ &&
                    x.scalar_type() == at::kHalf && x.device().is_cpu(),
                "takes an fp16 host input [tokens, ", in_, "]");
    x = x.contiguous();
    const int64_t tokens = x.size(0);
    const at::Half* lookup = nullptr;
    torch::Tensor rows_c;
    if (rows.has_value()) {
      rows_c = rows->contiguous();
      TORCH_CHECK(rows_c.scalar_type() == at::kHalf && rows_c.device().is_cpu() &&
                      rows_c.numel() == tokens * out_,
                  "per-layer rows must be fp16 host [tokens, ", out_, "]");
      lookup = rows_c.data_ptr<at::Half>();
    }
    auto out = torch::empty({tokens, out_}, torch::kHalf);
    const int64_t chunk = std::min<int64_t>(nnpa_max_dim_idx_size(), tokens);
    auto proj = torch::empty({chunk, out_}, torch::kFloat);
    const auto* src = x.data_ptr<at::Half>();
    for (int64_t start = 0; start < tokens; start += chunk) {
      const int64_t n = std::min(chunk, tokens - start);
      {
        ZTensor a(FP16, static_cast<uint32_t>(n), in_);
        ZTensor y(FP32, static_cast<uint32_t>(n), out_);
        a.load(src + start * in_);
        check(zdnn().matmul_op(a.get(), weight_->get(), bias_->get(),
                               MATMUL_OP_ADDITION, y.get()),
              "zdnn_matmul_op");
        y.store(proj.data_ptr<float>());
      }
      combine(proj.data_ptr<float>(), lookup ? lookup + start * out_ : nullptr,
              out.data_ptr<at::Half>() + start * out_, n * (out_ / width_));
    }
    return out;
  }

  // Upstream's norm(proj) and (norm(proj) + rows * embed_scale) * input_scale, one
  // layer slice of one token per iteration.
  void combine(const float* proj, const at::Half* lookup, at::Half* out,
               int64_t slices) const {
    const int64_t w = width_;
    const float* weight = norm_weight_.data_ptr<float>();
#pragma omp parallel for schedule(static) if (slices > 64)
    for (int64_t i = 0; i < slices; ++i) {
      const float* p = proj + i * w;
      float sum = 0.f;
      for (int64_t d = 0; d < w; ++d) sum += p[d] * p[d];
      const float r = 1.f / std::sqrt(sum / static_cast<float>(w) + eps_);
      at::Half* o = out + i * w;
      if (lookup == nullptr) {
        for (int64_t d = 0; d < w; ++d) o[d] = p[d] * r * weight[d];
      } else {
        const at::Half* q = lookup + i * w;
        for (int64_t d = 0; d < w; ++d)
          o[d] = (p[d] * r * weight[d] + static_cast<float>(q[d]) * embed_scale_) *
                 input_scale_;
      }
    }
  }

  int64_t in_, out_, width_;
  torch::Tensor norm_weight_;
  float eps_, embed_scale_, input_scale_;
  std::unique_ptr<ZTensor> weight_, bias_;
};

// An op rather than a method, so the profiler records it.
torch::Tensor nnpa_per_layer_project(
    const c10::intrusive_ptr<NnpaPerLayerProjection>& projection, torch::Tensor x,
    const std::optional<torch::Tensor>& rows) {
  return projection->project(std::move(x), rows);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_spyre_C, ops) {
  // 0 when this host cannot run NNPA ops: no libzdnn, or no NNPA (pre-z16).
  ops.def("nnpa_max_dim_idx_size() -> int", &nnpa_max_dim_idx_size);
  ops.class_<NnpaPerLayerProjection>("NnpaPerLayerProjection")
      .def(torch::init<torch::Tensor, torch::Tensor, double, double, double>());
  ops.def(
      "nnpa_per_layer_project(__torch__.torch.classes._spyre_C.NnpaPerLayerProjection "
      "projection, Tensor x, Tensor? rows) -> Tensor",
      &nnpa_per_layer_project);
}
