# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform

import pytest
import torch

from spyre_inference import nnpa


@pytest.fixture
def machine(monkeypatch):
    def set_machine(name: str) -> None:
        monkeypatch.setattr(platform, "machine", lambda: name)
        nnpa.nnpa_max_dim.cache_clear()

    yield set_machine
    nnpa.nnpa_max_dim.cache_clear()


@pytest.mark.parametrize("name", ["x86_64", "ppc64le"])
def test_offload_is_s390x_only(machine, monkeypatch, name):
    # Even with the op present, only s390x may offload.
    monkeypatch.setattr(nnpa, "has_sampling_kernels", lambda: True)
    machine(name)
    assert nnpa.nnpa_max_dim() == 0


@pytest.mark.skipif(platform.machine() == "s390x", reason="s390x may build the NNPA op")
def test_build_without_nnpa_op_reports_zero(machine):
    machine("s390x")
    assert nnpa.nnpa_max_dim() == 0


@pytest.mark.skipif(nnpa.nnpa_max_dim() == 0, reason="needs NNPA (z16+) and libzdnn")
@pytest.mark.parametrize("with_rows", [True, False])
@pytest.mark.parametrize("tokens", [1, 7, 300])
def test_nnpa_per_layer_project_matches_host_reference(tokens, with_rows):
    layers, width, hidden, eps, embed_scale, input_scale = 3, 256, 1536, 1e-6, 16.0, 0.5**0.5
    g = torch.Generator().manual_seed(0)
    weight_t = (torch.randn(hidden, layers * width, generator=g) * 0.02).half()
    norm_weight = (torch.rand(width, generator=g) + 0.5).half()
    x = torch.randn(tokens, hidden, generator=g).half()
    rows = torch.randn(tokens, layers * width, generator=g).half() if with_rows else None
    projection = torch.classes._spyre_C.NnpaPerLayerProjection(  # ty: ignore[unresolved-attribute]
        weight_t, norm_weight, eps, embed_scale, input_scale
    )

    actual = torch.ops._spyre_C.nnpa_per_layer_project(projection, x, rows)  # ty: ignore[unresolved-attribute]

    proj = (x.float() @ weight_t.float()).reshape(tokens, layers, width)
    expected = proj * torch.rsqrt(proj.square().mean(-1, keepdim=True) + eps) * norm_weight.float()
    if rows is not None:
        expected = (expected + rows.float().reshape(proj.shape) * embed_scale) * input_scale
    assert actual.dtype == torch.float16 and actual.shape == (tokens, layers * width)
    # DLFloat16 keeps a 9-bit mantissa; the accumulation stays wide.
    torch.testing.assert_close(actual.float(), expected.reshape(tokens, -1), atol=3e-2, rtol=3e-2)
