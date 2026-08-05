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

import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor

from .utils import convert


@LogitsProcessor.register_oot(name="LogitsProcessor")
class SpyreLogitsProcessor(LogitsProcessor):
    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Gather TP-sharded logits on Spyre, then move the result to CPU."""
        return convert(super()._gather_logits(logits), device="cpu")

    def _get_logits(self, hidden_states, lm_head, embedding_bias=None):
        """Ensure logits land on CPU regardless of TP size.

        For TP>1 ``_gather_logits`` already converts to CPU.  For TP=1 the
        gather is skipped by upstream, so logits would stay on Spyre and the
        downstream ``logits *= scale`` / ``logits.to(float32)`` would crash.
        """
        logits = super()._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None and logits.device.type != "cpu":
            logits = convert(logits, device="cpu")
        return logits
