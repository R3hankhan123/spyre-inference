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

"""Gemma-4 MTP: a separate assistant checkpoint that drafts from the target's KV cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.config import CompilationMode
from vllm.logger import init_logger

from spyre_inference.v1.spec_decode.base import SpyreSpecMethod

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig, VllmConfig

    from spyre_inference.v1.spec_decode.gemma4_proposer import SpyreGemma4Proposer

logger = init_logger(__name__)


def _extend_drafter_context(spec: SpeculativeConfig, target_max_model_len: int) -> None:
    """Let the MTP assistant draft over the target's whole context.

    Its config says 2048, and upstream zeroes the drafts of any step whose sequence is
    longer than the drafter's max_model_len, so past that every draft is rejected. The
    assistant has no KV of its own: it reads the target's cache at the target's
    positions, which is the shared positional space upstream already grants EAGLE drafts
    (#48894). Its rotary cache is sized from max_position_embeddings, so that goes up too.
    """
    draft_model_config = spec.draft_model_config
    if draft_model_config.max_model_len >= target_max_model_len:
        return
    hf_config = draft_model_config.hf_config
    # The draft layers build their rope from the text config.
    for config in (hf_config, getattr(hf_config, "text_config", hf_config)):
        positions = getattr(config, "max_position_embeddings", None)
        if positions is not None and positions < target_max_model_len:
            config.max_position_embeddings = target_max_model_len
    logger.info(
        "Extending the MTP drafter's max_model_len from %d to the target's %d.",
        draft_model_config.max_model_len,
        target_max_model_len,
    )
    draft_model_config.max_model_len = target_max_model_len


class Gemma4MtpMethod(SpyreSpecMethod):
    name = "mtp"
    # Every draft pass attends one row per request, through the target's builders.
    draft_query_len = 1

    @classmethod
    def matches(cls, spec: SpeculativeConfig) -> bool:
        return spec.use_gemma4_mtp()

    def check_config(self, vllm_config: VllmConfig) -> None:
        from spyre_inference.models.gemma4 import repair_head_dim_access

        spec = vllm_config.speculative_config
        assert spec is not None
        # The drafter's config was built beside the target's, with the checkpoint dtype.
        spec.draft_model_config.dtype = torch.float16
        draft_hf_config = spec.draft_model_config.hf_config
        repair_head_dim_access(getattr(draft_hf_config, "text_config", draft_hf_config))
        _extend_drafter_context(spec, vllm_config.model_config.max_model_len)
        # Async spec decode overlaps its bookkeeping on CUDA copy streams and events.
        if vllm_config.scheduler_config.async_scheduling:
            logger.info(
                "Disabling async scheduling: Spyre runs speculative decoding synchronously."
            )
            vllm_config.scheduler_config.async_scheduling = False

    def replace_proposer(self, runner: Any) -> None:
        from spyre_inference.v1.spec_decode.gemma4_proposer import create_spyre_drafter

        # Upstream built its own Gemma4Proposer; replace it before anything uses it.
        runner.drafter = create_spyre_drafter(runner.vllm_config, runner)

    def finish_drafter(self, runner: Any) -> None:
        """Load the assistant onto Spyre, compiled per block like the target."""
        from spyre_inference.v1.attention import attn_layer
        from spyre_inference.v1.worker.compile_granularity import (
            compile_blocks,
            repeated_block_lists,
        )
        from spyre_inference.v1.worker.spyre_model_runner import _SpyreDraftModelWrapper
        from spyre_inference.v1.worker.spyre_shape_bucketer import logits_row_buckets

        drafter = cast("SpyreGemma4Proposer", runner.drafter)
        logger.info("Loading drafter %s...", drafter.draft_model_config.model)
        drafter.load_model(runner.get_model())
        attn_layer.install_kv_readers(drafter.attention_layers())
        draft_model = drafter.model
        draft_model.to(device=runner._spyre_device)
        if not (
            runner.vllm_config.model_config.enforce_eager
            or runner.compilation_config.mode is CompilationMode.NONE
        ):
            num_blocks, _ = compile_blocks(repeated_block_lists(draft_model))
            logger.info("Wrapped %d drafter blocks for per-block compile on Spyre.", num_blocks)
        bucketer = runner.spyre_shape_bucketer
        drafter.model = _SpyreDraftModelWrapper(
            draft_model,
            runner._spyre_device,
            logits_row_buckets=(
                []
                if bucketer is None
                else logits_row_buckets(bucketer.bucket_sizes, runner.max_num_reqs)
            ),
            shape_bucketer=bucketer,
            model_dtype=runner._model_dtype(),
        )
