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

"""What a speculative-decoding method must provide to run on Spyre.

Verification is shared by every method and lives outside this interface: the verify
compile sizes (platform), the batched verify attention and the scheduler's draft
padding, the fused rejection sampler, and the verify-row logits widths. A method only
says how drafts are proposed: its config checks, its proposer, and where its drafter
loads. Each hook's default is upstream's behaviour, which is all a model-free method
such as n-gram needs.

The platform, worker and runner call these hooks at fixed points; ``registry.resolve``
picks the method from the speculative config.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig, VllmConfig


class SpyreSpecMethod:
    name: ClassVar[str]

    # Whether warmup records the drafter's own attention layers, with the causality its
    # model reports. A drafter that reads the target's KV cache (gemma-4 MTP) has its
    # layers in the target's attention groups instead.
    records_draft_attention: ClassVar[bool] = False

    # Query rows per request of the drafter's attention when it runs on the target's
    # attention builders, so warmup records the batched decode kernel at that width
    # beside the verify width. None when the drafter has no attention there.
    draft_query_len: ClassVar[int | None] = None

    @classmethod
    def matches(cls, spec: SpeculativeConfig) -> bool:
        raise NotImplementedError

    @classmethod
    def normalize_engine_args(cls, engine_args: Any) -> None:
        """Rewrite the user's raw speculative config before vLLM parses it."""

    def check_config(self, vllm_config: VllmConfig) -> None:
        """Platform-time checks and config fix-ups, before any worker starts."""

    def validate_runner(self, vllm_config: VllmConfig) -> None:
        """Checks the model runner makes before building anything."""

    def proposer_context(self) -> AbstractContextManager:
        """Wraps upstream's runner construction, which builds the proposer."""
        return nullcontext()

    def replace_proposer(self, runner: Any) -> None:
        """Runs after upstream's runner construction, to swap in a Spyre proposer."""

    def load_drafter(self, runner: Any) -> None:
        """The target is loaded on the host, before its padding fix-ups."""

    def place_drafter(self, runner: Any) -> None:
        """The target is on Spyre, not yet compiled."""

    def finish_drafter(self, runner: Any) -> None:
        """The target is compiled and wrapped."""

    def before_warmup(self, worker: Any) -> None:
        """Runs just before the worker's warmup."""

    def input_fits_in_drafter(
        self, runner: Any, upstream: Callable[[Any], bool], common_attn_metadata: Any
    ) -> bool:
        return upstream(common_attn_metadata)

    def propose(self, runner: Any, upstream: Callable[..., Any], *args: Any) -> Any:
        """The step's drafts. ``args`` are ``propose_draft_token_ids``'s, in order."""
        return upstream(*args)
