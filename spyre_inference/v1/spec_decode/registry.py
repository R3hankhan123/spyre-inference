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

"""The speculative-decoding methods Spyre supports, and which one a config asks for."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spyre_inference.v1.spec_decode.base import SpyreSpecMethod
from spyre_inference.v1.spec_decode.dflash import DFlashMethod
from spyre_inference.v1.spec_decode.gemma4_mtp import Gemma4MtpMethod
from spyre_inference.v1.spec_decode.ngram import NgramMethod, SuffixMethod

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig

METHODS: tuple[type[SpyreSpecMethod], ...] = (
    NgramMethod,
    SuffixMethod,
    Gemma4MtpMethod,
    DFlashMethod,
)


def resolve(spec: SpeculativeConfig | None) -> SpyreSpecMethod | None:
    """The method ``spec`` asks for; None without speculative decoding.

    Raises:
        ValueError: if Spyre does not support the method.
    """
    if spec is None:
        return None
    for method in METHODS:
        if method.matches(spec):
            return method()
    supported = ", ".join(m.name for m in METHODS)
    raise ValueError(
        f"Spyre does not support speculative decoding with method={spec.method!r} "
        f"(supported: {supported}; mtp needs a Gemma-4 MTP assistant checkpoint)."
    )


def normalize_engine_args(engine_args: Any) -> None:
    """Let each method rewrite the user's raw speculative config before vLLM parses it."""
    for method in METHODS:
        method.normalize_engine_args(engine_args)


def method_named(name: object) -> type[SpyreSpecMethod] | None:
    """The method a validated config's ``method`` names, for code that has only the config.

    Unambiguous once ``resolve`` has accepted the config at platform time.
    """
    return next((m for m in METHODS if m.name == name), None)
