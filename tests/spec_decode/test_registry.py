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

"""Which Spyre speculative-decoding method a config resolves to."""

from types import SimpleNamespace

import pytest

from spyre_inference.v1.spec_decode import registry
from spyre_inference.v1.spec_decode.dflash import DFlashMethod
from spyre_inference.v1.spec_decode.gemma4_mtp import Gemma4MtpMethod
from spyre_inference.v1.spec_decode.ngram import NgramMethod, SuffixMethod


def _spec(method, *, gemma4_mtp=False):
    return SimpleNamespace(
        method=method,
        use_gemma4_mtp=lambda: gemma4_mtp,
        use_dflash=lambda: method == "dflash",
    )


def test_no_speculative_config_has_no_method():
    assert registry.resolve(None) is None


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (_spec("ngram"), NgramMethod),
        (_spec("suffix"), SuffixMethod),
        (_spec("mtp", gemma4_mtp=True), Gemma4MtpMethod),
        (_spec("dflash"), DFlashMethod),
    ],
)
def test_each_supported_method_resolves_to_its_plugin(spec, expected):
    assert type(registry.resolve(spec)) is expected


@pytest.mark.parametrize("spec", [_spec("eagle3"), _spec("mtp", gemma4_mtp=False)])
def test_an_unsupported_method_is_rejected_by_name(spec):
    """An MTP checkpoint that is not a Gemma-4 assistant has no Spyre drafter."""
    with pytest.raises(ValueError, match=f"method={spec.method!r}.*ngram, suffix, mtp, dflash"):
        registry.resolve(spec)


def test_method_named_finds_a_validated_configs_method():
    assert registry.method_named("mtp") is Gemma4MtpMethod
    assert registry.method_named(None) is None


def test_only_gemma4_mtp_drafts_through_the_target_builders():
    assert [m for m in registry.METHODS if m.draft_query_len is not None] == [Gemma4MtpMethod]
    assert [m for m in registry.METHODS if m.records_draft_attention] == [DFlashMethod]
