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

"""Gemma 4's text decoder (``SpyreGemma4ForCausalLM``, MoE layers) and its startup hooks.

The hooks (``force_text_backbone``, ``repair_head_dim_access``) run for every model, so this
module must not import the vision model; that lives in ``gemma4_mm.py``.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from vllm.distributed import GroupCoordinator, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gemma4 import (
    Gemma4ForCausalLM,
    Gemma4Router,
    Gemma4SelfDecoderLayers,
)
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    replace_as,
)

from spyre_inference import envs
from spyre_inference.custom_ops.lazy_compile import CompileOutermost, maybe_compile
from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._prelaunch import disable_modalities
from spyre_inference.models._retype import retype
from spyre_inference.moe import SpyreMoERecipe, configure_spyre_moe_layer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from transformers import PretrainedConfig
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)

# Each "global" attribute vLLM's gemma-4 builder reads for full-attention layers,
# mapped to the per-layer attribute of the same role it is rebuilt from.
_GEMMA4_FULL_ATTENTION_ATTRS = {
    "global_head_dim": "head_dim",
    "num_global_key_value_heads": "num_key_value_heads",
}


def repair_head_dim_access(config: Any) -> None:
    """Let bare reads (config.head_dim, config.num_key_value_heads) return the
    sliding/global scalar, matching vLLM's sliding-layer path.

    transformers >=5.16 reclassifies gemma-4 as heterogeneous: the bare ``config.head_dim``
    read vLLM's gemma-4 builders do then raises ``AmbiguousGlobalPerLayerAttributeError``,
    and the ``global_*`` head/kv-head attributes vLLM needs are consumed into
    ``per_layer_config``. This restores the <=5.14 view in place.

    Idempotent and a no-op for any other config, because it runs from
    ``TorchSpyrePlatform.check_and_update_config`` rather than from ``hf_overrides``: the
    repair then also reaches a config the user overrode themselves. Nothing inside config
    creation needs it -- ``ModelConfig.get_head_size`` reads gemma-4's heterogeneous head
    dims through its own path -- so the platform hook is early enough.

    The ambiguous read lives in ``Gemma4TextConfig``, so a multimodal checkpoint's nested
    ``text_config`` needs it just as much as a text-only top-level config.
    """
    if getattr(config, "model_type", None) not in _GEMMA4_TEXT_MODEL_TYPES:
        return
    text_config = getattr(config, "text_config", config)
    for cfg in {id(config): config, id(text_config): text_config}.values():
        cfg.allow_global_per_layer_attribute_access = True
    per_layer = getattr(text_config, "per_layer_config", None)
    layer_types = getattr(text_config, "layer_types", None)
    if per_layer is not None and layer_types:
        full_idx = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
        for global_attr, src_attr in _GEMMA4_FULL_ATTENTION_ATTRS.items():
            if hasattr(text_config, global_attr) or not full_idx:
                continue
            values = {getattr(per_layer[i], src_attr) for i in full_idx}
            if len(values) == 1:
                setattr(text_config, global_attr, values.pop())


# The supported way to run a checkpoint that carries a vision tower as text-only: it skips
# the tower's weights and warmup entirely. Applied by default to a checkpoint with no
# tower; pass it as `hf_overrides` to opt a vision checkpoint in.
GEMMA4_TEXT_BACKBONE_OVERRIDE = {"architectures": ["Gemma4ForCausalLM"]}


# gemma-4 config model_types this fix applies to. Excludes the other gemma4_* types
# (unified, dspark, mtp, audio, vision): they have their own vLLM builders and must
# not be forced onto the text backbone.
_GEMMA4_TEXT_MODEL_TYPES = {"gemma4", "gemma4_text"}


def is_multimodal_gemma4(hf_config: Any) -> bool:
    """True for a gemma-4 config carrying a vision tower. Audio is out of scope."""
    if getattr(hf_config, "model_type", None) not in _GEMMA4_TEXT_MODEL_TYPES:
        return False
    return getattr(hf_config, "vision_config", None) is not None


def force_text_backbone(engine_args: EngineArgs, hf_config: PretrainedConfig | None) -> None:
    """Default a gemma-4 checkpoint with no vision tower to its text-only backbone.

    Only a default: it is the same ``hf_overrides`` value ``GEMMA4_TEXT_BACKBONE_OVERRIDE``
    documents, so a user override of any shape replaces it wholesale (the head-dim repair
    they also need runs from the platform hook, not from here). A vision checkpoint keeps
    its real ``architectures`` -- forcing ``Gemma4ForCausalLM`` there would strip the tower
    -- with video and audio capped at 0 per prompt, while an audio-only one is rejected
    rather than silently reduced to text.
    """
    if engine_args.hf_overrides:
        return
    # Detect gemma-4 by config model_type, not the checkpoint name: derivatives such as
    # medgemma / translategemma carry a gemma4 config under an unrelated name.
    if getattr(hf_config, "model_type", None) not in _GEMMA4_TEXT_MODEL_TYPES:
        return
    has_audio = getattr(hf_config, "audio_config", None) is not None
    # A multimodal Gemma4Config reports the same model_type as a text checkpoint.
    if is_multimodal_gemma4(hf_config):
        disable_modalities(engine_args, ("video", "audio") if has_audio else ("video",), "gemma-4")
        return
    if has_audio:
        # Falling through would force the text-only backbone and silently drop the
        # tower, leaving a model that looks fine but ignores its audio inputs.
        raise NotImplementedError(
            "gemma-4 audio is not supported on Spyre, and this checkpoint has an audio "
            "tower but no vision tower. Use a text-only or vision checkpoint."
        )
    engine_args.hf_overrides = dict(GEMMA4_TEXT_BACKBONE_OVERRIDE)
    logger.info("gemma-4: loading text-only backbone Gemma4ForCausalLM.")


def _fold_gemma4_expert_scale(
    down_weight: torch.Tensor, experts: slice, scale: torch.Tensor
) -> torch.Tensor:
    """Fold Gemma's output scale into a chunk ``experts`` of the source down stack."""
    return down_weight * scale[experts].detach().to(down_weight.dtype).view(-1, 1, 1)


def configure_gemma4_moe_layers(layers: Iterable[nn.Module]) -> int:
    """Register Gemma-4's full-softmax, scaled GELU expert recipe; returns the count."""

    configured = 0
    for decoder in layers:
        experts = getattr(decoder, "experts", None)
        if experts is None:
            continue
        configure_spyre_moe_layer(
            experts.routed_experts,
            SpyreMoERecipe(
                activation="gelu_tanh",
                routing="full_softmax",
                prepare_down_weight=partial(
                    _fold_gemma4_expert_scale,
                    scale=cast(Gemma4Router, decoder.router).per_expert_scale,
                ),
                # At 1,024 tokens four experts per trip are slower than one (TP 4 shapes).
                expert_group=4,
                expert_group_tokens=(128, 512),
            ),
        )
        configured += 1
    if configured:
        logger.info("Spyre: configured %d Gemma-4 MoE layers.", configured)
    return configured


class _PerLayerRows(torch.Tensor):
    """PLE tensor whose ``[:, layer_idx, :]`` returns a row with its own storage, so all
    layers share one compiled block (a view's storage_offset is guarded per layer)."""

    __torch_function__ = torch._C._disabled_torch_function_impl  # ty: ignore[invalid-method-override]

    spyre_rows: tuple[torch.Tensor, ...]

    def __getitem__(self, index: Any) -> torch.Tensor:
        if (
            isinstance(index, tuple)
            and len(index) == 3
            and index[0] == index[2] == slice(None)
            and isinstance(index[1], int)
        ):
            return self.spyre_rows[index[1]]
        return torch.Tensor.__getitem__(self, index)


class _HostResident:
    """Keeps a module's tensors in host memory when the model moves to Spyre."""

    def _apply(self, fn, recurse=True):
        return self


def _skip_weight(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    pass


class _HostPerLayerTable(_HostResident, nn.Module):
    """Gemma 4's per-layer table, whole, in the host memory of the rank that looks it up.

    Replaces the vocab-sharded embedding. Every other rank keeps an empty weight and
    skips the checkpoint tensor.
    """

    def __init__(self, num_embeddings: int, dim: int, dtype: torch.dtype, *, load: bool):
        super().__init__()
        rows = num_embeddings if load else 0
        self.weight = nn.Parameter(torch.empty(rows, dim, dtype=dtype, device="cpu"), False)
        self.weight.weight_loader = default_weight_loader if load else _skip_weight  # ty: ignore[unresolved-attribute]


def _ple_fits_nnpa(vllm_config: VllmConfig, decoder: Gemma4SelfDecoderLayers) -> bool:
    """Whether this host can take the per-layer inputs off the card (s390x with NNPA)."""
    from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod
    from spyre_inference.nnpa import nnpa_max_dim

    max_dim = nnpa_max_dim()
    if not max_dim:
        return False
    projection = decoder.per_layer_model_projection
    return (
        # The streaming loader writes the weights straight to the card.
        vllm_config.load_config.load_format != "spyre"
        # Host ops cannot be traced into a whole-model graph.
        and envs.SPYRE_COMPILE_GRANULARITY != "model"
        and type(projection.quant_method) is SpyreUnquantizedLinearMethod
        and max(projection.input_size, projection.output_size) <= max_dim
    )


def _rebuild_ple_for_host(
    model: nn.Module, decoder: Gemma4SelfDecoderLayers, vllm_config: VllmConfig
) -> None:
    """Swap the TP-sharded table and projection for whole ones before loading.

    TP rank 0 alone holds the table and looks it up. The projection is small, so every
    rank loads it whole; only rank 0 runs it.
    """
    tp = get_tp_group()
    modules: dict[str, nn.Module] = {
        "embed_tokens_per_layer": _HostPerLayerTable(
            decoder.vocab_size_per_layer_input,
            decoder.config.num_hidden_layers * decoder.hidden_size_per_layer_input,
            vllm_config.model_config.dtype,
            load=tp.rank_in_group == 0,
        )
    }
    if tp.world_size > 1:
        projection = decoder.per_layer_model_projection
        modules["per_layer_model_projection"] = ColumnParallelLinear(
            projection.input_size,
            projection.output_size,
            bias=False,
            gather_output=True,
            return_bias=False,
            quant_config=projection.quant_config,
            prefix=projection.prefix,
            disable_tp=True,
        )
    # Gemma4Model owns the modules (and loads them); the decoder shares them.
    for name, module in modules.items():
        setattr(model, name, module)
        setattr(decoder, name, module)


class SpyreGemma4SelfDecoderLayers(CompileOutermost, Gemma4SelfDecoderLayers):
    ple_on_nnpa: bool = False
    """Per-layer inputs off the card (``place_ple_on_nnpa``)."""

    def place_ple_on_nnpa(self, tp_group: GroupCoordinator | None = None) -> None:
        """Look up the per-layer table on the host and project on NNPA.

        The table stays in host memory, so the card holds none of it, and the only
        transfers are the token ids and input embeddings down and the result up. Under
        TP the group's first rank computes the result and broadcasts it on the host.
        """
        for module in (
            self.embed_tokens_per_layer,
            self.per_layer_model_projection,
            self.per_layer_projection_norm,
        ):
            if not isinstance(module, _HostResident):
                base = type(module)
                module.__class__ = type(f"Host{base.__name__}", (_HostResident, base), {})
        # Read while the buffers are still on the host.
        self.ple_embed_scale = float(self.embed_scale_per_layer)
        self.ple_input_scale = float(self.per_layer_input_scale)
        self.ple_projection_scale = float(self.per_layer_projection_scale)
        self.ple_group = tp_group
        self.ple_source = tp_group is None or tp_group.rank_in_group == 0
        self.nnpa_projection = None
        self.ple_on_nnpa = True
        logger.info("Spyre: Gemma-4 per-layer inputs run on the host and NNPA.")

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if self.ple_on_nnpa:
            if not self.ple_source:
                return None
            return self._host_per_layer_inputs(convert(input_ids, device="cpu"))
        return self._spyre_per_layer_inputs(input_ids)

    @maybe_compile(force=True)
    def _spyre_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        # The upstream mask needs a compiled region even outside decoder blocks.
        return super().get_per_layer_inputs(input_ids)

    def _host_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Upstream's masked lookup, on the host-resident table. Unscaled: the NNPA
        projection applies the embed scale in its combine pass."""
        ids = input_ids.long()
        ids = ids.masked_fill((ids < 0) | (ids >= self.vocab_size_per_layer_input), 0)
        table = cast(torch.Tensor, self.embed_tokens_per_layer.weight)
        embeds = F.embedding(ids, table)
        return embeds.reshape(
            *input_ids.shape, self.config.num_hidden_layers, self.hidden_size_per_layer_input
        )

    def _nnpa_project(
        self, inputs_embeds: torch.Tensor, per_layer_inputs: torch.Tensor | None
    ) -> torch.Tensor:
        """Upstream's projection, norm and combine: the matmul on NNPA, the rest in one fp32
        host pass."""
        shape = (
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        if self.ple_source:
            ple = self._nnpa_project_here(inputs_embeds, per_layer_inputs).reshape(shape)
        else:
            ple = torch.empty(shape, dtype=inputs_embeds.dtype)
        if self.ple_group is not None:
            group = self.ple_group
            dist.broadcast(ple, src=group.first_rank, group=group.cpu_group)
        return convert(ple, device=inputs_embeds.device)

    def _nnpa_project_here(
        self, inputs_embeds: torch.Tensor, per_layer_inputs: torch.Tensor | None
    ) -> torch.Tensor:
        if self.nnpa_projection is None:
            # SpyreUnquantizedLinearMethod already stored Wᵀ, [in, out], on the host.
            weight_t = cast(torch.Tensor, self.per_layer_model_projection.weight).float()
            norm = self.per_layer_projection_norm
            self.nnpa_projection = torch.classes._spyre_C.NnpaPerLayerProjection(  # ty: ignore[unresolved-attribute]
                (weight_t * self.ple_projection_scale).half(),
                norm.weight,
                norm.variance_epsilon,
                self.ple_embed_scale,
                self.ple_input_scale,
            )
        hidden = convert(inputs_embeds, device="cpu")
        hidden = hidden.reshape(-1, hidden.shape[-1])
        rows = None if per_layer_inputs is None else per_layer_inputs.reshape(hidden.shape[0], -1)
        return torch.ops._spyre_C.nnpa_per_layer_project(  # ty: ignore[unresolved-attribute]
            self.nnpa_projection, hidden, rows
        )

    @maybe_compile
    def split_per_layer_inputs(self, ple: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # One graph: L eager clones would cost L host launches per step.
        return tuple(ple[:, i, :].clone() for i in range(self.config.num_hidden_layers))

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs):
        if self.ple_on_nnpa:
            ple = self._nnpa_project(inputs_embeds, per_layer_inputs)
        else:
            ple = cast(
                torch.Tensor, super().project_per_layer_inputs(inputs_embeds, per_layer_inputs)
            )
        rows = ple.as_subclass(_PerLayerRows)
        rows.spyre_rows = self.split_per_layer_inputs(ple)
        return rows


class SpyreGemma4ForCausalLM(Gemma4ForCausalLM):
    """Gemma-4 on Spyre with Spyre MoE expert dispatch."""

    @classmethod
    def get_kv_cache_groups(
        cls, vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
    ) -> list[KVCacheGroupSpec]:
        speculative = vllm_config.speculative_config
        if speculative is None or not speculative.use_dflash():
            return get_kv_cache_groups(vllm_config, kv_cache_spec)
        # Full allocation shares a block table without merging target/draft windows.
        specs = {
            name: replace_as(spec, FullAttentionSpec, drop=("extra_retained_tokens",))
            if isinstance(spec, SlidingWindowSpec)
            else spec
            for name, spec in kv_cache_spec.items()
        }
        uniform = UniformTypeKVCacheSpecs.from_specs(specs)
        if uniform is None:
            raise ValueError("Gemma 4 DFlash requires one full-allocation KV cache group")
        return [KVCacheGroupSpec(list(specs), uniform)]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        decoder = self.model.self_decoder
        if decoder.embed_tokens_per_layer is not None:
            decoder = retype(decoder, SpyreGemma4SelfDecoderLayers)
            decoder.arm_outer_compile()
            if _ple_fits_nnpa(vllm_config, decoder):
                _rebuild_ple_for_host(self.model, decoder, vllm_config)
                tp = get_tp_group()
                decoder.place_ple_on_nnpa(tp if tp.world_size > 1 else None)
        # Without the recipe the experts run upstream's path, which is wrong on Spyre.
        # Either flag enables the MoE block, as in upstream's Gemma4DecoderLayer.
        is_moe_checkpoint = getattr(self.config, "enable_moe_block", False) or getattr(
            self.config, "use_second_mlp_block", False
        )
        if not configure_gemma4_moe_layers(self.model.layers) and is_moe_checkpoint:
            raise RuntimeError(
                "gemma-4 MoE checkpoint, but no decoder layer exposes `experts`: vLLM has "
                "moved them; update configure_gemma4_moe_layers."
            )
