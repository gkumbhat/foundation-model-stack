import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple
from typing_extensions import Unpack

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
)
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    get_attention_type,
)
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig
from fms.utils.headless import gather_outputs

logger = logging.getLogger(__name__)


_architecture_name = "gemma4"


@dataclass
class Gemma4TextConfig(ModelConfig):
    """
    Configuration for the FMS Gemma4 text model.

    All layers use global (full) attention — no sliding window attention.
    Parameters correspond to the global-attention configuration from:
    https://huggingface.co/google/gemma-4-12B-it

    HF config mapping (transformers 5+):
      vocab_size                  -> src_vocab_size
      hidden_size                 -> emb_dim
      num_attention_heads         -> nheads
      num_global_key_value_heads  -> kvheads  (global attention KV heads)
      num_hidden_layers           -> nlayers
      global_head_dim             -> head_dim  (global attention head dim)
      intermediate_size / hidden_size -> hidden_grow_factor
      rms_norm_eps                -> norm_eps
      rope_parameters             -> rope_parameters  (full dict; full_attention entry used)
      layer_types                 -> layer_types
      tie_word_embeddings         -> tie_heads
      final_logit_softcapping     -> logit_softcapping
      pad_token_id                -> pad_id
    """

    src_vocab_size: int = 262144
    emb_dim: int = 3840
    nheads: int = 16
    kvheads: int = 1  # num_global_key_value_heads; all layers use global attention
    nlayers: int = 48
    head_dim: int = 512  # global_head_dim; all layers use global attention head dim
    hidden_grow_factor: float = 15360 / 3840  # intermediate_size / hidden_size
    multiple_of: int = 256
    activation_fn: str = "gelu_pytorch_tanh"
    p_dropout: float = 0.0
    max_expected_seq_len: int = 262144
    norm_eps: float = 1e-06
    # rope_parameters mirrors the HF config structure:
    #   {"full_attention": {"rope_theta": ..., "partial_rotary_factor": ..., "rope_type": ...},
    #    "sliding_attention": {...}}
    # Only the "full_attention" entry is used; sliding attention is not supported.
    rope_parameters: Dict = field(
        default_factory=lambda: {
            "full_attention": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
                "rope_type": "proportional",
            }
        }
    )
    # layer_types mirrors the HF config field. Each entry is either
    # "sliding_attention" or "full_attention". The default reproduces the
    # Gemma4-12B pattern: 5 sliding then 1 full, repeated 8 times.
    layer_types: list = field(
        default_factory=lambda: (
            ["sliding_attention"] * 5 + ["full_attention"]
        ) * 8
    )
    qk_norm: bool = True  # per-head QK normalization (apply_norm_per_head)
    logit_softcapping: float = 30.0  # final_logit_softcapping
    tie_heads: bool = True
    # Attention is always unfused (UnfusedQKV) because FusedQKV does not
    # support per-head QK normalization.  fused_weights controls only the MLP.
    fused_weights: bool = False
    pad_id: int = 0
    linear_config: Optional[Mapping[str, Any]] = None


_12b_config = Gemma4TextConfig()


# =============== Modeling ======================


class _NormedValueProjection(nn.Module):
    """
    Value projection (nn.Linear) followed by per-head RMSNorm without a learnable
    scale — implements Gemma4's v_norm (Gemma4RMSNorm with with_scale=False).

    Applied to value states after projection but before the attention dot-product.
    Has no checkpoint weights; purely a runtime normalization.
    """

    def __init__(self, linear: nn.Linear, head_dim: int, eps: float):
        super().__init__()
        self.linear = linear
        self.head_dim = head_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)  # (B, L, kvheads * head_dim)
        # Per-head RMSNorm, no learnable scale (Gemma4 v_norm).
        # kvheads=1 for Gemma4, so out is (B, L, head_dim) — one head.
        # 3D mean(dim=2) stays on (B,L,D) with L pure in dim-1. Spyre-safe.
        inv_rms = torch.rsqrt(out.pow(2).mean(2, keepdim=True) + self.eps)
        return out * inv_rms


class Gemma4Block(nn.Module):
    def __init__(self, config: Gemma4TextConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.config = config

        if self.config.kvheads == 0:
            kvheads = self.config.nheads
        else:
            kvheads = self.config.kvheads
            assert self.config.nheads % self.config.kvheads == 0

        def _layernorm(emb_dim):
            return LayerNormParameterized(
                emb_dim,
                elementwise_scale=True,
                elementwise_shift=False,
                use_mean=False,
                eps=self.config.norm_eps,
                use_high_precision_pow=True,
            )

        self.ln = _layernorm(self.config.emb_dim)           # input_layernorm
        self.post_attn_ln = _layernorm(self.config.emb_dim) # post_attention_layernorm
        self.ff_ln = _layernorm(self.config.emb_dim)        # pre_feedforward_layernorm
        self.post_ff_ln = _layernorm(self.config.emb_dim)   # post_feedforward_layernorm

        # Per-layer scalar — initialised to 1 (identity); trained to modulate output.
        self.register_buffer("layer_scalar", torch.ones(1))

        # Attention is always unfused: FusedQKV does not create q_norm/k_norm
        # parameters, so per-head QK normalization requires UnfusedQKV.
        self.attn = MultiHeadAttention(
            self.config.emb_dim,
            self.config.head_dim,
            self.config.head_dim,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=False,
            position_encoder=rotary_emb,
            fused=False,
            linear_config=self.config.linear_config,
            apply_norm_per_head=self.config.qk_norm,
            norm_eps=self.config.norm_eps if self.config.qk_norm else None,
            head_dim=self.config.head_dim,
        )
        # v_norm: wrap the value projection with a no-scale per-head RMSNorm.
        # Has no checkpoint weights (with_scale=False in HF).
        self.attn.in_proj.value = _NormedValueProjection(
            self.attn.in_proj.value,
            head_dim=self.config.head_dim,
            eps=self.config.norm_eps,
        )

        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=False,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x,
        *,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        self_attn_past_key_value = past_key_value_state

        residual = x
        x = self.ln(x)
        x = self.attn(
            q=x,
            position_ids=position_ids,
            past_key_value_state=self_attn_past_key_value,
            use_cache=use_cache,
            **attn_kwargs,
        )
        cache = None
        if use_cache:
            x, cache = x
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        x = self.post_attn_ln(x)
        x = x + residual

        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        x = self.post_ff_ln(x)
        x = x + residual

        x = x * self.layer_scalar

        if use_cache:
            return (x, cache)
        else:
            return x


class Gemma4Headless(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
    ):
        super().__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        self.embedding = nn.Embedding(
            self.config.src_vocab_size,
            self.config.emb_dim,
            padding_idx=self.config.pad_id,
        )

        full_attn_rope = self.config.rope_parameters.get("full_attention", {})
        self.rot_emb = RotaryEmbedding(
            dim=self.config.head_dim,
            ratio=full_attn_rope.get("rope_theta", 1000000.0),
            max_seq_len=self.config.max_expected_seq_len,
            partial_rope=full_attn_rope.get("partial_rotary_factor", 0.25),
        )

        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = Gemma4Block(self.config, self.rot_emb)
            block = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

        dec_norm = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.dec_norm = self.distributed_strategy.distribute_module(
            dec_norm, final_layers=True
        )

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.embedding.weight, mean=0.0, std=self.config.emb_dim**-0.5
        )

        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        for m in self.modules():
            if (
                isinstance(m, MultiHeadAttention)
                or isinstance(m, GatedLinearUnit)
                or isinstance(m, LayerNormParameterized)
            ):
                m.reset_parameters()

    def _clean_up_rot_emb_cache(
        self,
        cached_freqs: dict[Optional[torch.device], dict[int, torch.Tensor]],
        max_seq_len_cached: dict[Optional[torch.device], int],
    ):
        for dev in list(cached_freqs.keys()):
            for alp in list(cached_freqs[dev].keys()):
                if cached_freqs[dev][alp].device == torch.device("meta"):
                    del cached_freqs[dev][alp]
                    if len(cached_freqs[dev]) == 0:
                        del cached_freqs[dev]
                        del max_seq_len_cached[dev]

    def post_init(self):
        self._clean_up_rot_emb_cache(
            self.rot_emb.cached_freqs,
            self.rot_emb.max_seq_len_cached,
        )
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

    def forward(
        self,
        x_in,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        if x_in.dim() == 2:
            x_in = self.embedding(x_in)
            # Gemma normalizes token embeddings by sqrt(hidden_size)
            x_in = x_in * (self.config.emb_dim**0.5)

        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x_in,
                position_ids=position_ids,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
                **attn_kwargs,
            )

            if use_cache:
                x_in, present_key_value_state = output
                present_key_value_states.append(present_key_value_state)
            else:
                x_in = output

        dec_out = x_in
        dec_out = self.dec_norm(dec_out)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states


class Gemma4Text(nn.Module):
    def __init__(
        self,
        config: Optional[Gemma4TextConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = Gemma4TextConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = Gemma4Headless(self.config, self.distributed_strategy)
        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

    @classmethod
    def from_config(cls, config: Gemma4TextConfig) -> "Gemma4Text":
        return cls(config)

    def get_config(self) -> Gemma4TextConfig:
        return self.config

    def reset_parameters(self):
        self.head.weight.data.normal_(
            0,
            1 / math.sqrt(math.sqrt(self.config.emb_dim * self.config.src_vocab_size)),
        )
        self.base_model.reset_parameters()

    def post_init(self):
        if self.config.tie_heads:
            if self.head.weight.device == torch.device("meta"):
                self.head.weight = self.base_model.embedding.weight
            else:
                self.base_model.embedding.weight = self.head.weight
        self.base_model.post_init()

    def forward(
        self,
        x: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        use_cache: bool = False,
        last_n_tokens: int = 0,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        get_attention_type(**attn_kwargs)["validate_attn_kwargs"](
            input_ids=x,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            **attn_kwargs,
        )
        output, cache = self.base_model(
            x, position_ids, past_key_value_states, use_cache, **attn_kwargs
        )

        output = gather_outputs(output, last_n_tokens, **attn_kwargs)
        preds = self.head(output)

        if self.config.logit_softcapping > 0.0:
            preds = (
                torch.tanh(preds / self.config.logit_softcapping)
                * self.config.logit_softcapping
            )

        if use_cache:
            return preds, cache
        else:
            return preds


# =============== Registration ==================


def _gemma4_factory_factory(config):
    def factory(**kwargs):
        return Gemma4Text(config, **kwargs)

    return factory


models.register_model(
    _architecture_name, "12b", _gemma4_factory_factory(_12b_config)
)


# =============== Serialization ==================


def _weight_fusion(
    input_sd: Mapping[str, Any],
    model_config: Optional[Gemma4TextConfig] = None,
    **kwargs,
) -> Mapping[str, Any]:
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        # Attention is always UnfusedQKV (required for q_norm/k_norm support),
        # so only the MLP weights are fused.
        new_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(new_sd)
    return new_sd


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)


# Reused by both _hf_to_fms_names and _filter_sliding_attn_weights
_LAYER_IDX_RE = re.compile(r"base_model\.layers\.(\d+)\.")


def _hf_to_fms_names(
    input_sd: Mapping[str, Any],
    model_config: Optional[Gemma4TextConfig] = None,
    **kwargs,
) -> Mapping[str, Any]:
    """
    Rename HuggingFace Gemma4UnifiedForConditionalGeneration weight keys to FMS naming.

    Checkpoint key structure (from google/gemma-4-12B-it):
      model.language_model.embed_tokens.weight
      model.language_model.norm.weight
      model.language_model.layers.N.{input,pre_feedforward,post_attention,post_feedforward}_layernorm.weight
      model.language_model.layers.N.self_attn.{q,k,o}_proj.weight  (sliding: also v_proj)
      model.language_model.layers.N.self_attn.{q,k}_norm.weight
      model.language_model.layers.N.mlp.{gate,up,down}_proj.weight
      model.language_model.layers.N.layer_scalar          <- dropped (no FMS equivalent)
      model.{embed_audio,embed_vision,vision_embedder}.*  <- dropped (vision/audio)
      lm_head is absent (tied to embed_tokens, handled by post_init)

    Gemma4 has 4 norms per layer; this FMS implementation uses only 2 (pre-attention
    and pre-FFN). The post_attention_layernorm and post_feedforward_layernorm weights
    are dropped on load.

    attention_k_eq_v: for full_attention layers the checkpoint has no v_proj —
    K and V share the same weight.  We duplicate k_proj as in_proj.value so
    that the downstream weight-fusion step receives all three (Q, K, V) keys.
    """
    _DROP_PREFIXES = (
        "model.embed_audio.",
        "model.embed_vision.",
        "model.vision_embedder.",
        "model.vision_tower.",
        "model.multi_modal_projector.",
        "vision_tower.",
        "multi_modal_projector.",
    )
    _DROP_SUFFIXES = ()  # all layer keys are now mapped

    replacements = [
        (r"^model\.language_model\.embed_tokens", "base_model.embedding"),
        (r"^model\.language_model\.norm", "base_model.dec_norm"),
        (r"^model\.language_model\.layers", "base_model.layers"),
        (r"^model\.language_model\.lm_head\.weight", "head.weight"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        # v_proj → value.linear because the value projection is wrapped in
        # _NormedValueProjection (v_norm); the nn.Linear lives at .linear
        (r"self_attn\.v_proj", "attn.in_proj.value.linear"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"self_attn\.q_norm", "attn.in_proj.q_norm"),
        (r"self_attn\.k_norm", "attn.in_proj.k_norm"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "post_attn_ln"),
        (r"pre_feedforward_layernorm", "ff_ln"),
        (r"post_feedforward_layernorm", "post_ff_ln"),
    ]

    layer_types = (
        model_config.layer_types
        if model_config is not None
        else (["sliding_attention"] * 5 + ["full_attention"]) * 8
    )
    full_attn_indices = {i for i, t in enumerate(layer_types) if t == "full_attention"}

    new_sd = {}
    for name, param in input_sd.items():
        if any(name.startswith(p) for p in _DROP_PREFIXES):
            continue
        if any(name.endswith(s) for s in _DROP_SUFFIXES):
            continue
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)

        new_sd[new_name] = param

        # For full_attention layers k_proj and v_proj share the same weight
        # (attention_k_eq_v=True in the HF config). The checkpoint only stores
        # k_proj, so we duplicate it under the value key so the weight-fusion
        # step receives all three of Q, K, V.
        if "attn.in_proj.key" in new_name:
            m = _LAYER_IDX_RE.search(new_name)
            if m and int(m.group(1)) in full_attn_indices:
                # attention_k_eq_v=True: full-attention layers share K and V weights.
                # The value projection lives at .value.linear (wrapped in v_norm).
                value_name = new_name.replace(
                    "attn.in_proj.key", "attn.in_proj.value.linear"
                )
                new_sd[value_name] = param

    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any],
    model_config: Optional[Gemma4TextConfig] = None,
    **kwargs,
) -> Mapping[str, Any]:
    """
    Transpose Q/K weights to account for FMS vs HF RoPE convention differences.
    See mistral.py for the detailed explanation of why this transpose is needed.
    """
    new_sd = {}

    if model_config is None:
        head_dim = 512
        logger.warning("Missing model_config, assuming default global head_dim=512")
    else:
        head_dim = model_config.head_dim

    rope_params = ["weight", "bias"]
    trans_required_pattern = re.compile(
        f"base_model\\.layers\\.[0-9]+\\.attn\\.in_proj\\.(query|key)\\.({'|'.join(rope_params)})"
    )

    for name, param in input_sd.items():
        if bool(trans_required_pattern.search(name)):
            temp = param
            num_heads = temp.size(0) // head_dim

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # bias
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)


_ATTN_SUBKEYS = (
    "attn.in_proj",
    "attn.dense",
)


def _filter_sliding_attn_weights(
    input_sd: Mapping[str, Any],
    model_config: Optional[Gemma4TextConfig] = None,
    **kwargs,
) -> Mapping[str, Any]:
    """
    Drop attention weights that belong to sliding-attention layers.

    The Gemma4 checkpoint mixes two layer types identified by model_config.layer_types:
      - "sliding_attention": smaller head dims (head_dim=256, kvheads=8)
      - "full_attention":    global head dims (head_dim=512, kvheads=1) — what FMS uses

    Detection is done from the layer index embedded in the key name: the layer
    index is extracted, looked up in layer_types, and if the entry is
    "sliding_attention" the weight is dropped.  MLP and norm weights are not
    attention subkeys and pass through unchanged.
    """
    if model_config is None:
        layer_types: list = (["sliding_attention"] * 5 + ["full_attention"]) * 8
        logger.warning(
            "Missing model_config in _filter_sliding_attn_weights; "
            "using default Gemma4-12B layer_types pattern."
        )
    else:
        layer_types = model_config.layer_types

    new_sd: dict = {}
    n_dropped = 0

    for name, param in input_sd.items():
        if not any(sub in name for sub in _ATTN_SUBKEYS):
            new_sd[name] = param
            continue

        m = _LAYER_IDX_RE.search(name)
        if m is None:
            new_sd[name] = param
            continue

        layer_idx = int(m.group(1))
        if layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention":
            new_sd[name] = param
        else:
            n_dropped += 1

    if n_dropped:
        logger.warning(
            f"Dropped {n_dropped} attention weight(s) from sliding-attention layers "
            "(identified by layer index via model_config.layer_types). "
            "Those layers will retain randomly initialised attention weights."
        )
    return new_sd


serialization.register_adapter_step(
    _architecture_name, "filter_sliding_attn", _filter_sliding_attn_weights
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "filter_sliding_attn", "hf_to_fms_rope", "weight_fusion"],
)
