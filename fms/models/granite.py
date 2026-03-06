import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple
from typing_extensions import Unpack

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    get_attention_type,
)
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.linear import get_linear_type
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig
from fms.utils.headless import gather_outputs


logger = logging.getLogger(__name__)


@dataclass
class GraniteConfig(ModelConfig):
    src_vocab_size: int = 100352  # can be set by tokenizer
    emb_dim: int = 2560
    norm_eps: float = 1e-5
    nheads: int = 40
    head_dim: int = 64  # getattr(config, "head_dim", emb_dim // nheads)
    kvheads: int = 8
    nlayers: int = 40
    pad_id: int = 100256
    hidden_grow_factor: float = 8192/ 2560
    multiple_of: int = 256
    activation_fn: str = "swish"
    p_dropout: float = 0.0
    max_expected_seq_len: int = 8192
    ntk_scaling: bool = False
    attn_bias: bool = False
    mlp_bias: bool = False
    tie_heads: bool = True
    rope_theta: float = 10_000.0
    embedding_multiplier: float = 1.0
    logits_scaling: float = 1.0
    residual_multiplier: float = 1.0
    attention_multiplier: float = 1.0
    linear_config: Optional[Mapping[str, Any]] = None
    fused_weights: bool = False


class DimensionPadding(nn.Module):
    """Pads embedding dimension to meet Spyre layout requirements"""
    def __init__(self, original_dim: int, padded_dim: int):
        super().__init__()
        self.original_dim = original_dim
        self.padded_dim = padded_dim
        self.padding_size = padded_dim - original_dim

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad last dimension from original_dim to padded_dim"""
        if self.padding_size == 0:
            return x
        # x shape: [..., original_dim]
        padding = torch.zeros(
            *x.shape[:-1], self.padding_size,
            dtype=x.dtype, device=x.device
        )
        return torch.cat([x, padding], dim=-1)

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        """Remove padding from last dimension"""
        if self.padding_size == 0:
            return x
        return x[..., :self.original_dim]

class GraniteBlock(nn.Module):
    def __init__(self, config: GraniteConfig, rotary_emb: RotaryEmbedding):
        super(GraniteBlock, self).__init__()
        self.config = config
        emb_kq = self.config.head_dim
        emb_v = self.config.head_dim

        self.ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )

        if self.config.kvheads == 0:
            kvheads = self.config.nheads
        else:
            kvheads = self.config.kvheads
            assert self.config.nheads % self.config.kvheads == 0

        self.attn = MultiHeadAttention(
            self.config.emb_dim,
            emb_kq,
            emb_v,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=self.config.attn_bias,
            position_encoder=rotary_emb,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
            scale_factor=self.config.attention_multiplier,
        )

        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=self.config.mlp_bias,
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
        # if the cache is not empty, we need to get the kv cache for self and cross attention
        self_attn_past_key_value = past_key_value_state

        # first we do MHA and Add&Norm
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
        # residual connection
        x = x * self.config.residual_multiplier + residual

        # then we do FF and Add&Norm
        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # another residual
        x = x * self.config.residual_multiplier + residual

        if use_cache:
            return (x, cache)
        else:
            return x


class GraniteHeadless(nn.Module):
    def __init__(
        self,
        config: Optional[GraniteConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(GraniteHeadless, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = GraniteConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        # Store original dimension for weight loading
        self.original_emb_dim = 2560
        self.target_emb_dim = 4096
        self.needs_padding = (self.original_emb_dim != self.target_emb_dim)

        # Override config to use target dimension
        if self.needs_padding:
            print(f"⚠️  Model will use emb_dim={self.target_emb_dim} (padded from {self.original_emb_dim})")
            self.config.emb_dim = self.target_emb_dim

        self.width = self.config.emb_dim
        self.pad_id = self.config.pad_id
        self.max_expected_seq_len = self.config.max_expected_seq_len

        # Embedding with TARGET dimension (will be padded during load)
        self.embedding = nn.Embedding(
            self.config.src_vocab_size,
            self.config.emb_dim,  # Use target dimension
            padding_idx=self.config.pad_id,
        )

        # ... rest of initialization remains the same ...

    def _pad_weight_tensor(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Pad a weight tensor along specified dimension from original to target size"""
        if not self.needs_padding:
            return tensor

        padding_size = self.target_emb_dim - self.original_emb_dim

        if dim == 0:  # Pad rows (output dimension)
            padding = torch.zeros(
                padding_size, *tensor.shape[1:],
                dtype=tensor.dtype, device=tensor.device
            )
            return torch.cat([tensor, padding], dim=0)
        elif dim == 1:  # Pad columns (input dimension)
            padding = torch.zeros(
                tensor.shape[0], padding_size, *tensor.shape[2:],
                dtype=tensor.dtype, device=tensor.device
            )
            return torch.cat([tensor, padding], dim=1)
        else:
            raise ValueError(f"Unsupported padding dimension: {dim}")

    def load_state_dict(self, state_dict, strict=True):
        """Override to handle dimension padding during weight loading"""
        if not self.needs_padding:
            return super().load_state_dict(state_dict, strict=strict)

        # Pad weights that need it
        padded_state_dict = {}
        for key, value in state_dict.items():
            if 'embedding.weight' in key:
                # Pad embedding: [vocab_size, original_dim] -> [vocab_size, target_dim]
                padded_state_dict[key] = self._pad_weight_tensor(value, dim=1)
                print(f"Padded {key}: {value.shape} -> {padded_state_dict[key].shape}")

            elif any(x in key for x in ['attn.wq.weight', 'attn.wk.weight', 'attn.wv.weight']):
                # Pad attention input: [head_dim * nheads, original_dim] -> [head_dim * nheads, target_dim]
                padded_state_dict[key] = self._pad_weight_tensor(value, dim=1)
                print(f"Padded {key}: {value.shape} -> {padded_state_dict[key].shape}")

            elif 'attn.wo.weight' in key:
                # Pad attention output: [original_dim, head_dim * nheads] -> [target_dim, head_dim * nheads]
                padded_state_dict[key] = self._pad_weight_tensor(value, dim=0)
                print(f"Padded {key}: {value.shape} -> {padded_state_dict[key].shape}")

            elif any(x in key for x in ['ff_sub_layer', 'w1.weight', 'w2.weight', 'w3.weight']):
                # Pad FFN weights
                if 'w1.weight' in key or 'w3.weight' in key:
                    # Input projection: [intermediate, original_dim] -> [intermediate, target_dim]
                    padded_state_dict[key] = self._pad_weight_tensor(value, dim=1)
                elif 'w2.weight' in key:
                    # Output projection: [original_dim, intermediate] -> [target_dim, intermediate]
                    padded_state_dict[key] = self._pad_weight_tensor(value, dim=0)
                print(f"Padded {key}: {value.shape} -> {padded_state_dict[key].shape}")

            elif any(x in key for x in ['ln.weight', 'ff_ln.weight', 'dec_norm.weight']):
                # Pad layer norm: [original_dim] -> [target_dim]
                padding = torch.ones(
                    self.target_emb_dim - self.original_emb_dim,
                    dtype=value.dtype, device=value.device
                )
                padded_state_dict[key] = torch.cat([value, padding], dim=0)
                print(f"Padded {key}: {value.shape} -> {padded_state_dict[key].shape}")

            else:
                # No padding needed
                padded_state_dict[key] = value

        return super().load_state_dict(padded_state_dict, strict=strict)

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

        if x_in.dim() == 2:  # input is not already embedded
            x_in = self.embedding(x_in)  # Shape: [..., original_emb_dim]

        # PAD EMBEDDINGS
        if self.use_padding:
            x_in = self.dim_padding.pad(x_in)  # Shape: [..., padded_emb_dim]

        x_in = x_in * self.config.embedding_multiplier

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

        # UNPAD BEFORE RETURNING
        if self.use_padding:
            dec_out = self.dim_padding.unpad(dec_out)  # Shape: [..., original_emb_dim]

        return dec_out, present_key_value_states

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.embedding.weight, mean=0.0, std=self.config.emb_dim**-0.5
        )

        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        # Call reset_parameters for relevant sub-layers
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
        # remove meta tensors from cached_freqs
        for dev in list(cached_freqs.keys()):
            for alp in list(cached_freqs[dev].keys()):
                if cached_freqs[dev][alp].device == torch.device("meta"):
                    del cached_freqs[dev][alp]
                    if len(cached_freqs[dev]) == 0:
                        del cached_freqs[dev]
                        del max_seq_len_cached[dev]

    def post_init(self):
        # This function is called in `get_model` after the model is
        # fully initalized on the correct device

        self._clean_up_rot_emb_cache(
            self.rot_emb.cached_freqs,
            self.rot_emb.max_seq_len_cached,
        )

        # init RoPE on the right device(s)
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)




class Granite(nn.Module):
    def __init__(
        self,
        config: Optional[GraniteConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(Granite, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = GraniteConfig()
        self.config = self.config.updated(**kwargs)

        print("=================================")
        print(self.config)
        print("=================================")
        self.distributed_strategy = distributed_strategy

        self.base_model = GraniteHeadless(self.config, self.distributed_strategy)

        # Head uses target dimension (same as base_model after padding)
        self.head = nn.Linear(
            self.base_model.config.emb_dim,  # Use padded dimension
            self.config.src_vocab_size,
            bias=False
        )

    def load_state_dict(self, state_dict, strict=True):
        """Override to handle head weight padding"""
        if self.base_model.needs_padding:
            padded_state_dict = {}
            for key, value in state_dict.items():
                if 'head.weight' in key:
                    # Pad head: [vocab_size, original_dim] -> [vocab_size, target_dim]
                    padded_state_dict[key] = self.base_model._pad_weight_tensor(value, dim=1)
                    print(f"Padded {key}: {value.shape} -> {padded_state_dict[key].shape}")
                else:
                    padded_state_dict[key] = value
            state_dict = padded_state_dict

        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_config(cls, config: GraniteConfig) -> "Granite":
        return cls(config)

    def get_config(self) -> GraniteConfig:
        return self.config

    def reset_parameters(self):
        self.head.weight.data.normal_(
            0,
            1 / math.sqrt(math.sqrt(self.config.emb_dim * self.config.src_vocab_size)),
        )
        self.base_model.reset_parameters()

    def post_init(self):
        # if this model ties weights, they are tied here
        if self.config.tie_heads:
            # handle assignment of non-meta weights to meta parameters
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
            x,
            position_ids,
            past_key_value_states,
            use_cache,
            **attn_kwargs,
        )

        output = gather_outputs(output, last_n_tokens, **attn_kwargs)
        preds = self.head(output)
        preds = preds / self.config.logits_scaling

        if use_cache:
            return preds, cache
        else:
            return preds


_8b_config = GraniteConfig(
    src_vocab_size=49155,
    emb_dim=4096,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=40,
    hidden_grow_factor=12800 / 4096,
    max_expected_seq_len=8192,
    rope_theta=10_000.0,
    pad_id=0,
    p_dropout=0.0,  # overwriting config.json
    tie_heads=True,
    embedding_multiplier=12.0,
    logits_scaling=16.0,
    residual_multiplier=0.22,
    attention_multiplier=0.0078125,
)

_3_1_2b_config = GraniteConfig(
    src_vocab_size=49155,
    emb_dim=2048,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=40,
    hidden_grow_factor=8192 / 2048,
    max_expected_seq_len=131072,
    rope_theta=5000000.0,
    pad_id=0,
    p_dropout=0.0,
    tie_heads=True,
    embedding_multiplier=12.0,
    logits_scaling=8.0,
    residual_multiplier=0.22,
    attention_multiplier=0.015625,
)

_architecture_name = "granite"


def _granite_factory_factory(config):
    def factory(**kwargs):
        return Granite(config, **kwargs)

    return factory


models.register_model(_architecture_name, "8b", _granite_factory_factory(_8b_config))
models.register_model(
    _architecture_name, "3_1_2b", _granite_factory_factory(_3_1_2b_config)
)


def _weight_fusion(
    input_sd: Mapping, model_config: Optional[GraniteConfig] = None, **kwargs
):
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        new_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(
            serialization._attn_unfused_to_fused_step(new_sd)
        )
    return new_sd


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"^lm_head.weight", "head.weight"),
        (r"^model.embed_tokens.weight", "base_model.embedding.weight"),
        (r"^model.norm", "base_model.dec_norm"),
        (r"^model.layers", "base_model.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)


serialization.register_adapter_step(
    _architecture_name,
    "weight_expansion_for_mismatched_head_dim",
    serialization._weight_expansion_for_mismatched_head_dim,  # type: ignore[arg-type]
)


def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    elif "int8" in linear_type:
        # quantize_weight is fms-model-optimizer identifier of weight clip values
        return ["weight", "bias", "quantize_weight"]
    elif "fp8" in linear_type:
        return ["weight", "weight_scale", "input_scale", "bias"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[GraniteConfig] = None, **kwargs
) -> Mapping[str, Any]:
    new_sd = {}

    if model_config:
        head_size = model_config.emb_dim // model_config.nheads
    else:
        logger.warning("Missing model_config, assuming defaults for head_size")
        head_size = 128  # Good default for most models

    for name, param in input_sd.items():
        # Some checkpoints have weights in different precisions, which can have
        # auxiliary tensors (see _get_rope_params e.g. gptq, fp8).
        # Thus, we need to get rope_params per parameter.
        linear_type_str = "torch_linear"
        if model_config and model_config.linear_config:
            linear_type_str = get_linear_type(
                model_config.linear_config,
                module_name=name,
            )
        rope_params = _get_rope_params(linear_type_str)
        trans_required_pattern = re.compile(
            f"base_model.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})$"
        )

        # hf -> fms requires a transpose operation for the query and key
        # weight and bias parameters for Llama models
        # This transpose is due to the different implementation of RoPE in
        # HF and FMS. While FMS follows the original RoPE paper
        # (https://arxiv.org/abs/2104.09864), HF has its own implementation
        # that doesn't respect the order of outputs. This is OK as long as you
        # rearrange the weights of the query and key projections, as the
        # combination projection + RoPE ends up producing the same outputs.
        # Therefore, to make FMS produce the correct order of outputs when
        # loading from an HF checkpoint, we need to undo the transformation
        # that HF does from the original Meta weights
        is_gptq_2d_qparam = "gptq" in linear_type_str and param.dim() == 2
        if bool(trans_required_pattern.match(name)) and param.numel() > 1:
            temp = param
            if is_gptq_2d_qparam:
                # GPTQ qweights are [in_feat, out_feat] (unlike usual [out_feat, in_feat])
                # and are fully transposed before & after process.
                # GPTQ scales and qzeros are also transposed accordingly
                temp = temp.transpose(0, 1)
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            num_heads = temp.size(0) // head_size

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # 1-dim parameters
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            if is_gptq_2d_qparam:
                temp = temp.transpose(0, 1)

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


def _hf_gptq_granite_check(
    input_sd: Mapping[str, Any], model_config: Optional[GraniteConfig] = None, **kwargs
) -> Mapping[str, Any]:
    has_fused_weights = True
    linear_type = "torch_linear"
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]

    if not callable(linear_type) and "gptq" in linear_type and has_fused_weights:
        raise ValueError(
            "GPTQ HF granite checkpoints cannot be loaded into a model with fused weights"
        )

    return input_sd


serialization.register_adapter_step(
    "granite", "hf_gptq_fusion_check", _hf_gptq_granite_check
)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    [
        "hf_to_fms_names",
        "hf_to_fms_rope",
        "hf_gptq_fusion_check",
        "weight_fusion",
    ],
)
