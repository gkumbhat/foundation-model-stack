import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple
from typing_extensions import Unpack

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
)

from fms.modules.attention import AttentionKwargs

from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig

from fms.modules.layernorm import LayerNormParameterized
from fms.models.mistral import MistralConfig, Mistral
from fms.models.pixtral_vision import PixtralVisionConfig

logger = logging.getLogger(__name__)


@dataclass
class Mistral3Config(ModelConfig):
    """
    Composite configuration for the FMS Mistral3 multimodal model.

    This wraps a Mistral (text) config for Mistral3 - Pixtral is not added yet.
    Fields default to the standard HF Mistral3 settings unless overridden.
    """
    text_config: MistralConfig = field(default_factory=MistralConfig)
    vision_config: PixtralVisionConfig = field(default_factory=PixtralVisionConfig)
    projector_hidden_act: str = "gelu"
    multimodal_projector_bias: bool = False
    spatial_merge_size: int = 2
    image_token_index: int = 10
    vision_feature_layer: int | list[int] = -1
    tie_heads: bool = False
    ### FMS Specific
    fused_weights: bool = True  # True For CPU/GPU = T, False for AIU


_24b_config = Mistral3Config()


class Mistral3(nn.Module):
    def __init__(
        self,
        config: Optional[Mistral3Config] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()

        if config is not None:
            self.config = config
        else:
            self.config = Mistral3Config()

        self.config = self.config.updated(**kwargs)

        if not self.config.fused_weights:
            self.config.text_config.fused_weights = False

        self.distributed_strategy = distributed_strategy

        # Currently, we always use mistral for the LLM
        self.language_model = Mistral(
            self.config.text_config, self.distributed_strategy
        )

    @classmethod
    def from_config(cls, config: Mistral3Config) -> "Mistral3":
        return cls(config)

    def get_config(self) -> ModelConfig:
        return self.config.text_config

    def reset_parameters(self):
        self.language_model.reset_parameters()

    def post_init(self):
        # Language model post init will handle head tying etc.
        self.language_model.post_init()

    def prepare_inputs_for_generation(
        self,
        iteration: int,
        input_ids: torch.Tensor,
        kwargs: dict[str, Any],
    ):
        # NOTE: This is written to be compatible with Transformers, which
        # is how we should handle preprocessing here; not mistral-commons
        pixel_values = kwargs.get("pixel_values", None)
        image_sizes = kwargs.get("image_sizes", None)
        input_embeds = kwargs.get("inputs", None)

        embeds = self._get_text_embeddings(input_ids, input_embeds)
        img_features = self._get_image_features(pixel_values, image_sizes)
        if img_features is not None:
            embeds = self._merge_multimodal_embeddings(
                input_ids,
                embeds,
                img_features,
                dtype=img_features.dtype,
                device=img_features.device
            )
        return embeds, kwargs

    def _get_text_embeddings(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
    ):
        # Precomputed embeddings take priority over input IDs
        if input_embeds is not None:
            return input_embeds
        return self.language_model.base_model.embedding(input_ids)

    def _get_image_features(
        self,
        pixel_values: torch.Tensor | None,
        image_sizes: torch.Tensor | None,
    ):
        if pixel_values is not None:
            raise NotImplementedError("Image encoder not implemented")
        return None

    def _merge_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        img_features: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(text_embeds).to(device)
        image_features = img_features.to(device, dtype)
        return text_embeds.masked_scatter(special_image_mask, image_features)

    def forward(
        self,
        input_ids_or_embeds: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        use_cache: bool = False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        outputs = self.language_model(
            input_ids_or_embeds,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            use_cache=use_cache,
            **attn_kwargs,
        )
        return outputs


_architecture_name = "mistral3"


def _mistral3_factory_factory(config):
    def factory(**kwargs):
        return Mistral3(config, **kwargs)

    return factory


models.register_model(_architecture_name, "24b", _mistral3_factory_factory(_24b_config))


# =============== Serialization ==================


def _weight_fusion(
    input_sd: Mapping[str, Any], model_config: Optional[Mistral3Config] = None, **kwargs
) -> Mapping[str, Any]:
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


def _hf_gptq_mistral3_check(
    input_sd: Mapping[str, Any], model_config: Optional[MistralConfig] = None, **kwargs
) -> Mapping[str, Any]:
    model_config = model_config.text_config  # type: ignore[union-attr]
    has_fused_weights = True
    linear_type = "torch_linear"
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]

    if "gptq" in linear_type and has_fused_weights:
        raise ValueError(
            "GPTQ HF mistral3 checkpoints cannot be loaded into a model with fused weights"
        )

    return input_sd


serialization.register_adapter_step(
    _architecture_name, "hf_gptq_fusion_check", _hf_gptq_mistral3_check
)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = replacements = [
        # Language Model
        (r"^language_model.lm_head.weight", "language_model.head.weight"),
        (
            r"^language_model.model.embed_tokens.weight",
            "language_model.base_model.embedding.weight",
        ),
        (r"^language_model.model.norm", "language_model.base_model.dec_norm"),
        (r"^language_model.model.layers", "language_model.base_model.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
        # Vision Model
        (r"feed_forward\.gate_proj", "ff_sub_layer.wg"),
        (r"feed_forward\.up_proj", "ff_sub_layer.w1"),
        (r"feed_forward\.down_proj", "ff_sub_layer.w2"),
        (r"attention\.k_proj", "attn.in_proj.key"),
        (r"attention\.v_proj", "attn.in_proj.value"),
        (r"attention\.q_proj", "attn.in_proj.query"),
        (r"attention\.o_proj", "attn.dense"),
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


def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[MistralConfig] = None, **kwargs
) -> Mapping[str, Any]:
    new_sd = {}
    model_config = model_config.text_config  # type: ignore[union-attr]
    if model_config:
        head_size = model_config.head_dim
        linear_type = "torch_linear"
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]
    else:
        logger.warning("Missing model_config, assuming defaults for head_size")
        head_size = 128  # Good default for most models
        linear_type = "torch_linear"

    rope_params = _get_rope_params(linear_type)
    trans_required_pattern = re.compile(
        f"language_model.base_model.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})"
    )
    for name, param in input_sd.items():
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
        # that HF does from the original Meta weights:
        if bool(trans_required_pattern.search(name)):
            temp = param
            if "gptq" in linear_type and temp.dim() == 2:
                # GPTQ qweights are [in_feat, out_feat] (unlike usual [out_feat, in_feat])
                # and are fully transposed before & after process
                temp = temp.transpose(0, 1)
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            num_heads = temp.size(0) // head_size

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # bias
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            if "gptq" in linear_type and temp.dim() == 2:
                temp = temp.transpose(0, 1)

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "hf_gptq_fusion_check", "weight_fusion"],
)
