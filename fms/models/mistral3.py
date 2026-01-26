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


# Patch Merger and Projector are largely derived
# from Transformers (v4) implementation.
class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config

        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2,
            hidden_size,
            bias=False,
        )

    def forward(
        self, image_features: torch.Tensor, image_sizes: torch.Tensor
    ) -> torch.Tensor:
        img_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size)
            for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in img_sizes]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(
            image_features.split(tokens_per_image)
        ):
            # Reshape image_tokens into a 2D grid
            h, w = img_sizes[image_index]
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            # NOTE (Alex / Gaurav) check if unfold is compatible with AIU compile
            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        image_features = self.merging_layer(image_features)
        return image_features

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.merging_layer.weight)


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config
        self.patch_merger = Mistral3PatchMerger(config)
        self.norm = LayerNormParameterized(
            self.config.vision_config.hidden_size,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.text_config.norm_eps,
            use_high_precision_pow=True,
        )

        self.config = config
        num_feature_layers = (
            1
            if isinstance(config.vision_feature_layer, int)
            else len(config.vision_feature_layer)
        )
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.emb_dim,
            bias=config.multimodal_projector_bias,
        )
        self.act = str_to_activation(config.projector_hidden_act)
        self.linear_2 = nn.Linear(
            config.text_config.emb_dim,
            config.text_config.emb_dim,
            bias=config.multimodal_projector_bias,
        )

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor):
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        if self.config.multimodal_projector_bias:
            nn.init.normal_(self.linear_1.bias, std=1e-6)
            nn.init.normal_(self.linear_2.bias, std=1e-6)


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
        # Vision encoder and projector for multimodal features
        self.multi_modal_projector = Mistral3MultiModalProjector(
            self.config,
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

        # Only consider image features at decode time
        if iteration == 0:
            img_features = self._get_image_features(pixel_values, image_sizes)
            if img_features is not None:
                embeds = self._merge_multimodal_embeddings(
                    input_ids,
                    embeds,
                    img_features,
                    dtype=embeds.dtype,
                    device=embeds.device,
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
        raise NotImplementedError("vision tower has not been implemented yet.")
        # NOTE: Below is WIP and has been tested with stubbed inputs from HF's
        # Encoder, so it should be (mostly) functionally correct.
        _, _, image_features = self.vision_tower(
            pixel_values, output_hidden_states=True
        )

        # Handle multiple vision feature layers
        if isinstance(self.config.vision_feature_layer, int):
            selected_image_feature = image_features[self.config.vision_feature_layer]
        else:
            hs_pool = [
                image_features[layer_idx]
                for layer_idx in self.config.vision_feature_layer
            ]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        # Run the multimodal projector on selected features
        image_features = self.multi_modal_projector(selected_image_feature, image_sizes)

        # Split out the stacked image features
        downsample_ratio = (
            self.config.vision_config.patch_size * self.config.spatial_merge_size
        )
        split_sizes = [
            (height // downsample_ratio) * (width // downsample_ratio)
            for height, width in image_sizes
        ]
        image_features = torch.split(image_features.squeeze(0), split_sizes)
        image_features = torch.cat(image_features, dim=0)
        return image_features

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
