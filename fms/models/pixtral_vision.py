import logging
from dataclasses import dataclass

from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.utils.config import ModelConfig
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
)
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils.activation import str_to_activation

import torch
from torch import nn
from typing import Any, Unpack


logger = logging.getLogger(__name__)

@dataclass
class PixtralVisionConfig(ModelConfig):
    # Identical configuration to the vision encoder in
    # mistralai/Mistral-Small-3.2-24B-Instruct-2506
    hidden_size: int = 1024
    intermediate_size: int = 4096
    nlayers: int = 24
    nheads: int = 16
    nchannels: int = 3
    image_size: int = 1540
    patch_size: int = 14
    hidden_act: str = "silu"
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    # FMS specific
    linear_config: dict[str, Any] | None = None
    fused_weights: bool = True

class PixtralRMSNorm(LayerNormParameterized):
    """Pixtral's RMS Norm using the FMS implementation of LayerNorm."""
    def __init__(self, normalized_shape: int, eps: float):
        super().__init__(
            normalized_shape,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=eps,
            use_high_precision_pow=True,
        )

class PixtralAttentionLayer(nn.Module):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.config = config
        head_dim = self.config.hidden_size // self.config.nheads
        mlp_grow_factor = self.config.intermediate_size / self.config.hidden_size

        # Attention related
        self.attention_norm = PixtralRMSNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.layer_norm_eps,
        )
        self.attn = MultiHeadAttention(
            emb_dim=self.config.hidden_size,
            emb_kq=head_dim,
            emb_v=head_dim,
            nheads=config.nheads,
            kvheads=config.nheads,
            p_dropout=self.config.attention_dropout,
            linear_config=self.config.linear_config,
            fused=self.config.fused_weights,
            scale_factor=head_dim**-0.5,
        )

        # Feedforward related
        self.ffn_norm = PixtralRMSNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.layer_norm_eps,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.hidden_size,
            hidden_grow_factor=mlp_grow_factor,
            activation_fn=str_to_activation(self.config.hidden_act),
            use_bias=False,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attn(q=hidden_states, **attn_kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ff_sub_layer(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def reset_parameters(self):
        for m in self.modules():
            m.reset_parameters()

class PixtralTransformer(nn.Module):
    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [PixtralAttentionLayer(config) for _ in range(config.nlayers)]
        )

    def forward(
        self,
        inputs_embeds,
        output_hidden_states=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        # TODO: Currently aligns with siglip to return an empty tuple
        # when there are no hidden states, but None would make more sense
        # here and better align with HF Transformers for readability.
        encoder_states = ()

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
            hidden_states = encoder_layer(hidden_states, **attn_kwargs)

        return hidden_states, encoder_states

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

class PixtralVisionModel(nn.Module):
    def __init__(
        self,
        config: PixtralVisionConfig | None = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = PixtralVisionConfig()
        self.config = self.config.updated(**kwargs)
        self.patch_size = self.config.patch_size

        self.distributed_strategy = distributed_strategy
        self.patch_conv = nn.Conv2d(
            in_channels=self.config.nchannels,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
            padding="valid",
        )
        self.ln_pre = PixtralRMSNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.layer_norm_eps,
        )
        self.transformer = PixtralTransformer(self.config)
        self.patch_positional_embedding = None # Gaurav porting 2D rope
        

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor | list[tuple[int, int]] | None=None,
        output_hidden_states=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        # TODO - do we really need this?
        attn_kwargs["attn_name"] = attn_kwargs.get("attn_name", "sdpa_bidirectional")

        if image_sizes is None:
            batch_size, _, height, width = pixel_values.shape
            image_sizes = [(height, width)] * batch_size
        # Pass images through initial convolution independently + flatten
        patch_embeds = self.patch_conv(pixel_values)
        # TODO: Check / fix potential graph break in positional encoding
        patch_embeds_list = [
            embed[..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)]
            for embed, size in zip(patch_embeds, image_sizes)
        ]

        patch_embeds = torch.cat([p.flatten(1).T for p in patch_embeds_list], dim=0).unsqueeze(0)
        patch_embeds = self.ln_pre(patch_embeds)

        # Positional embedding (TODO)
        position_embeddings = patch_embeds
        
        # Invoke the actual transformer
        return self.transformer(
            position_embeddings,
            output_hidden_states=output_hidden_states,
            **attn_kwargs,
        )

    # TODO - probably should post init positions here maybe

    def reset_parameters(self):
        self.transformer.reset_parameters()

# NOTE: We do not currently offer support for Pixtral as a standalone
# vision encoder, as this model is largely used in the composite
# architecture for Mistral3 within FMS. While there are standalone vision
# models for pixtral, they primarily use Mistral's model format instead
# of HF Transformers, and would need to be converted for direct use in FMS.
#
# If the need for pixtral to run as a standalone vision encoder is pressing
# in the future, we can take the normal pattern and add a factory for it,
# then port the vision parts of the adapters from Mistral3 here.
