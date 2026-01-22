import logging
from dataclasses import dataclass

from fms.utils.config import ModelConfig
from typing import Any


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
    patch_size: int = 16
    hidden_act: str = "silu"
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    linear_config: dict[str, Any] | None = None
    fused_weights: bool = True
    head_dim: int = 64
    initializer_range: float = 0.02
