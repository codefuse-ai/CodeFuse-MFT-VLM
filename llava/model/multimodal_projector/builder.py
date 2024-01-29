import torch
import torch.nn as nn
import re
from functools import partial
from ..multimodal_encoder.visual import Resampler
import math


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class VLCrossAttention(nn.Module):
    def __init__(self, config, vision_tower):
        super().__init__()
        n_queries = 256
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_pool = Resampler(
            grid_size=int(math.sqrt(n_queries)),
            embed_dim=config.hidden_size,
            num_heads=config.hidden_size // 128,
            kv_dim=vision_tower.hidden_size,
            norm_layer=norm_layer,
        )
        self.ln_post = norm_layer(config.hidden_size)
        self.proj = nn.Parameter((config.hidden_size** -0.5) * torch.randn(config.hidden_size, config.hidden_size))
        
    def forward(self, x):
        x = self.attn_pool(x)
        x = self.ln_post(x)
        x = x @ self.proj

        return x


def build_vision_projector(config, delay_load=False, vision_tower=None, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    print("PROJECTOR TYPE: ", projector_type)

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    elif "cross_attn" in projector_type:
        vl_cross_attn = VLCrossAttention(config, vision_tower)
        return vl_cross_attn


    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')