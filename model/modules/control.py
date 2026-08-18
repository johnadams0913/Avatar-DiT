import torch
import torch.nn as nn

from typing import Tuple
from einops import rearrange
from model.util import checkpoint_wrapper, zero_module
from .layer import MemoryEfficientAttention, sequential_downsample_3d, rope_params


class ControlTransformer(nn.Module):
    def __init__(
            self,
            in_dim,
            dim,
            out_dim,
            mlp_ratio = 1,
            dim_head = 256,
            n_layers = 5,
            patch_size = (1, 2, 2),
            causal = False,
    ):
        super().__init__()
        self.patch_embed = sequential_downsample_3d(in_dim, dim, patch_size)
        self.transformer = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                MemoryEfficientAttention(
                    dim, heads=dim//dim_head, dim_head=dim_head, causal=causal, qk_norm=True
                ),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * mlp_ratio),
                    nn.GELU(),
                    nn.Linear(dim * mlp_ratio, dim)
                )
            ]) for _ in range(n_layers)
        ])
        self.zero_layer = zero_module(nn.Linear(dim, out_dim))
        self.freqs = torch.cat([
            rope_params(1024, dim_head - 4 * (dim_head // 6)),
            rope_params(1024, 2 * (dim_head // 6)),
            rope_params(1024, 2 * (dim_head // 6))
        ], dim=1)
        # self.zero_layer = zero_module(nn.Conv3d(dim, out_dim, kernel_size=1, padding=1))

    @checkpoint_wrapper
    def forward(self, x):
        x = self.patch_embed(x)
        self.freqs = self.freqs.to(x.device)

        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> b (f h w) c')
        grid_sizes = (f, h, w)
        for norm1, attn, norm2, mlp in self.transformer:
            x = x + attn(norm1(x), freqs=self.freqs, grid_sizes=grid_sizes)
            x = x + mlp(norm2(x))
        # x = rearrange(x, 'b (h w f) c -> b c f h w', h=h, w=w)
        return self.zero_layer(x)


class FaceControlTransformer(nn.Module):
    def __init__(
            self,
            dim,
            in_dim: int,
            out_dim: int,
            in_dim_2: int = None,
            mlp_ratio: int = 1,
            dim_head: int = 256,
            n_layers: int = 5,
            patch_size: Tuple[int] = (1, 2, 2),
            causal: bool = False,
    ):
        super().__init__()
        self.patch_embed = sequential_downsample_3d(in_dim, dim, patch_size)
        self.patch_embed_face = sequential_downsample_3d(in_dim_2, dim, patch_size)

        self.transformer = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                MemoryEfficientAttention(
                    dim, heads=dim//dim_head, dim_head=dim_head, causal=causal, qk_norm=True
                ),
                nn.LayerNorm(dim),
                MemoryEfficientAttention(
                    dim, heads=dim//dim_head, dim_head=dim_head, causal=causal, qk_norm=True
                ),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * mlp_ratio),
                    nn.GELU(),
                    nn.Linear(dim * mlp_ratio, dim)
                )
            ]) for _ in range(n_layers)
        ])
        self.zero_layer = zero_module(nn.Linear(dim, out_dim))
        self.freqs = torch.cat([
            rope_params(1024, dim_head - 4 * (dim_head // 6)),
            rope_params(1024, 2 * (dim_head // 6)),
            rope_params(1024, 2 * (dim_head // 6))
        ], dim=1)
        # self.zero_layer = zero_module(nn.Conv3d(dim, out_dim, kernel_size=1, padding=1))

    @checkpoint_wrapper
    def forward(self, x, x_face=None):
        self.freqs = self.freqs.to(x.device)

        x = self.patch_embed(x)
        x_f = self.patch_embed_face(x_face)
        b, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> b (f h w) c")
        x_f = rearrange(x_f, "b c f h w -> b (f h w) c")

        grid_sizes = (f, h, w)

        for norm1, attn, norm2, attn_2, norm3, mlp in self.transformer:
            x = x + attn(norm1(x), freqs=self.freqs, grid_sizes=grid_sizes)
            x = x + attn_2(norm2(x), context=x_f, freqs=self.freqs, grid_sizes=grid_sizes)
            x = x + mlp(norm3(x))
        # x = rearrange(x, 'b (h w f) c -> b c f h w', h=h, w=w)
        return self.zero_layer(x)


class TemporalCausalTransformer(nn.Module):
    def __init__(
            self,
            in_dim,
            dim,
            out_dim,
            mlp_ratio = 1,
            dim_head = 256,
            n_layers = 5,
            patch_size = (1, 2, 2),
    ):
        super().__init__()
        self.patch_embed = sequential_downsample_3d(in_dim, dim, patch_size)
        self.transformer = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                MemoryEfficientAttention(dim, heads=dim//dim_head, dim_head=dim_head, causal=True, rope=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * mlp_ratio),
                    nn.GELU(),
                    nn.Linear(dim * mlp_ratio, dim)
                )
            ) for _ in range(n_layers)
        ])
        self.zero_layer = zero_module(nn.Linear(dim, out_dim))
        # self.zero_layer = zero_module(nn.Conv3d(dim, out_dim, kernel_size=1, padding=1))

    @checkpoint_wrapper
    def forward(self, x):
        x = self.patch_embed(x)

        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b h w) f c')
        for layer in self.transformer:
            x = x + layer(x)
        x = rearrange(x, '(b h w) f c -> b (f h w) c', h=h, w=w)
        return self.zero_layer(x)