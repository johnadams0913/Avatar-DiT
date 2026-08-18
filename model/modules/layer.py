import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util import exists, default, checkpoint_wrapper, zero_module, rank_zero_print
from einops import rearrange

ATTN_PRECISION = torch.float16

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
    FLASH_ATTN_AVAILABLE = False
    XFORMERS_IS_AVAILBLE = False
    rank_zero_print("Flash attn 3 is available")
    causal_ops = {"causal": True}

except ModuleNotFoundError:
    try:
        import flash_attn

        FLASH_ATTN_AVAILABLE = True
        rank_zero_print("Flash attn 2 is available")
        causal_ops = {"causal": True}

    except ModuleNotFoundError:
        try:
            import xformers

            XFORMERS_IS_AVAILBLE = True
            causal_ops = {"attn_bias": xformers.ops.LowerTriangularMask()}
            rank_zero_print("XFormers is available")
        except:
            Warning("Flash_attn and XFormers are not available, PyTorch official attention will be used.")
            XFORMERS_IS_AVAILBLE = False
            causal_ops = {"is_causal": True}
        finally:
            FLASH_ATTN_AVAILABLE = False
    finally:
        FLASH_ATTN_3_AVAILABLE = False


def half(x):
    if x.dtype not in [torch.float16, torch.bfloat16]:
        x = x.to(ATTN_PRECISION)
    return x

def attn_processor(q, k, v, attn_mask = None, *args, **kwargs):
    if attn_mask is not None:
        if XFORMERS_IS_AVAILBLE:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=attn_mask, *args, **kwargs
            )
        else:
            q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, *args, **kwargs
            ).transpose(1, 2)
    else:
        if FLASH_ATTN_3_AVAILABLE:
            dtype = v.dtype
            q, k, v = map(lambda t: half(t), (q, k, v))
            out = flash_attn_interface.flash_attn_func(q, k, v, *args, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            out = out.to(dtype)
        elif FLASH_ATTN_AVAILABLE:
            dtype = v.dtype
            q, k, v = map(lambda t: half(t), (q, k, v))
            out = flash_attn.flash_attn_func(q, k, v, *args, **kwargs).to(dtype)
        elif XFORMERS_IS_AVAILBLE:
            out = xformers.ops.memory_efficient_attention(q, k, v, *args, **kwargs)
        else:
            q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, *args, **kwargs).transpose(1, 2)
    return out


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

    
def rope(x, grid_sizes, freqs):
    bs, n, c = x.size(0), x.size(2), x.size(3) // 2
    f, h, w = grid_sizes
    seq_len = grid_sizes[0] * grid_sizes[1] * grid_sizes[2]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(bs, seq_len, n, -1, 2))
    freqs_expanded = torch.cat([
        freqs[0][:f].view(1, f, 1, 1, -1).expand(bs, f, h, w, -1),
        freqs[1][:h].view(1, 1, h, 1, -1).expand(bs, f, h, w, -1),
        freqs[2][:w].view(1, 1, 1, w, -1).expand(bs, f, h, w, -1)
    ], dim=-1).reshape(bs, seq_len, 1, -1)

    x_rotated = torch.view_as_real(x_complex * freqs_expanded).flatten(3)
    return x_rotated.to(x.dtype)


def rope_2d(x, grid_sizes, freqs):
    bs, n, c = x.size(0), x.size(2), x.size(3) // 2
    h, w = grid_sizes
    seq_len = grid_sizes[0] * grid_sizes[1]
    freqs = freqs.split([c // 2, c // 2], dim=1)

    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(bs, seq_len, n, -1, 2))
    freqs_expanded = torch.cat([
        freqs[0][:h].view(1, h, 1, -1).expand(bs, h, w, -1),
        freqs[1][:w].view(1, 1, w, -1).expand(bs, h, w, -1)
    ], dim=-1).reshape(bs, seq_len, 1, -1)

    x_rotated = torch.view_as_real(x_complex * freqs_expanded).flatten(3)
    return x_rotated.to(x.dtype)


def rope_1d(x, freqs):
    bs, seq_len, heads = x.shape[:3]

    x_complex = torch.view_as_complex(
        x.float().reshape(bs, seq_len, heads, -1, 2)
    )

    # Simply use the first seq_len frequencies directly
    freqs = freqs[:seq_len].view(1, seq_len, 1, -1)

    x_out = x_complex * freqs
    x_out = torch.view_as_real(x_out).flatten(3)

    return x_out.type_as(x)

@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes=None, freqs=None):
    """ 3D RoPE from WAN """
    if not exists(grid_sizes):
        return x
    elif len(grid_sizes) == 3:
        x = rope(x, grid_sizes, freqs)
    elif len(grid_sizes) == 2:
        x = rope_2d(x, grid_sizes, freqs)
    elif len(grid_sizes) == 1:
        x = rope_1d(x, freqs)
    return x


def sequential_downsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(16, 32, 3, padding=1, stride=2),
        nn.SiLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(32, 96, 3, padding=1, stride=2),
        nn.SiLU(),
        nn.Conv2d(96, 96, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(96, 256, 3, padding=1, stride=2),
        nn.SiLU(),
        zero_module(nn.Conv2d(256, out_channels, 3, padding=1))
    )


def sequential_downsample_3d(in_dim, dim, patch_size):
    return nn.Sequential(
        nn.Conv3d(in_dim, 32, 3, padding=1),
        nn.SiLU(),
        nn.Conv3d(32, 32, 3, padding=1),
        nn.SiLU(),
        nn.Conv3d(32, 64, 3, padding=(0, 1, 1), stride=(1, 2, 2)),
        nn.SiLU(),
        nn.Conv3d(64, 64, 3, padding=1),
        nn.SiLU(),
        nn.Conv3d(64, 256, 3, padding=1, stride=2),
        nn.SiLU(),
        nn.Conv3d(256, 256, 3, padding=1),
        nn.SiLU(),
        nn.Conv3d(256, 512, 3, padding=1, stride=2),
        nn.SiLU(),
        nn.Conv3d(512, dim, 3, padding=1),
        nn.SiLU(),
        zero_module(nn.Conv3d(dim, dim, kernel_size=patch_size, stride=patch_size))
    )

def sequential_downsample_1d(in_dim, dim):
    return nn.Sequential(
        nn.Conv1d(in_dim, 128, 3, padding=1),
        nn.SiLU(),
        nn.Conv1d(128, 128, 3, padding=1),
        nn.SiLU(),
        nn.Conv1d(128, 256, 3, padding=1, stride=2),
        nn.SiLU(),
        nn.Conv1d(256, 256, 3, padding=1),
        nn.SiLU(),
        nn.Conv1d(256, 512, 3, padding=1, stride=2),
        nn.SiLU(),
        nn.Conv1d(512, dim, 3, padding=1),
    )


class MemoryEfficientAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
            self,
            query_dim,
            context_dim = None,
            heads = None,
            dim_head = 64,
            dropout = 0.0,
            log = False,
            causal = False,
            qk_norm = False,
            **kwargs
    ):
        super().__init__()
        if log:
            rank_zero_print(
                f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
                f"{heads} heads.")

        heads = heads or query_dim // dim_head
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.qk_norm = qk_norm
        self.attn_ops = causal_ops if causal else {}

        if self.qk_norm:
            self.norm_q = nn.LayerNorm(inner_dim)
            self.norm_k = nn.LayerNorm(inner_dim)

        self.bg_scale = 1.
        self.fg_scale = 1.
        self.merge_scale = 0.
        self.mask_threshold = 0.05

    @checkpoint_wrapper
    def forward(self, x, context = None, scale = 1., *args, **kwargs):
        context = default(context, x)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
        out = self.attn_forward(q, k, v, scale, *args, **kwargs)

        return self.to_out(out)

    def attn_forward(self, q, k, v, scale = 1., freqs = None, grid_sizes=None, *args, **kwargs):
        q, k, v = map(
            lambda t: rearrange(t, "b n (h c) -> b n h c", h=self.heads),
            (q, k, v)
        )
        
        q, k = map(lambda t: rope_apply(t, grid_sizes, freqs), (q, k))
        out = attn_processor(q, k, v, **self.attn_ops) * scale
        out = rearrange(out, "b n h c -> b n (h c)", h=self.heads)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, *args, **kwargs):
        super().__init__()
        in_dim = in_dim
        out_dim = default(out_dim, in_dim)
        self.residual = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1),
        )
        self.shortcut = nn.Conv3d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
    
    @checkpoint_wrapper
    def forward(self, x):
        h = self.shortcut(x)
        x = self.residual(x)
        return x + h