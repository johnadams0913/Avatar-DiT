# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from einops import  rearrange
from typing import List
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import PeftAdapterMixin

from ..util import exists, checkpoint_wrapper, default
from .motion_encoder import Generator
from .face_blocks import FaceAdapter, FaceEncoder
from .attention import flash_attention
from .layer import rope_params


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes=None, freqs=None, temporal_offset=0):
    if not exists(grid_sizes):
        return x

    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        t_off = temporal_offset if isinstance(temporal_offset, int) else int(temporal_offset[i])
        freqs_i = torch.cat([
            freqs[0][t_off:t_off+f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @checkpoint_wrapper
    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    @checkpoint_wrapper
    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    @checkpoint_wrapper
    def forward(self, x, seq_lens, grid_sizes, freqs, temporal_offset=0,
                kv_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            temporal_offset(int): Offset for temporal RoPE positions (for AR chunks).
            kv_cache(dict|None): Pre-allocated global KV cache for AR generation.
                When provided: writes K,V in-place, reads full history for attention.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        q = rope_apply(q, grid_sizes, freqs, temporal_offset)
        k = rope_apply(k, grid_sizes, freqs, temporal_offset)

        if kv_cache is not None:
            num_new = seq_lens.max().item()
            end_idx = kv_cache['end_index'].item()
            kv_cache['k'][:, end_idx:end_idx + num_new] = k[:, :num_new].detach()
            kv_cache['v'][:, end_idx:end_idx + num_new] = v[:, :num_new].detach()
            new_end = end_idx + num_new

            max_attn = kv_cache.get('max_attn_size', new_end)
            read_start = max(0, new_end - max_attn)
            k_full = kv_cache['k'][:, read_start:new_end]
            v_full = kv_cache['v'][:, read_start:new_end]
            k_lens = torch.tensor([new_end - read_start] * b,
                                  dtype=torch.long, device=x.device)
            kv_cache['end_index'].fill_(new_end)

            x = flash_attention(q=q, k=k_full, v=v_full, k_lens=k_lens,
                                window_size=self.window_size)
        else:
            x = flash_attention(q=q, k=k, v=v, k_lens=seq_lens,
                                window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    @checkpoint_wrapper
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    @checkpoint_wrapper
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 checkpoint=True):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.checkpoint = checkpoint

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    @checkpoint_wrapper
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        temporal_offset=0,
        kv_cache=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            temporal_offset(int): Offset for temporal RoPE positions.
            kv_cache(dict|None): Per-layer KV cache dict for AR generation.
        """
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention (with optional KV cache)
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs, temporal_offset=temporal_offset, kv_cache=kv_cache)
        x = x + y * e[2]

        # cross-attention & ffn
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    @checkpoint_wrapper
    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    @checkpoint_wrapper
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class HeadAnimate(Head):
    @checkpoint_wrapper
    def forward(self, x, e):
        """
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class WanAnimateSelfAttention(WanSelfAttention):
    @checkpoint_wrapper
    def forward(self, x, seq_lens, grid_sizes, freqs, context=None):
        """
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            kv = default(context, x)
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(kv)).view(b, s, n, d)
            v = self.v(kv).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAnimateCrossAttention(WanSelfAttention):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        use_img_emb=True
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps
        )
        self.use_img_emb = use_img_emb

        if use_img_emb:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    @checkpoint_wrapper
    def forward(self, x, context, context_lens):
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        """
        if self.use_img_emb:
            context_img = context[:, :257]
            context = context[:, 257:]
        else:
            context = context

        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if self.use_img_emb:
            k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
            v_img = self.v_img(context_img).view(b, -1, n, d)
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)

        if self.use_img_emb:
            img_x = img_x.flatten(2)
            x = x + img_x

        x = self.o(x)
        return x

class WanMultiViewAttentionBlock(WanSelfAttention):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.mv_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )
        self.norm1 = WanLayerNorm(dim, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        
    @checkpoint_wrapper
    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        Args:
            x(Tensor): Shape [B * 2, L, C], first half and second half batches are different views
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, n, c = x.shape
        x = super().forward(x, seq_lens, grid_sizes, freqs)
        x = x.view(b//2, n*2, c)

        return x

class WanAnimateAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        use_img_emb=True,
        multi_view_attn=False,
        checkpoint=True,
    ):

        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.checkpoint = checkpoint
        
        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanAnimateSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        
        self.norm3 = WanLayerNorm(
            dim, eps, elementwise_affine=True
        ) if cross_attn_norm else nn.Identity()
            
        self.cross_attn = WanAnimateCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, use_img_emb=use_img_emb)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        
        # multi-view attention
        if multi_view_attn:
            self.mv_norm = WanLayerNorm(dim, eps)
            self.mv_attn = WanAnimateSelfAttention(dim, num_heads, window_size, qk_norm, eps)

    @checkpoint_wrapper
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        """
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs
        )
        x = x + y * e[2]

        if hasattr(self, 'mv_attn'):
            x_in = self.mv_norm(x)
            x = self.mv_attn(
                x_in, seq_lens, grid_sizes, freqs, context=x_in.flip(0)
            ) + x
            
        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    @checkpoint_wrapper
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class WanAnimateModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=36,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        motion_encoder_dim=512,
        camera_dim=None,
        use_img_emb=True,
        use_mv_attn=False,
        use_checkpoint=True,
        **kwargs,
    ):

        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.motion_encoder_dim = motion_encoder_dim
        self.use_img_emb = use_img_emb
        
        self.multi_view = exists(camera_dim)

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.pose_patch_embedding = nn.Conv3d(
            16, dim, kernel_size=patch_size, stride=patch_size
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAnimateAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps, use_img_emb, use_mv_attn, 
                              checkpoint=use_checkpoint) for _ in range(num_layers)
        ])

        # head
        self.head = HeadAnimate(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        self.img_emb = MLPProj(1280, dim)

        # face adapters        
        self.motion_encoder = Generator(size=512, style_dim=512, motion_dim=20)
        self.face_encoder = FaceEncoder(
            in_dim=motion_encoder_dim,
            hidden_dim=self.dim,
            num_heads=4,
        )
        self.face_adapter = FaceAdapter(
            heads_num=self.num_heads,
            hidden_dim=self.dim,
            num_adapter_layers=self.num_layers // 5,
        )

        if self.multi_view:
            # camera projections
            self.camera_embedding = nn.Sequential(
                nn.Conv3d(camera_dim, dim, kernel_size=1, stride=1), 
                nn.SiLU(), 
                zero_module(nn.Conv3d(dim, dim, kernel_size=1, stride=1))
            )
            # self.camera_embedding = nn.Sequential(
                # nn.Linear(camera_dim, dim), nn.SiLU(), zero_module(nn.Linear(dim, dim))
            # )
            # self.camera_projection = nn.Sequential(nn.SiLU(), zero_module(nn.Linear(dim, dim * 6)))

        # selective gradient mode for face training
        self._selective_grad = False
        self._inject_interval = 5

        # initialize weights
        self.init_weights()

    def enable_selective_grad(self):
        """Enable selective gradient for face/flame training mode.

        Only blocks with adapter injection (every 5th) participate in backward.
        Non-injection blocks use straight-through gradient estimation:
        forward computes normally, backward passes gradient as identity.

        This skips backward recomputation for ~80% of frozen blocks.
        """
        self._selective_grad = True
        for i, block in enumerate(self.blocks):
            if i % self._inject_interval != 0:
                block.checkpoint = False

    def after_patch_embedding(
        self, 
        x: List[torch.Tensor], 
        control, 
        face_pixel_values, 
        motion_vec_flame, 
        smpl_latents,
        e_camera = None
    ):
        if exists(control) or exists(smpl_latents):
            if exists(control) and exists(smpl_latents):
                control = [self.pose_patch_embedding(u.unsqueeze(0) + smpl.unsqueeze(0)) for u, smpl in zip(control, smpl_latents)]
            elif exists(control):
                control = [self.pose_patch_embedding(u.unsqueeze(0)) for u in control]
            else:
                control = [self.pose_patch_embedding(smpl.unsqueeze(0)) for smpl in smpl_latents]
            
            for x_, control_ in zip(x, control):
                x_[:, :, -control_.shape[2]:] += control_
                if exists(e_camera):
                    x_[:, :, -control_.shape[2]:] += e_camera

        if exists(face_pixel_values):
            T = face_pixel_values.size(2)
            face_pixel_values = rearrange(face_pixel_values, "b c t h w -> (b t) c h w")

            encode_bs = 8
            face_pixel_values_tmp = []
            for i in range(math.ceil(face_pixel_values.shape[0]/encode_bs)):
                face_pixel_values_tmp.append(self.motion_encoder.get_motion(face_pixel_values[i*encode_bs:(i+1)*encode_bs]))

            motion_vec = torch.cat(face_pixel_values_tmp)
            motion_vec = rearrange(motion_vec, "(b t) c -> b t c", t=T)

            if exists(motion_vec_flame):
                motion_vec += motion_vec_flame
        else:
            motion_vec = motion_vec_flame

        if exists(motion_vec): 
            motion_vec = motion_vec.to(x[0].dtype)
            motion_vec = self.face_encoder(motion_vec)
            B, L, H, C = motion_vec.shape
            pad_face = torch.zeros(B, 1, H, C).type_as(motion_vec)
            motion_vec = torch.cat([pad_face, motion_vec], dim=1)
        return x, motion_vec


    def after_transformer_block(self, block_idx, x, motion_vec, motion_masks=None):
        if block_idx % 5 == 0 and exists(motion_vec):
            # print(f"x.shape: {x.shape}, motion_vec.shape: {motion_vec.shape}")
            adapter_args = [x, motion_vec, motion_masks]
            residual_out = self.face_adapter.fuser_blocks[block_idx // 5](*adapter_args)
            x = residual_out + x
        return x

    def forward(
        self,
        x,
        t,
        clip_fea,
        context,
        seq_len,
        y = None,
        control = None,
        camera_params = None,
        face_pixel_values = None,
        smpl_latents = None,
        motion_vec = None
    ):
        # params
        device = self.patch_embedding.weight.device
        dtype = self.patch_embedding.weight.dtype
        e_camera = None

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        if exists(camera_params):
            e_camera = self.camera_embedding(camera_params.to(dtype))
        else:
            e_camera = 0.

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0).to(dtype)) for u in x]
        x, motion_vec = self.after_patch_embedding(x, control, face_pixel_values, motion_vec, smpl_latents, e_camera)

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float()
            )
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        e, e0 = e.to(dtype), e0.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if self.use_img_emb:
            context_clip = self.img_emb(clip_fea) # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        for idx, block in enumerate(self.blocks):
            if self._selective_grad and idx % self._inject_interval != 0:
                # Straight-through: forward computes normally, backward is identity
                with torch.no_grad():
                    x_out = block(x, **kwargs)
                x = x + (x_out - x).detach()
            else:
                x = block(x, **kwargs)
            x = self.after_transformer_block(idx, x, motion_vec)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u for u in x]


    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)