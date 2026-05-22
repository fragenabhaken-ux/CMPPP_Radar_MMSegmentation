from __future__ import annotations

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmseg.registry import MODELS


@MODELS.register_module()
class DINOTransformerNeck(BaseModule):
    """Trainable Transformer neck on top of frozen DINOv3 patch features.

    Input:
        tuple/list containing spatial feature maps. For the current
        FrozenTimmDINOv3Backbone this is usually one feature map:
            (B, C_in, H_patch, W_patch)

    Output:
        tuple containing one spatial feature map:
            (B, embed_dims, H_patch, W_patch)

    The neck first optionally projects C_in -> embed_dims, then applies
    standard TransformerEncoderLayer blocks over flattened patch tokens.

    Note:
        attn_dropout is kept in the interface for future compatibility, but
        PyTorch's nn.TransformerEncoderLayer exposes only one shared dropout
        argument for attention and FFN dropout. Therefore, attn_dropout is not
        used separately in this implementation.
    """

    def __init__(
        self,
        in_channels: int = 384,
        embed_dims: int = 384,
        in_index: int = 0,
        num_layers: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        norm_cfg: dict = dict(type="LN"),
        init_cfg: dict | None = None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = int(in_channels)
        self.embed_dims = int(embed_dims)
        self.in_index = int(in_index)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.attn_dropout = float(attn_dropout)

        if self.embed_dims % self.num_heads != 0:
            raise ValueError(
                f"embed_dims={self.embed_dims} must be divisible by "
                f"num_heads={self.num_heads}."
            )

        if self.in_channels != self.embed_dims:
            self.input_proj = nn.Linear(self.in_channels, self.embed_dims)
        else:
            self.input_proj = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dims,
            nhead=self.num_heads,
            dim_feedforward=int(self.embed_dims * self.mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
        )

        _, self.out_norm = build_norm_layer(norm_cfg, self.embed_dims)

    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            if len(inputs) <= self.in_index:
                raise ValueError(
                    f"DINOTransformerNeck got {len(inputs)} input feature maps, "
                    f"but in_index={self.in_index} was requested."
                )
            x = inputs[self.in_index]
        else:
            x = inputs

        if x.ndim != 4:
            raise ValueError(
                "DINOTransformerNeck expects a spatial feature map "
                f"(B,C,H,W), got {tuple(x.shape)}."
            )

        b, c, h, w = x.shape

        if c != self.in_channels:
            raise ValueError(
                f"DINOTransformerNeck expected {self.in_channels} input channels, "
                f"but got {c}."
            )

        # (B, C, H, W) -> (B, N, C), where N = H_patch * W_patch
        tokens = x.flatten(2).transpose(1, 2).contiguous()

        tokens = self.input_proj(tokens)
        tokens = self.transformer(tokens)
        tokens = self.out_norm(tokens)

        # (B, N, C) -> (B, C, H, W)
        out = tokens.transpose(1, 2).contiguous().view(
            b, self.embed_dims, h, w
        )

        return (out,)


'''
from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange

from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmseg.registry import MODELS


class QKNorm(nn.Module):
    """
    Normalises keys and querries using RMS normalization.
    The default implementation uses seperate scalings for Q and K, and only 
    scalings for the head dimension, i.e. shared scalings accross heads.
    The ``head_shape`` argument needs to be aligned with the shape of your
    Q and K tensors, e.g. if they are shaped (B, NH, L, HD), then head_shape 
    should be HD/(HD,) for head-wise scalings, (NH, 1, HD) for head-specific scalings;
    if they are shaped (B, L, D), then head_shape should be (D,) for 
    single head feature-wise scalings; if they are shaped (B, L, NH, HD), then 
    head_shape should be (NH, HD) or (HD), etc.

    The motivation of this design is to be able to test these two hypotheses:

    1. The scaling factors for Q and K are multiplicated against each other 
    nearly directly afterwards. Only RoPE and similar lead to actual 
    mathematical differences compared to shared weights

    [s_q1 * q1, s_q2 * q2] @ Rot(a)^T @ Rot(b) @ [s_k1 * k1, s_k2 * k2]^T 
     = [s_q1 * q1, s_q2 *q2] @ Rot(b-a) [s_k1 * k1, s_k2 * k2]^T
    with cos = cos(b-a) and sin = sin(b-a)
     =   cos * (s_q1*s_k1 * q1*k1 + s_q2*s_k2 * q2*k2) 
       + sin * (s_q1*s_k2 * q1*k2 - s_q2*s_k1 * q2*k1)
    with v1 = s_q1*s_k1, v2 = s_q2*s_k2, v3 = s_q1*s_k2, and v4 = s_q2*s_k1 = v1*v2/v3
     =   cos * (v1 * q1*k1 + v2 * q2*k2) + sin * (v3 * q1*k2 - v1*v2/v3 * q2*k1)
    
    Note how the the only difference to just considering the joint scaling 
    (s_q1 = s_k1 = sqrt(v1) etc.) is the capability to slightly adjust the sinus
    component. However, when considering that the purpose of RoPE is to bias the
    model to high attention when angles a and b are similar, i.e. when 
    sin(a-b) is approx 0, then the benefit of having separate scalings for Q 
    and K is very questionable. The only remaining potnetial benefit are 
    potential gradient stability benefits for disentangling scalings of K and Q.

    2. Shared scaling accross heads is sufficient for implementations without
    positional embeddings, as the Q and K weight matrices can arbitrarily 
    permute their rows to achieve head specific scalings of different features,
    just which scalings to assign is shared accross heads.
    However, when introducing positional embedders, all heads will have to share
    if they weight higher/lower frequency details of the positional embeddings.
    Having multiple heads feels contrary to forcing all heads to aggree on one
    context length bias, therefore this interface allowing for head-specific
    scalings to test this hypothesis.
    """

    def __init__(self, head_shape: int | Tuple[int, ...], 
                 shared_qk_weights = False, eps : float = 1e-6):
        super().__init__()
        self.q_scale = nn.Parameter(torch.ones(head_shape))
        self.k_scale = nn.Parameter(torch.ones(head_shape)) if not shared_qk_weights else self.q_scale
        self.eps = eps
    
    def _RMS(self, x: Tensor, scale: Tensor) -> Tensor:
        x_f = x.float()
        rrms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * rrms).to(x.dtype) * scale

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        # cast back to v's dtype so mixed-precision stays stable
        return self._RMS(q, self.q_scale).to(v.dtype), self._RMS(k, self.k_scale).to(v.dtype)

# from https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py#L818-L833
def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim # not used?
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))],
            dim=-3,
        )

        return emb.unsqueeze(1)

class ParallelTransformerBlock(nn.Module):
    """
    Pre-norm transformer block with **parallel** attention and MLP streams.

    A single large linear (``linear1``) projects the normalised input into
    QKV tokens *and* the MLP gate simultaneously:

        [q, k, v | gate] = LN(x) @ W1ᵀ

    After attention and gate activation, a second linear (``linear2``) maps
    back to the residual stream:

        x  ←  x + W2ᵀ · [attn_out(q,k,v) ⊕ act(gate)]

    This fuses two separate sub-layers, saves one residual add and one norm,
    and makes better use of modern GPU memory bandwidth.

    Args:
        dim:         Token (model) dimension D.
        head_dim:    Per-head dimension HD.  num_heads = D // HD.
        mlp_ratio:   MLP hidden dim = dim * mlp_ratio.
        bias:        Whether linear layers use a bias term.
        dropout:     Attention dropout (only active during training).
        use_qk_norm: Apply per-head RMSNorm to Q and K before attention.
        sdpa_fn:     Default positional encoder / attention callable.
                     Must satisfy the GeometricAttention interface:
                         fn(q, k, v, **kwargs) -> Tensor
                     where q/k/v are (B, NH, L, HD).
                     Defaults to the base GeometricAttention (= plain SDPA).
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        qk_scale_shape: int | Tuple[int, ...] | None = None,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        assert dim % head_dim == 0, (
            f"dim={dim} must be divisible by head_dim={head_dim}"
        )
       
        self.num_heads = dim // head_dim
        self.head_dim  = head_dim
        self.mlp_hidden = int(dim * mlp_ratio)
        if qk_scale_shape is None:
            qk_scale_shape = head_dim  # default: separate scaling for each feature, shared accross heads


        self.pre_norm = nn.LayerNorm(dim, bias=bias)

        # Fused QKV + MLP-gate projection
        self.linear1 = nn.Linear(dim, 3 * dim + self.mlp_hidden, bias=bias)

        # Output: attn_out (dim) ++ mlp_out (mlp_hidden) → dim
        self.linear2 = nn.Linear(dim + self.mlp_hidden, dim, bias=bias)

        self.qk_norm = QKNorm(head_shape=qk_scale_shape) if use_qk_norm else None
        self.mlp_act = nn.SELU()

    def forward(
        self,
        x: Tensor,
        pos: Tensor | None = None,
        **sdpa_kwargs,
    ) -> Tensor:
        """
        Args:
            x:          (B, L, D)  input tokens
            pos:        (B, 1, N, HD//2, 2, 2) pre-computed RoPE rotation matrices
            **sdpa_kwargs:  extra keyword arguments forwarded to sdpa_fn
                            (e.g. ``attn_mask``).

        Returns:
            (B, L, D) output tokens
        """

        # --- single pre-norm ---
        h = self.pre_norm(x)

        # --- fused projection ---
        qkv, hidden_mlp = self.linear1(h).split(
            [3 * self.num_heads * self.head_dim, self.mlp_hidden], dim=-1
        )

        # split heads:  (B, L, 3*NH*HD) → (B, NH, L, HD) × 3
        q, k, v = rearrange(qkv, "b l (k nh hd) -> k b nh l hd",
                             k=3, nh=self.num_heads)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k, v)
        
        # encode each patches relative position via RoPE
        if pos is not None:
            q, k = apply_rope(q, k, pos)


        # attention — any GeometricAttention or callable with same signature
        attn_out = nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, **sdpa_kwargs
        )
        attn_out = rearrange(attn_out, "b nh l hd -> b l (nh hd)")

        # parallel MLP gate stream
        mlp_out = self.mlp_act(hidden_mlp)

        # fused output projection + residual
        x = x + self.linear2(torch.cat([attn_out, mlp_out], dim=-1))
        return x


class TransformerNeck(BaseModule):
    """Trainable Transformer neck on top of frozen DINOv3 patch features.

    Args:
        in_channels (int): Number of channels in the input feature map. Default: 384.
        embed_dims (int): Number of channels in the output feature map. Default: 384.
        in_index (int): Index of the input feature map to use if multiple are provided (e.g. output of multiple stages). Default: 0.
        num_layers (int): Number of Transformer encoder layers. Default: 6.
        num_heads (int): Number of attention heads in each Transformer encoder layer. Default: 6.
        mlp_ratio (float): Ratio of the hidden dimension in the MLP to the embedding dimension. Default: 4.0.
        dropout (float): Dropout rate for the Transformer encoder layers. Default: 0.0.
        attn_dropout (float): Dropout rate for attention weights in the Transformer encoder layers. Default: 0.0.
        norm_cfg (dict): Configuration for the normalization layer. Default: dict(type="LN").
        init_cfg (dict | None): Initialization configuration. Default: None.
    
    """

    def __init__(
        self,
        in_channels: int = 384,
        embed_dims: int = 384,
        in_index: int = 0,
        num_layers: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        norm_cfg: dict = dict(type="LN"),
        init_cfg: dict | None = None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = int(in_channels)
        self.embed_dims = int(embed_dims)
        self.in_index = int(in_index)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.attn_dropout = float(attn_dropout)

        if self.embed_dims % self.num_heads != 0:
            raise ValueError(
                f"embed_dims={self.embed_dims} must be divisible by "
                f"num_heads={self.num_heads}."
            )

        if self.in_channels != self.embed_dims:
            self.input_proj = nn.Linear(self.in_channels, self.embed_dims)
        else:
            self.input_proj = nn.Identity()
        
        dim = self.embed_dims // self.num_heads

        # first dim/2 dimension of each head dimension encode x-axis position,
        # second dim/2 dimension of each head dimension encode y-axis position
        self.pos_embedder = EmbedND(dim=dim, theta=2000, axes_dim=[dim//2, dim//2])

        self.layer_list = nn.ModuleList(
            ParallelTransformerBlock(
                dim=self.embed_dims,
                head_dim=dim,
                mlp_ratio=mlp_ratio,
                use_qk_norm=True,
                bias=False,
            )
            for _ in range(self.num_layers)
        )

        _, self.out_norm = build_norm_layer(norm_cfg, self.embed_dims)

        
    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            if len(inputs) <= self.in_index:
                raise ValueError(
                    f"DINOTransformerNeck got {len(inputs)} input feature maps, "
                    f"but in_index={self.in_index} was requested."
                )
            x = inputs[self.in_index]
        else:
            x = inputs

        if x.ndim != 4:
            raise ValueError(
                "DINOTransformerNeck expects a spatial feature map "
                f"(B,C,H,W), got {tuple(x.shape)}."
            )

        b, c, h, w = x.shape

        if c != self.in_channels:
            raise ValueError(
                f"DINOTransformerNeck expected {self.in_channels} input channels, "
                f"but got {c}."
            )

        pidx = torch.stack(
            [torch.arange(h, device=x.device).repeat_interleave(w),  # (H*W,)
             torch.arange(w, device=x.device).repeat(h)],  # (H*W,)
            dim=-1
        ).unsqueeze(0).expand(b, -1, -1)  # (B, N, 2)
        pos = self.pos_embedder(pidx)  # (B, 1, N, D)

        # (B, C, H, W) -> (B, N, C), where N = H_patch * W_patch
        tokens = x.flatten(2).transpose(1, 2).contiguous()

        tokens = self.input_proj(tokens)
        dropout = self.attn_dropout if self.training else 0.0
        for layer in self.layer_list:
            tokens = layer(tokens, pos, dropout_p=dropout)
        tokens = self.out_norm(tokens)

        # (B, N, C) -> (B, C, H, W)
        out = tokens.transpose(1, 2).contiguous().view(
            b, self.embed_dims, h, w
        )

        return (out,)
 '''