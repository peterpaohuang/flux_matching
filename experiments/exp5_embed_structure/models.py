from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.exponential import ExponentialIS
from src.normalizer import LossNormalizer

@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param, alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    for name, buf in model.named_buffers():
        if name in ema_buffers and torch.is_floating_point(buf):
            ema_buffers[name].copy_(buf)

class SinusoidalSigmaEmbedding(nn.Module):
    def __init__(self, emb_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_period = max_period

    def forward(self, sigmas: torch.Tensor) -> torch.Tensor:
        sigmas = sigmas.reshape(-1)
        half = self.emb_dim // 2
        device, dtype = sigmas.device, sigmas.dtype
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1)
        )
        x = torch.log(sigmas.clamp_min(1e-12)).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        if self.emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, seq_len: int, emb_dim: int):
        super().__init__()
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        half = emb_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
        table = torch.cat([torch.sin(pos * freqs), torch.cos(pos * freqs)], dim=1)
        if emb_dim % 2 == 1:
            table = F.pad(table, (0, 1))
        self.register_buffer("table", table, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table[: x.shape[1]].unsqueeze(0).to(dtype=x.dtype, device=x.device)

class AttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("hidden dim must be divisible by n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x: torch.Tensor, attn_mask) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, Dh]
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.view(1, 1, T, T), float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        h = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        h = self.out(h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class ModelConfig:
    seq_len: int = 50
    input_dim: int = 4
    hidden_dim: int = 96
    depth: int = 2       # number of encoder levels (each halves the sequence length)
    n_heads: int = 4     # unused, kept for config compatibility
    dropout: float = 0.0
    causal: bool = False

class TrajectoryDriftModel(nn.Module):
    """
    Raw head predicts unit-scale h(x, sigma). The score is h / sigma.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.pos_emb = SinusoidalPositionEmbedding(cfg.seq_len, cfg.hidden_dim)
        self.sigma_emb = nn.Sequential(
            SinusoidalSigmaEmbedding(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [AttentionBlock(cfg.hidden_dim, cfg.n_heads, dropout=cfg.dropout) for _ in range(cfg.depth)]
        )
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.input_dim)

        if cfg.causal:
            mask = torch.triu(torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool), diagonal=1)
        else:
            mask = None
        self.register_buffer("attn_mask", mask, persistent=False)

        self.importance_sampler_net = ExponentialIS()
        self.s_noise_normalizer = LossNormalizer()

    def forward_raw(self, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, D], got {tuple(x.shape)}")
        if x.shape[1] != self.cfg.seq_len or x.shape[2] != self.cfg.input_dim:
            raise ValueError(
                f"Expected [B, {self.cfg.seq_len}, {self.cfg.input_dim}], got {tuple(x.shape)}"
            )

        h = self.input_proj(x)
        h = h + self.pos_emb(h)
        sigma_context = self.sigma_emb(sigmas).unsqueeze(1)
        h = h + sigma_context
        for block in self.blocks:
            h = block(h, self.attn_mask)
        h = self.out_proj(self.out_norm(h))
        return h

    def forward(self, x: torch.Tensor, sigmas: torch.Tensor, return_raw: bool = False):
        raw = self.forward_raw(x, sigmas)
        sigma_img = sigmas.reshape(-1, 1, 1).to(dtype=x.dtype, device=x.device)
        score = raw / sigma_img.clamp_min(1e-12)
        if return_raw:
            return raw
        return score