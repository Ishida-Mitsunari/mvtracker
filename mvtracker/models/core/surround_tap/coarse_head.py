"""Local BEV-patch attention head: track token -> gated Δxy (no Z)."""
from __future__ import annotations

import torch
import torch.nn as nn


class CoarseBEVHead(nn.Module):
    """Single-layer attention from a track token onto local BEV patches.

    Parameters are named under ``coarse_head.*`` so
    ``trainable_name_prefixes: [coarse_]`` selects them.
    """

    def __init__(self, dim: int = 128, num_heads: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        self.dxy = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, feat: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """Predict gated XY residual.

        Args:
            feat: (B, S, N, C) track features.
            patches: (B, S, N, P, C) local BEV tokens (one or more pyramid levels).

        Returns:
            delta_xy: (B, S, N, 2)
        """
        B, S, N, C = feat.shape
        assert patches.shape[:3] == (B, S, N) and patches.shape[-1] == C
        p = patches.shape[3]
        q = self.q_proj(feat).reshape(B * S * N, 1, C)
        k = self.k_proj(patches).reshape(B * S * N, p, C)
        v = self.v_proj(patches).reshape(B * S * N, p, C)
        # attn(query, key, value); key_padding not needed (grid_sample zeros OOB)
        tok, _ = self.attn(q, k, v, need_weights=False)
        tok = self.out_norm(tok.squeeze(1) + q.squeeze(1))
        dxy = self.dxy(tok)
        gate = torch.sigmoid(self.gate(tok))
        return (gate * dxy).view(B, S, N, 2)
