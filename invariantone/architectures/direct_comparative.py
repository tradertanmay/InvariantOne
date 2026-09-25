"""
Direct Comparative Decision Head (Phase 1D-A).
Computes decision scores directly from pooled option representations without any frozen C0 score prior.

Supports 3 scientific ablation modes:
1. D0 (absolute_only): score_i = b(h_i)
2. D1 (pairwise_only): score_i = comparative_i = 1/(K-1) * sum_{j != i} d_ij
3. D2 (combined): score_i = b(h_i) + comparative_i

Guarantees:
- Strict antisymmetry: d_ij = phi(h_i, h_j) - phi(h_j, h_i) => d_ij = -d_ji, d_ii = 0
- Variable K support (K=1 returns comparative = 0)
- Structural permutation equivariance: score(pi(H)) = pi(score(H))
- Zero option-index embeddings or positional encodings
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

AblationMode = Literal["combined", "absolute_only", "pairwise_only"]


class DirectComparativeHead(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2560,
        proj_dim: int = 256,
        comparator_hidden_dim: int = 256,
        mode: AblationMode = "combined",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim
        self.comparator_hidden_dim = comparator_hidden_dim
        self.mode: AblationMode = mode

        # 1. Option Projection (shared for all options)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim, bias=True),
            nn.LayerNorm(proj_dim),
        )

        # 2. Absolute Option Scorer b(h_i)
        self.abs_net = nn.Sequential(
            nn.Linear(proj_dim, proj_dim, bias=True),
            nn.SiLU(),
            nn.Linear(proj_dim, 1, bias=True),
        )

        # 3. Pairwise Comparator MLP phi(h_i, h_j)
        # Input: [z_i, z_j, z_i - z_j, z_i * z_j] in R^(4 * proj_dim) = R^1024
        pairwise_in_dim = 4 * proj_dim
        self.pair_net = nn.Sequential(
            nn.Linear(pairwise_in_dim, comparator_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(comparator_hidden_dim, 1, bias=True),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> dict[str, int]:
        """Returns exact trainable parameter counts by component."""
        proj_params = sum(p.numel() for p in self.proj.parameters() if p.requires_grad)
        abs_params = sum(p.numel() for p in self.abs_net.parameters() if p.requires_grad)
        pair_params = sum(p.numel() for p in self.pair_net.parameters() if p.requires_grad)

        if self.mode == "absolute_only":
            active = proj_params + abs_params
        elif self.mode == "pairwise_only":
            active = proj_params + pair_params
        else:  # combined
            active = proj_params + abs_params + pair_params

        return {
            "proj_parameters": proj_params,
            "abs_parameters": abs_params,
            "pair_parameters": pair_params,
            "active_trainable_parameters": active,
            "total_model_parameters": proj_params + abs_params + pair_params,
        }

    def forward(self, h_options: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_options: Tensor of shape [K, hidden_size] or [B, K, hidden_size].
        Returns:
            scores: Tensor of shape [K] or [B, K].
        """
        is_batched = h_options.dim() == 3
        if not is_batched:
            h_options = h_options.unsqueeze(0)  # [1, K, hidden_size]

        # Ensure h_options matches projection layer dtype
        proj_dtype = self.proj[0].weight.dtype
        if h_options.dtype != proj_dtype:
            h_options = h_options.to(proj_dtype)

        b_sz, k_opts, _ = h_options.shape

        # Step 1: Project each option to R^proj_dim
        # z: [B, K, proj_dim]
        z = self.proj(h_options)

        # Step 2: Compute absolute score b(h_i)
        # abs_scores: [B, K]
        if self.mode in ("combined", "absolute_only"):
            abs_scores = self.abs_net(z).squeeze(-1)
        else:
            abs_scores = torch.zeros(b_sz, k_opts, device=h_options.device, dtype=h_options.dtype)

        # Step 3: Compute pairwise comparative score
        if self.mode in ("combined", "pairwise_only") and k_opts > 1:
            # Construct all pairs: z_i in [B, K, 1, D], z_j in [B, 1, K, D]
            z_i = z.unsqueeze(2).expand(b_sz, k_opts, k_opts, self.proj_dim)
            z_j = z.unsqueeze(1).expand(b_sz, k_opts, k_opts, self.proj_dim)

            # u_ij = [z_i, z_j, z_i - z_j, z_i * z_j]
            diff = z_i - z_j
            prod = z_i * z_j
            u_ij = torch.cat([z_i, z_j, diff, prod], dim=-1)  # [B, K, K, 1024]
            u_ji = torch.cat([z_j, z_i, -diff, prod], dim=-1)  # [B, K, K, 1024]

            # a_ij = phi(u_ij), a_ji = phi(u_ji)
            a_ij = self.pair_net(u_ij).squeeze(-1)  # [B, K, K]
            a_ji = self.pair_net(u_ji).squeeze(-1)  # [B, K, K]

            # Enforce exact antisymmetry: d_ij = a_ij - a_ji => d_ij = -d_ji, d_ii = 0
            d_ij = a_ij - a_ji

            # Mask out diagonal to ensure exact zero self-comparison
            eye_mask = torch.eye(k_opts, device=h_options.device, dtype=torch.bool).unsqueeze(0)
            d_ij = d_ij.masked_fill(eye_mask, 0.0)

            # Comparative score: mean over competitors
            comp_scores = d_ij.sum(dim=-1) / (k_opts - 1)  # [B, K]
        else:
            comp_scores = torch.zeros(b_sz, k_opts, device=h_options.device, dtype=h_options.dtype)

        # Final unified score
        if self.mode == "absolute_only":
            scores = abs_scores
        elif self.mode == "pairwise_only":
            scores = comp_scores
        else:  # combined
            scores = abs_scores + comp_scores

        if not is_batched:
            scores = scores.squeeze(0)

        return scores
