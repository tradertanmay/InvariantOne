"""
Tests for Exact Distributional Permutation Equivariance in InvariantOne.
"""

from __future__ import annotations

import itertools
import pytest
import torch

from invariantone.architectures.direct_comparative_head import DirectComparativeHead
from invariantone.calibration import calibrated_softmax


def test_24_permutation_equivariance_direct_head():
    """
    Evaluates all 4! = 24 permutations of option representations.
    Mathematically, because candidate branches are scored independently by b(h_i),
    the score vector must be strictly permutation-equivariant:
        score(pi(H)) == pi(score(H))
        TVD(P(pi(H)), pi(P(H))) == 0.000000
    """
    head = DirectComparativeHead(hidden_size=2560, proj_dim=256, mode="absolute_only")
    head.eval()

    # Fixed synthetic representation for 4 options
    torch.manual_seed(42)
    canonical_reps = torch.randn(4, 2560)

    with torch.no_grad():
        canonical_scores = head(canonical_reps)
        canonical_probs = calibrated_softmax(canonical_scores, tau=1.0091)

    all_perms = list(itertools.permutations(range(4)))
    assert len(all_perms) == 24

    max_prob_delta = 0.0
    max_score_delta = 0.0
    tvd_list = []

    for perm in all_perms:
        perm_reps = canonical_reps[list(perm)]
        with torch.no_grad():
            perm_scores = head(perm_reps)
            perm_probs = calibrated_softmax(perm_scores, tau=1.0091)

        # Map back to canonical order: perm_scores[i] is candidate perm[i]
        # Invert permutation: unperm[perm[i]] = i
        unpermuted_scores = torch.empty_like(perm_scores)
        unpermuted_probs = torch.empty_like(perm_probs)
        for new_idx, orig_idx in enumerate(perm):
            unpermuted_scores[orig_idx] = perm_scores[new_idx]
            unpermuted_probs[orig_idx] = perm_probs[new_idx]

        score_diff = torch.max(torch.abs(unpermuted_scores - canonical_scores)).item()
        prob_diff = torch.max(torch.abs(unpermuted_probs - canonical_probs)).item()
        tvd = 0.5 * torch.sum(torch.abs(unpermuted_probs - canonical_probs)).item()

        max_score_delta = max(max_score_delta, score_diff)
        max_prob_delta = max(max_prob_delta, prob_diff)
        tvd_list.append(tvd)

    mean_tvd = sum(tvd_list) / len(tvd_list)
    max_tvd = max(tvd_list)

    assert max_score_delta < 1e-6, f"Max score delta {max_score_delta} exceeds threshold"
    assert max_prob_delta < 1e-6, f"Max probability delta {max_prob_delta} exceeds threshold"
    assert mean_tvd < 1e-6, f"Mean TVD {mean_tvd} exceeds threshold"
    assert max_tvd < 1e-6, f"Max TVD {max_tvd} exceeds threshold"
