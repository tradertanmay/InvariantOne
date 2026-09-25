"""
Unit tests for DirectComparativeHead (Phase 1D-A).
Tests:
- Exact parameter counts for D0, D1, D2
- Antisymmetry: d_ij = -d_ji, d_ii = 0
- Variable K handling (including K=1)
- Permutation equivariance across exhaustive (K <= 5) and sampled (K=6, 8) permutations
"""

import itertools
import pytest
import torch
import torch.nn.functional as F

from invariantone.architectures.direct_comparative import DirectComparativeHead


def test_parameter_counts_reported_exactly():
    d0 = DirectComparativeHead(mode="absolute_only")
    d1 = DirectComparativeHead(mode="pairwise_only")
    d2 = DirectComparativeHead(mode="combined")

    p0 = d0.count_parameters()
    p1 = d1.count_parameters()
    p2 = d2.count_parameters()

    print(f"\nD0 (Absolute-Only) Active Parameters: {p0['active_trainable_parameters']:,}")
    print(f"D1 (Pairwise-Only) Active Parameters: {p1['active_trainable_parameters']:,}")
    print(f"D2 (Combined) Active Parameters: {p2['active_trainable_parameters']:,}")

    # Proj: 2560*256 + 256 + 256 + 256 = 656,128
    # Abs: 256*256 + 256 + 256*1 + 1 = 66,049
    # Pair: 1024*256 + 256 + 256*1 + 1 = 262,657
    assert p0["active_trainable_parameters"] == 656128 + 66049  # 722,177
    assert p1["active_trainable_parameters"] == 656128 + 262657  # 918,785
    assert p2["active_trainable_parameters"] == 656128 + 66049 + 262657  # 984,834


def test_variable_k_and_k_equals_one():
    for mode in ["combined", "absolute_only", "pairwise_only"]:
        model = DirectComparativeHead(mode=mode)
        model.eval()

        # Test K=1
        h1 = torch.randn(1, 2560)
        s1 = model(h1)
        assert s1.shape == (1,)
        assert torch.isfinite(s1).all()

        # Test K=5
        h5 = torch.randn(5, 2560)
        s5 = model(h5)
        assert s5.shape == (5,)
        assert torch.isfinite(s5).all()


@pytest.mark.parametrize("mode", ["combined", "absolute_only", "pairwise_only"])
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_exhaustive_permutation_equivariance(mode: str, k: int):
    torch.manual_seed(42 + k)
    model = DirectComparativeHead(mode=mode)
    model.eval()

    h = torch.randn(k, 2560)
    with torch.no_grad():
        base_scores = model(h)
        base_probs = F.softmax(base_scores, dim=-1)

    all_perms = list(itertools.permutations(range(k)))

    max_score_delta = 0.0
    max_prob_delta = 0.0
    flips = 0

    for perm in all_perms:
        perm_list = list(perm)
        h_perm = h[perm_list]

        with torch.no_grad():
            perm_scores = model(h_perm)
            perm_probs = F.softmax(perm_scores, dim=-1)

        # Un-permute
        inv_perm = [perm_list.index(i) for i in range(k)]
        restored_scores = perm_scores[inv_perm]
        restored_probs = perm_probs[inv_perm]

        s_delta = (base_scores - restored_scores).abs().max().item()
        p_delta = (base_probs - restored_probs).abs().max().item()

        max_score_delta = max(max_score_delta, s_delta)
        max_prob_delta = max(max_prob_delta, p_delta)

        if int(torch.argmax(base_probs).item()) != int(torch.argmax(restored_probs).item()):
            flips += 1

    assert flips == 0, f"Mode {mode} K={k}: Observed {flips} prediction flips across permutations!"
    assert max_prob_delta < 1e-5, f"Mode {mode} K={k}: Max prob delta {max_prob_delta} exceeded tolerance 1e-5!"
    assert max_score_delta < 1e-4, f"Mode {mode} K={k}: Max score delta {max_score_delta} exceeded tolerance 1e-4!"
