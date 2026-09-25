"""
Unit tests for InvariantOne Confidence and Uncertainty Layer.

Tests:
1. Single-model confidence, margin, entropy, normalized entropy
2. Single-model selective prediction and abstention criteria
3. Multi-checkpoint EnsembleDecisionResult properties and metrics
4. Ensemble disagreement, mutual information proxy, and choice agreement
5. Seed-aware configuration and cryptographic resolution for Seeds 42, 43, 44
"""

import math
import pytest
import torch

from invariantone import DecisionResult, EnsembleDecisionResult, InvariantOneConfig
from invariantone.config import FROZEN_SEEDS_HASHES
from invariantone.loading.checkpoint_loader import resolve_checkpoint_paths
from invariantone.loading.hash_verification import verify_checkpoint_hash


def test_decision_result_confidence_and_margin():
    """Verifies confidence (max prob) and margin (top1 - top2) calculations."""
    res = DecisionResult(
        choice="Option A",
        choice_index=0,
        probabilities=[0.75, 0.15, 0.08, 0.02],
        scores=[2.0, 0.5, 0.0, -1.0],
        options=["Option A", "Option B", "Option C", "Option D"],
    )
    assert pytest.approx(res.confidence, rel=1e-5) == 0.75
    assert pytest.approx(res.margin, rel=1e-5) == 0.60
    assert res.abstained is False


def test_decision_result_binary_margin():
    """Verifies margin on K=2 binary decision."""
    res = DecisionResult(
        choice="Yes",
        choice_index=0,
        probabilities=[0.8, 0.2],
        scores=[1.5, -0.5],
        options=["Yes", "No"],
    )
    assert pytest.approx(res.confidence, rel=1e-5) == 0.8
    assert pytest.approx(res.margin, rel=1e-5) == 0.6


def test_decision_result_entropy_and_normalized_entropy():
    """Verifies Shannon entropy and [0, 1] normalized entropy bounds."""
    # 1. Uniform distribution across K=4 options -> normalized_entropy = 1.0
    res_uniform = DecisionResult(
        choice="A",
        choice_index=0,
        probabilities=[0.25, 0.25, 0.25, 0.25],
        scores=[0.0, 0.0, 0.0, 0.0],
        options=["A", "B", "C", "D"],
    )
    assert pytest.approx(res_uniform.entropy, rel=1e-5) == math.log(4)
    assert pytest.approx(res_uniform.normalized_entropy, rel=1e-5) == 1.0

    # 2. Near-deterministic distribution -> normalized_entropy ~ 0.0
    res_certain = DecisionResult(
        choice="A",
        choice_index=0,
        probabilities=[0.9997, 0.0001, 0.0001, 0.0001],
        scores=[10.0, -10.0, -10.0, -10.0],
        options=["A", "B", "C", "D"],
    )
    assert res_certain.entropy < 0.01
    assert 0.0 <= res_certain.normalized_entropy < 0.01


def test_decision_result_should_abstain():
    """Tests selective prediction abstention logic."""
    res = DecisionResult(
        choice="Action 1",
        choice_index=0,
        probabilities=[0.60, 0.35, 0.05],
        scores=[1.0, 0.5, -1.0],
        options=["Action 1", "Action 2", "Action 3"],
    )
    # Confidence is 0.60, margin is 0.25
    assert res.should_abstain(min_confidence=0.70) is True
    assert res.should_abstain(min_confidence=0.55) is False
    assert res.should_abstain(min_margin=0.30) is True
    assert res.should_abstain(min_margin=0.20) is False
    assert res.should_abstain(min_confidence=0.50, min_margin=0.20) is False
    assert res.should_abstain() is False


def test_decision_result_to_dict():
    """Ensures serialization includes uncertainty fields."""
    res = DecisionResult(
        choice="A",
        choice_index=0,
        probabilities=[0.7, 0.3],
        scores=[1.0, 0.0],
        options=["A", "B"],
        abstained=True,
    )
    d = res.to_dict()
    assert d["choice"] == "A"
    assert "confidence" in d
    assert "margin" in d
    assert "entropy" in d
    assert "normalized_entropy" in d
    assert d["abstained"] is True


def test_ensemble_decision_result_metrics():
    """Verifies ensemble metrics across 3 seeds."""
    p42 = [0.80, 0.10, 0.10]
    p43 = [0.70, 0.20, 0.10]
    p44 = [0.60, 0.10, 0.30]

    mean_probs = [0.70, 0.133333, 0.166667]

    res = EnsembleDecisionResult(
        choice="Option 1",
        choice_index=0,
        probabilities=mean_probs,
        scores=[2.0, 0.5, 0.8],
        options=["Option 1", "Option 2", "Option 3"],
        seed_probabilities={42: p42, 43: p43, 44: p44},
        seed_choices={42: 0, 43: 0, 44: 0},
        ensemble_mean_probability=0.70,
        ensemble_std=0.10,
        ensemble_min=0.60,
        ensemble_max=0.80,
        choice_agreement=1.0,
        predictive_entropy=0.7963,
        expected_entropy=0.7500,
        mutual_information=0.0463,
        ensemble_stds=[0.10, 0.0577, 0.1155],
        ensemble_mins=[0.60, 0.10, 0.10],
        ensemble_maxs=[0.80, 0.20, 0.30],
    )

    # Base properties
    assert pytest.approx(res.confidence, rel=1e-4) == 0.70
    assert res.choice_agreement == 1.0
    assert pytest.approx(res.ensemble_std, rel=1e-4) == 0.10
    assert res.ensemble_min == 0.60
    assert res.ensemble_max == 0.80
    assert res.mutual_information >= 0.0

    # Abstention on agreement or std
    assert res.should_abstain(min_agreement=1.0) is False
    assert res.should_abstain(max_ensemble_std=0.05) is True
    assert res.should_abstain(max_ensemble_std=0.15) is False


def test_ensemble_disagreement_case():
    """Verifies choice agreement when seeds disagree."""
    res = EnsembleDecisionResult(
        choice="Option 1",
        choice_index=0,
        probabilities=[0.50, 0.40, 0.10],
        scores=[1.0, 0.8, -0.5],
        options=["Option 1", "Option 2", "Option 3"],
        seed_choices={42: 0, 43: 0, 44: 1},
        choice_agreement=2.0 / 3.0,
    )
    assert pytest.approx(res.choice_agreement, rel=1e-4) == 0.6667
    assert res.should_abstain(min_agreement=1.0) is True
    assert res.should_abstain(min_agreement=0.66) is False


def test_config_for_seed():
    """Verifies seed-specific config construction with frozen hashes."""
    for s in [42, 43, 44]:
        cfg = InvariantOneConfig.for_seed(seed=s)
        assert cfg.primary_seed == s
        assert cfg.lora_sha256 == FROZEN_SEEDS_HASHES[s]["lora"]
        assert cfg.head_sha256 == FROZEN_SEEDS_HASHES[s]["head"]

    with pytest.raises(ValueError, match="Unknown seed 99"):
        InvariantOneConfig.for_seed(seed=99)


def test_checkpoint_loader_resolves_all_three_seeds():
    """Verifies resolve_checkpoint_paths locates checkpoints for seeds 42, 43, and 44."""
    for s in [42, 43, 44]:
        lora_p, head_p = resolve_checkpoint_paths(seed=s)
        assert lora_p.exists()
        assert head_p.exists()
        # Verify hashes
        verify_checkpoint_hash(lora_p, FROZEN_SEEDS_HASHES[s]["lora"], strict=True)
        verify_checkpoint_hash(head_p, FROZEN_SEEDS_HASHES[s]["head"], strict=True)


def test_format_summary():
    """Verifies clean human-readable formatting of decision results."""
    res = DecisionResult(
        choice="Close inlet valve",
        choice_index=0,
        probabilities=[0.82, 0.11, 0.05, 0.02],
        scores=[2.0, 0.5, 0.0, -1.0],
        options=["Close inlet valve", "Maintain state", "Open bypass", "Increase pressure"],
    )
    s = res.format_summary()
    assert "choice: Close inlet valve" in s
    assert "confidence:          0.8200" in s
    assert "margin:              0.7100" in s

    ens_res = EnsembleDecisionResult(
        choice="Close inlet valve",
        choice_index=0,
        probabilities=[0.82, 0.11, 0.05, 0.02],
        scores=[2.0, 0.5, 0.0, -1.0],
        options=["Close inlet valve", "Maintain state", "Open bypass", "Increase pressure"],
        seed_choices={42: 0, 43: 0, 44: 0},
        choice_agreement=1.0,
        ensemble_std=0.025,
        mutual_information=0.012,
    )
    s_ens = ens_res.format_summary()
    assert "choice agreement:     3/3" in s_ens
    assert "ensemble std:         0.0250" in s_ens
    assert "mutual information:   0.0120" in s_ens


def test_invariantone_profiles():
    """Verifies that operational abstention profiles evaluate correctly."""
    from invariantone import InvariantOneProfile

    # Profile thresholds existence
    assert InvariantOneProfile.STANDARD.mode == "default"
    assert InvariantOneProfile.HIGH_ASSURANCE.mode == "default"
    assert InvariantOneProfile.ENSEMBLE_ASSURANCE.mode == "assurance"
    assert InvariantOneProfile.ULTRA_ASSURANCE.mode == "assurance"

    # Confident decision
    res_confident = DecisionResult(
        choice="A",
        choice_index=0,
        probabilities=[0.90, 0.05, 0.03, 0.02],
        scores=[3.0, 0.0, -1.0, -2.0],
        options=["A", "B", "C", "D"],
    )
    assert not res_confident.should_abstain(profile=InvariantOneProfile.STANDARD)
    assert not res_confident.should_abstain(profile=InvariantOneProfile.HIGH_ASSURANCE)

    # Moderate decision (confidence 0.70)
    res_moderate = DecisionResult(
        choice="A",
        choice_index=0,
        probabilities=[0.70, 0.20, 0.05, 0.05],
        scores=[2.0, 1.0, 0.0, 0.0],
        options=["A", "B", "C", "D"],
    )
    assert not res_moderate.should_abstain(profile=InvariantOneProfile.STANDARD)
    assert res_moderate.should_abstain(profile=InvariantOneProfile.HIGH_ASSURANCE)  # 0.70 < 0.85

    # Ensemble high disagreement
    ens_disagreement = EnsembleDecisionResult(
        choice="A",
        choice_index=0,
        probabilities=[0.78, 0.12, 0.05, 0.05],
        scores=[2.0, 0.5, 0.0, 0.0],
        options=["A", "B", "C", "D"],
        seed_choices={42: 0, 43: 1, 44: 0},
        choice_agreement=2/3,
        ensemble_std=0.18,
        mutual_information=0.045,
    )
    assert ens_disagreement.should_abstain(profile=InvariantOneProfile.ENSEMBLE_ASSURANCE)  # agreement < 1.0
    assert ens_disagreement.should_abstain(profile=InvariantOneProfile.ULTRA_ASSURANCE)

    # Ensemble consensus & low spread
    ens_consensus = EnsembleDecisionResult(
        choice="A",
        choice_index=0,
        probabilities=[0.85, 0.05, 0.05, 0.05],
        scores=[3.0, 0.0, 0.0, 0.0],
        options=["A", "B", "C", "D"],
        seed_choices={42: 0, 43: 0, 44: 0},
        choice_agreement=1.0,
        ensemble_std=0.02,
        mutual_information=0.010,
    )
    assert not ens_consensus.should_abstain(profile=InvariantOneProfile.ENSEMBLE_ASSURANCE)
    assert not ens_consensus.should_abstain(profile=InvariantOneProfile.ULTRA_ASSURANCE)


def test_conformal_finite_sample_rank_formula():
    """Audits the finite-sample conformal quantile rank formula."""
    import math

    n_cal = 224
    # k = ceil((n_cal + 1) * (1 - alpha))
    k_90 = math.ceil((n_cal + 1) * 0.90)
    k_95 = math.ceil((n_cal + 1) * 0.95)

    assert k_90 == 203  # ceil(225 * 0.90) = 203
    assert k_95 == 214  # ceil(225 * 0.95) = 213.75 -> 214


