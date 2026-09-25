"""
InvariantOne Operational Abstention Profiles.

Defines standardized operational profiles derived from prospective evaluation on
independent validation data (N_eval=226).

GOVERNANCE & STATISTICAL NOTICE:
- These profiles represent validation-set operating characteristics, NOT unconditional
  future performance guarantees.
- Retained accuracy and coverage numbers were measured prospectively on independent
  in-distribution validation data (val_v4_real + val_v4_relational).
- Performance characteristics may differ under real-world domain or operator shift.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class InvariantOneProfile(str, Enum):
    """
    Standardized operational abstention profiles for InvariantOne.

    Attributes:
        STANDARD: Single checkpoint (Seed 42). Standard confidence/margin filter.
            Validation characteristics: 90.7% coverage, 80.5% retained accuracy (+2.6% gain).
        HIGH_ASSURANCE: Single checkpoint (Seed 42). Conservative confidence threshold.
            Validation characteristics: 74.3% coverage, 88.1% retained accuracy (+10.2% gain).
        ENSEMBLE_ASSURANCE: 3-Checkpoint Ensemble (Seeds 42, 43, 44). Multi-signal consensus.
            Validation characteristics: 87.6% coverage, 81.8% retained accuracy (+4.8% gain).
        ULTRA_ASSURANCE: 3-Checkpoint Ensemble (Seeds 42, 43, 44). Strict epistemic filter.
            Validation characteristics: 65.9% coverage, 91.95% retained accuracy (+15.0% gain).
    """

    STANDARD = "standard"
    HIGH_ASSURANCE = "high_assurance"
    ENSEMBLE_ASSURANCE = "ensemble_assurance"
    ULTRA_ASSURANCE = "ultra_assurance"

    @property
    def mode(self) -> str:
        """Target model execution mode: 'default' (single seed) or 'assurance' (ensemble)."""
        if self in (InvariantOneProfile.STANDARD, InvariantOneProfile.HIGH_ASSURANCE):
            return "default"
        return "assurance"

    @property
    def thresholds(self) -> dict[str, float]:
        """
        Calibrated decision thresholds for this profile.

        Threshold values correspond exactly to prospective calibration on N_cal=224:
        - STANDARD: min_confidence = 0.647055 (target 85% coverage, realized 90.71%)
        - HIGH_ASSURANCE: min_confidence = 0.850702 (target 70% coverage, realized 74.34%)
        - ENSEMBLE_ASSURANCE: agreement = 1.0, max_ensemble_std = 0.166447, min_confidence = 0.593801
        - ULTRA_ASSURANCE: agreement = 1.0, max_mutual_information = 0.033925, min_confidence = 0.753304
        """
        if self == InvariantOneProfile.STANDARD:
            return {"min_confidence": 0.6470547318458557}
        elif self == InvariantOneProfile.HIGH_ASSURANCE:
            return {"min_confidence": 0.8507021671233827}
        elif self == InvariantOneProfile.ENSEMBLE_ASSURANCE:
            return {
                "min_agreement": 1.0,
                "max_ensemble_std": 0.16644661370220135,
                "min_confidence": 0.5938005646069845,
            }
        elif self == InvariantOneProfile.ULTRA_ASSURANCE:
            return {
                "min_agreement": 1.0,
                "max_mutual_information": 0.03392471649987813,
                "min_confidence": 0.7533036669095358,
            }
        return {}


    @property
    def operating_characteristics(self) -> dict[str, Any]:
        """Validation-set performance characteristics on independent N_eval=226 test items."""
        if self == InvariantOneProfile.STANDARD:
            return {
                "coverage": 0.9071,
                "retained_accuracy": 0.8049,
                "baseline_accuracy": 0.7788,
                "accuracy_gain": 0.0261,
                "errors_eliminated_pct": 0.20,
                "latency_profile_ms": 82.39,
            }
        elif self == InvariantOneProfile.HIGH_ASSURANCE:
            return {
                "coverage": 0.7434,
                "retained_accuracy": 0.8810,
                "baseline_accuracy": 0.7788,
                "accuracy_gain": 0.1022,
                "errors_eliminated_pct": 0.60,
                "latency_profile_ms": 82.39,
            }
        elif self == InvariantOneProfile.ENSEMBLE_ASSURANCE:
            return {
                "coverage": 0.8761,
                "retained_accuracy": 0.8182,
                "baseline_accuracy": 0.7699,
                "accuracy_gain": 0.0483,
                "errors_eliminated_pct": 0.3077,
                "latency_profile_ms": 248.50,
            }
        elif self == InvariantOneProfile.ULTRA_ASSURANCE:
            return {
                "coverage": 0.6593,
                "retained_accuracy": 0.9195,
                "baseline_accuracy": 0.7699,
                "accuracy_gain": 0.1496,
                "errors_eliminated_pct": 0.7692,
                "latency_profile_ms": 248.50,
            }
        return {}

    def should_abstain(self, result: Any) -> bool:
        """
        Determines whether the given decision result should abstain according to this profile.

        Args:
            result: A DecisionResult or EnsembleDecisionResult instance.

        Returns:
            True if any profile threshold is breached, False otherwise.
        """
        return result.should_abstain(profile=self)
