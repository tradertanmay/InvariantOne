"""
Typed Decision Result objects for InvariantOne.

Provides structured, validated return types for single decisions and batched inferences.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DecisionResult:
    """
    Structured result returned by InvariantOne.decide().

    Attributes:
        choice: Selected option string (argmax over probabilities).
        choice_index: 0-indexed integer position of the selected option.
        probabilities: Calibrated probability distribution over candidate options (sums to 1.0).
        scores: Uncalibrated raw scalar scores output by the comparative head.
        options: Ordered list of candidate options evaluated.
        state: Input state string.
        question: Input question string.
        temperature: Calibration temperature tau applied during softmax.
        latency_ms: End-to-end inference latency in milliseconds.
        device: Device string on which inference was executed.
        model_version: InvariantOne release version string.
        metadata: Arbitrary additional diagnostic metadata.
    """

    choice: str
    choice_index: int
    probabilities: list[float]
    scores: list[float]
    options: list[str]
    state: str = ""
    question: str = ""
    temperature: float = 1.0091
    latency_ms: float = 0.0
    device: str = "cpu"
    model_version: str = "1.0.0"
    abstained: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates mathematical invariants of the decision result."""
        # 1. Finite scores and probabilities
        for i, s in enumerate(self.scores):
            if not math.isfinite(s):
                raise ValueError(f"Non-finite score encountered at index {i}: {s}")
        for i, p in enumerate(self.probabilities):
            if not math.isfinite(p):
                raise ValueError(f"Non-finite probability encountered at index {i}: {p}")
            if p < 0.0:
                raise ValueError(f"Negative probability encountered at index {i}: {p}")

        # 2. Probability normalization
        prob_sum = sum(self.probabilities)
        if not (0.99 <= prob_sum <= 1.01):
            raise ValueError(f"Probabilities do not sum to 1.0 (sum={prob_sum})")

        # 3. Choice index validity
        k_opts = len(self.options)
        if not (0 <= self.choice_index < k_opts):
            raise ValueError(f"choice_index {self.choice_index} is out of bounds for {k_opts} options")
        if self.options[self.choice_index] != self.choice:
            raise ValueError(
                f"choice '{self.choice}' does not match options[{self.choice_index}]='{self.options[self.choice_index]}'"
            )

    @property
    def confidence(self) -> float:
        """
        Model-assigned calibrated decision probability: max(probabilities).

        NOTE: Softmax confidence represents the model-assigned calibrated decision
        probability under its calibrated temperature scaling, NOT the probability
        that the answer is objectively correct in the real world.
        """
        return max(self.probabilities) if self.probabilities else 0.0

    @property
    def margin(self) -> float:
        """
        Decision margin between the top probability and second-highest probability:
        margin = p_(1) - p_(2). For binary decisions (K=2), margin = 2*p_(1) - 1.
        """
        if len(self.probabilities) < 2:
            return self.confidence
        sorted_p = sorted(self.probabilities, reverse=True)
        return float(sorted_p[0] - sorted_p[1])

    @property
    def entropy(self) -> float:
        """
        Shannon entropy of the choice distribution:
        H(p) = -sum_i p_i * ln(p_i)
        """
        h = 0.0
        for p in self.probabilities:
            if p > 1e-12:
                h -= p * math.log(p)
        return float(h)

    @property
    def normalized_entropy(self) -> float:
        """
        Shannon entropy normalized to [0, 1]:
        normalized_entropy = H(p) / ln(K)

        Interpretation:
        0.0 = maximally concentrated (certain)
        1.0 = maximally uncertain (uniform distribution across K options)
        """
        k = len(self.probabilities)
        if k <= 1:
            return 0.0
        h_norm = self.entropy / math.log(k)
        return float(max(0.0, min(1.0, h_norm)))

    def should_abstain(
        self,
        min_confidence: float | None = None,
        min_margin: float | None = None,
        profile: Any | None = None,
    ) -> bool:
        """
        Determines whether the model should abstain from making a decision
        based on confidence, margin, or a standardized operational profile.

        Args:
            min_confidence: Minimum required decision confidence (max probability).
            min_margin: Minimum required margin between top-1 and top-2 candidates.
            profile: An optional InvariantOneProfile instance specifying calibrated thresholds.

        Returns:
            True if any abstention condition is triggered, False otherwise.
        """
        if profile is not None:
            t = profile.thresholds
            if min_confidence is None:
                min_confidence = t.get("min_confidence")
            if min_margin is None:
                min_margin = t.get("min_margin")

        if min_confidence is not None and self.confidence < min_confidence:
            return True
        if min_margin is not None and self.margin < min_margin:
            return True
        return False


    def format_summary(self) -> str:
        """Formats result into a clean, human-readable decision and uncertainty summary."""
        lines = [f"choice: {self.choice}", "", "probabilities:"]
        max_opt_len = max(len(opt) for opt in self.options) if self.options else 20
        for opt, prob in zip(self.options, self.probabilities):
            lines.append(f"  {opt:<{max_opt_len + 4}} {prob:.4f}")
        lines.append("")
        lines.append(f"confidence:          {self.confidence:.4f}")
        lines.append(f"margin:              {self.margin:.4f}")
        lines.append(f"normalized_entropy:  {self.normalized_entropy:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Converts result to a JSON-serializable dictionary including uncertainty metrics."""
        d = asdict(self)
        d["confidence"] = self.confidence
        d["margin"] = self.margin
        d["entropy"] = self.entropy
        d["normalized_entropy"] = self.normalized_entropy
        return d

    def __repr__(self) -> str:
        return (
            f"DecisionResult(choice={self.choice!r}, "
            f"choice_index={self.choice_index}, "
            f"confidence={round(self.confidence, 4)}, "
            f"margin={round(self.margin, 4)}, "
            f"entropy={round(self.entropy, 4)}, "
            f"probabilities={[round(p, 4) for p in self.probabilities]})"
        )


@dataclass(frozen=True)
class EnsembleDecisionResult(DecisionResult):
    """
    Structured result returned by InvariantOneEnsemble.decide().

    Aggregates multi-checkpoint predictions (e.g., Seeds 42, 43, 44) to expose
    model uncertainty, ensemble spread, checkpoint disagreement, and mutual information.

    GOVERNANCE AND STATISTICAL TERMINOLOGY:
    - [ensemble_min, ensemble_max] is an observed model range / ensemble spread across seeds.
      It is NOT a formal 95% confidence interval and must not be described as such.
    - Softmax confidence represents model-assigned calibrated decision probability,
      not objective correctness probability.
    - Mutual information is an ensemble disagreement / epistemic-uncertainty proxy,
      not exact Bayesian epistemic uncertainty.

    Attributes:
        seed_probabilities: Dict mapping seed integer to probability list [K].
        seed_choices: Dict mapping seed integer to selected choice integer index.
        ensemble_mean_probability: Mean probability for the selected option across seeds.
        ensemble_std: Sample standard deviation of probability for the selected option across seeds.
        ensemble_min: Minimum probability for the selected option across seeds (observed range lower bound).
        ensemble_max: Maximum probability for the selected option across seeds (observed range upper bound).
        choice_agreement: Fraction of seeds agreeing with the ensemble choice (e.g. 2/3 = 0.6667).
        predictive_entropy: Shannon entropy of the ensemble mean probability distribution H(p_bar).
        expected_entropy: Mean Shannon entropy across individual checkpoint distributions E[H(p)].
        mutual_information: Disagreement proxy MI = H(p_bar) - E[H(p)] (epistemic uncertainty proxy).
        ensemble_stds: Per-option standard deviations across seeds.
        ensemble_mins: Per-option minimum probabilities across seeds.
        ensemble_maxs: Per-option maximum probabilities across seeds.
    """

    seed_probabilities: dict[int, list[float]] = field(default_factory=dict)
    seed_choices: dict[int, int] = field(default_factory=dict)
    ensemble_mean_probability: float = 0.0
    ensemble_std: float = 0.0
    ensemble_min: float = 0.0
    ensemble_max: float = 0.0
    choice_agreement: float = 1.0
    predictive_entropy: float = 0.0
    expected_entropy: float = 0.0
    mutual_information: float = 0.0
    ensemble_stds: list[float] = field(default_factory=list)
    ensemble_mins: list[float] = field(default_factory=list)
    ensemble_maxs: list[float] = field(default_factory=list)

    def should_abstain(
        self,
        min_confidence: float | None = None,
        min_margin: float | None = None,
        min_agreement: float | None = None,
        max_ensemble_std: float | None = None,
        max_mutual_information: float | None = None,
        profile: Any | None = None,
    ) -> bool:
        """
        Determines whether the ensemble should abstain from a decision based on
        confidence, margin, agreement, ensemble spread, mutual information, or
        a standardized operational profile.

        Args:
            min_confidence: Minimum required ensemble confidence (max mean probability).
            min_margin: Minimum required margin between top-1 and top-2 mean probabilities.
            min_agreement: Minimum fraction of seeds that must agree (e.g. 1.0 for unanimous).
            max_ensemble_std: Maximum allowed standard deviation across seeds for the top choice.
            max_mutual_information: Maximum allowed mutual information (epistemic disagreement proxy).
            profile: An optional InvariantOneProfile instance specifying calibrated thresholds.

        Returns:
            True if any abstention condition is triggered, False otherwise.
        """
        if profile is not None:
            t = profile.thresholds
            if min_confidence is None:
                min_confidence = t.get("min_confidence")
            if min_margin is None:
                min_margin = t.get("min_margin")
            if min_agreement is None:
                min_agreement = t.get("min_agreement")
            if max_ensemble_std is None:
                max_ensemble_std = t.get("max_ensemble_std")
            if max_mutual_information is None:
                max_mutual_information = t.get("max_mutual_information")

        if super().should_abstain(min_confidence=min_confidence, min_margin=min_margin):
            return True
        if min_agreement is not None and self.choice_agreement < min_agreement:
            return True
        if max_ensemble_std is not None and self.ensemble_std > max_ensemble_std:
            return True
        if max_mutual_information is not None and self.mutual_information > max_mutual_information:
            return True
        return False


    def to_dict(self) -> dict[str, Any]:
        """Converts result to a JSON-serializable dictionary."""
        d = super().to_dict()
        d["predictive_entropy"] = self.predictive_entropy
        d["expected_entropy"] = self.expected_entropy
        d["mutual_information"] = self.mutual_information
        return d

    def format_summary(self) -> str:
        """Formats ensemble result with spread, agreement, and disagreement metrics."""
        base_summary = super().format_summary()
        n_seeds = len(self.seed_choices) if self.seed_choices else 3
        n_agreed = int(round(self.choice_agreement * n_seeds))
        lines = [
            base_summary,
            "",
            f"choice agreement:     {n_agreed}/{n_seeds}",
            f"ensemble std:         {self.ensemble_std:.4f}",
            f"mutual information:   {self.mutual_information:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"EnsembleDecisionResult(choice={self.choice!r}, "
            f"choice_index={self.choice_index}, "
            f"ensemble_mean_prob={round(self.ensemble_mean_probability, 4)}, "
            f"ensemble_std={round(self.ensemble_std, 4)}, "
            f"choice_agreement={round(self.choice_agreement, 4)}, "
            f"predictive_entropy={round(self.predictive_entropy, 4)}, "
            f"mutual_information={round(self.mutual_information, 4)})"
        )


@dataclass(frozen=True)
class BatchDecisionResult:
    """
    Structured collection of results returned by InvariantOne.decide_batch().
    """

    results: list[DecisionResult]
    total_latency_ms: float = 0.0
    batch_size: int = 0

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, idx: int) -> DecisionResult:
        return self.results[idx]

    def __iter__(self):
        return iter(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_latency_ms": self.total_latency_ms,
            "batch_size": len(self.results),
        }
