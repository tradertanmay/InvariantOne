"""
Probability calibration metrics and post-hoc temperature scaling.
Evaluates Expected Calibration Error (ECE), adaptive ECE, and fits optimal post-hoc temperature.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Mapping

import numpy as np
from scipy.optimize import minimize_scalar


def expected_calibration_error(
    probabilities: Sequence[Mapping[str, float]],
    targets: Sequence[str],
    num_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width confidence bins in [0, 1].
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if not targets:
        return 0.0

    confidences = []
    accuracies = []

    for prob_map, target in zip(probabilities, targets):
        if not prob_map:
            confidences.append(0.0)
            accuracies.append(0.0)
            continue
        # Find top predicted choice and its confidence
        top_choice = max(prob_map.items(), key=lambda x: x[1])
        confidences.append(float(top_choice[1]))
        accuracies.append(1.0 if top_choice[0] == target else 0.0)

    confs = np.array(confidences)
    accs = np.array(accuracies)
    n = len(confs)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Include left boundary, right boundary inclusive on the last bin
        if i == num_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accs[in_bin])
            bin_conf = np.mean(confs[in_bin])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return round(float(ece), 6)


class TemperatureScaler:
    """
    Post-hoc temperature scaling fitted strictly on a held-out calibration split.
    Scales logits/scores s -> s / tau to minimize negative log-likelihood.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(
        self,
        raw_scores_list: Sequence[Mapping[str, float]],
        targets: Sequence[str],
        bounds: tuple[float, float] = (0.1, 10.0),
    ) -> float:
        """
        Fit optimal temperature tau on calibration predictions using scipy optimizer.
        """
        if not raw_scores_list or not targets:
            self.temperature = 1.0
            return 1.0

        def nll_objective(tau: float) -> float:
            total_nll = 0.0
            for scores, target in zip(raw_scores_list, targets):
                opts = list(scores.keys())
                vals = np.array([scores[o] for o in opts], dtype=np.float64)
                # Scaled softmax
                scaled = vals / tau
                max_s = np.max(scaled)
                exp_s = np.exp(scaled - max_s)
                probs = exp_s / np.sum(exp_s)

                target_idx = opts.index(target) if target in opts else -1
                p_true = probs[target_idx] if target_idx >= 0 else 1e-12
                total_nll += -math.log(max(1e-12, float(p_true)))
            return total_nll / len(raw_scores_list)

        res = minimize_scalar(nll_objective, bounds=bounds, method="bounded")
        self.temperature = round(float(res.x), 4)
        return self.temperature

    def scale_probabilities(
        self,
        raw_scores: Mapping[str, float],
    ) -> dict[str, float]:
        """Apply fitted temperature to raw logits/scores and return calibrated probabilities."""
        opts = list(raw_scores.keys())
        vals = np.array([raw_scores[o] for o in opts], dtype=np.float64)
        scaled = vals / self.temperature
        max_s = np.max(scaled)
        exp_s = np.exp(scaled - max_s)
        probs = exp_s / np.sum(exp_s)
        return {opt: round(float(p), 6) for opt, p in zip(opts, probs)}
