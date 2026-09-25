"""
Calibration Diagnostics & Reliability Analysis for InvariantOne.

Computes:
1. Reliability diagram data (10-bin confidence vs empirical accuracy)
2. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
3. Decision margin distributions (s_correct - s_second_best)
4. Shannon entropy distributions over predicted probabilities
5. Score compression analysis (raw score dynamic range, standard deviation)
6. Negative Log-Likelihood (NLL) and Brier score
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np


def compute_reliability_diagram(
    probabilities: Sequence[Mapping[str, float]],
    targets: Sequence[str],
    num_bins: int = 10,
) -> dict[str, Any]:
    """
    Computes 10-bin reliability diagram statistics.
    Returns bin ranges, bin sample counts, mean confidence, mean accuracy, and calibration gap.
    """
    if not targets:
        return {"bins": [], "ece": 0.0, "mce": 0.0}

    confidences = []
    accuracies = []

    for prob_map, target in zip(probabilities, targets):
        if not prob_map:
            confidences.append(0.0)
            accuracies.append(0.0)
            continue
        top_opt, top_conf = max(prob_map.items(), key=lambda x: x[1])
        confidences.append(float(top_conf))
        accuracies.append(1.0 if top_opt == target else 0.0)

    confs = np.array(confidences)
    accs = np.array(accuracies)
    n = len(confs)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins = []
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            in_bin = (confs >= low) & (confs <= high)
        else:
            in_bin = (confs >= low) & (confs < high)

        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(accs[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            gap = abs(bin_acc - bin_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
        else:
            bin_acc = 0.0
            bin_conf = float((low + high) / 2.0)
            gap = 0.0

        bins.append({
            "bin_idx": i,
            "range": [round(float(low), 2), round(float(high), 2)],
            "count": count,
            "mean_confidence": round(bin_conf, 4),
            "empirical_accuracy": round(bin_acc, 4),
            "calibration_gap": round(gap, 4),
        })

    return {
        "bins": bins,
        "ece": round(float(ece), 6),
        "mce": round(float(mce), 6),
        "total_samples": n,
    }


def compute_margin_distribution(
    scores_list: Sequence[Mapping[str, float]],
    targets: Sequence[str],
) -> dict[str, Any]:
    """
    Computes margin: s(target) - max_{opt != target} s(opt).
    Positive margin means correct classification.
    """
    margins = []
    for scores, target in zip(scores_list, targets):
        if not scores or target not in scores:
            continue
        target_score = float(scores[target])
        other_scores = [float(v) for k, v in scores.items() if k != target]
        if other_scores:
            runner_up = max(other_scores)
            margin = target_score - runner_up
            margins.append(margin)

    if not margins:
        return {"count": 0}

    arr = np.array(margins)
    return {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "fraction_positive": round(float(np.mean(arr > 0)), 4),
    }


def compute_entropy_distribution(
    probabilities: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """
    Computes Shannon entropy H(p) = -sum p_i log2(p_i) for each prediction.
    """
    entropies = []
    for prob_map in probabilities:
        if not prob_map:
            continue
        vals = [float(v) for v in prob_map.values() if v > 1e-12]
        if not vals:
            continue
        # Shannon entropy in bits
        h = -sum(p * math.log2(p) for p in vals)
        entropies.append(h)

    if not entropies:
        return {"count": 0}

    arr = np.array(entropies)
    return {
        "count": len(arr),
        "mean_bits": round(float(np.mean(arr)), 4),
        "std_bits": round(float(np.std(arr)), 4),
        "median_bits": round(float(np.median(arr)), 4),
        "p10_bits": round(float(np.percentile(arr, 10)), 4),
        "p90_bits": round(float(np.percentile(arr, 90)), 4),
        "min_bits": round(float(np.min(arr)), 4),
        "max_bits": round(float(np.max(arr)), 4),
    }


def compute_score_compression(
    raw_scores_list: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """
    Computes dynamic range and standard deviation of raw scores across options.
    Detects score compression (e.g. all options scoring ~0.33 vs distinct logit peaks).
    """
    ranges = []
    stds = []
    for scores in raw_scores_list:
        if not scores or len(scores) < 2:
            continue
        vals = np.array(list(scores.values()), dtype=np.float64)
        ranges.append(float(np.max(vals) - np.min(vals)))
        stds.append(float(np.std(vals)))

    if not ranges:
        return {"count": 0}

    return {
        "count": len(ranges),
        "mean_dynamic_range": round(float(np.mean(ranges)), 4),
        "median_dynamic_range": round(float(np.median(ranges)), 4),
        "mean_score_std": round(float(np.mean(stds)), 4),
        "median_score_std": round(float(np.median(stds)), 4),
    }


def compute_full_calibration_diagnostics(
    probabilities: Sequence[Mapping[str, float]],
    targets: Sequence[str],
    raw_scores_list: Sequence[Mapping[str, float]] | None = None,
    num_bins: int = 10,
) -> dict[str, Any]:
    """
    Aggregates all calibration, margin, entropy, and score compression diagnostics.
    """
    rel = compute_reliability_diagram(probabilities, targets, num_bins=num_bins)
    prob_margins = compute_margin_distribution(probabilities, targets)
    entropy = compute_entropy_distribution(probabilities)

    # Compute NLL and Brier score
    nll_vals = []
    brier_vals = []
    for prob_map, target in zip(probabilities, targets):
        if not prob_map:
            continue
        p_target = float(prob_map.get(target, 0.0))
        nll_vals.append(-math.log(max(1e-12, p_target)))

        # Brier score: sum_k (p_k - y_k)^2
        brier = 0.0
        for opt, p in prob_map.items():
            y = 1.0 if opt == target else 0.0
            brier += (p - y) ** 2
        brier_vals.append(brier)

    raw_margins = (
        compute_margin_distribution(raw_scores_list, targets)
        if raw_scores_list
        else None
    )
    compression = (
        compute_score_compression(raw_scores_list)
        if raw_scores_list
        else None
    )

    return {
        "expected_calibration_error": rel["ece"],
        "maximum_calibration_error": rel["mce"],
        "negative_log_likelihood": round(float(np.mean(nll_vals)), 6) if nll_vals else 0.0,
        "brier_score": round(float(np.mean(brier_vals)), 6) if brier_vals else 0.0,
        "reliability_bins": rel["bins"],
        "probability_margins": prob_margins,
        "raw_score_margins": raw_margins,
        "entropy_distribution": entropy,
        "score_compression": compression,
    }
