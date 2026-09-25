"""
Calibration module for InvariantOne.

Applies frozen temperature scaling (tau* = 1.0091) to raw logits produced by
the DirectComparativeHead, ensuring calibrated confidence outputs without retraining.
"""

from __future__ import annotations

import math
from typing import Sequence, overload

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_CALIBRATION_TEMPERATURE = 1.0091


@overload
def calibrated_softmax(scores: torch.Tensor, tau: float = DEFAULT_CALIBRATION_TEMPERATURE) -> torch.Tensor:
    ...


@overload
def calibrated_softmax(scores: Sequence[float], tau: float = DEFAULT_CALIBRATION_TEMPERATURE) -> list[float]:
    ...


def calibrated_softmax(
    scores: torch.Tensor | Sequence[float],
    tau: float = DEFAULT_CALIBRATION_TEMPERATURE,
) -> torch.Tensor | list[float]:
    """
    Computes temperature-scaled softmax probabilities over raw comparative scores:
        P_i = exp(score_i / tau) / sum_j exp(score_j / tau)

    Args:
        scores: 1D or 2D tensor or sequence of float scores.
        tau: Temperature scaling parameter (frozen tau* = 1.0091).

    Returns:
        Calibrated probabilities matching the input container type.

    Raises:
        ValueError: If tau <= 0 or if scores contain NaN/Inf.
    """
    if tau <= 0.0:
        raise ValueError(f"Calibration temperature tau must be strictly positive, got {tau}")

    if isinstance(scores, torch.Tensor):
        if not torch.all(torch.isfinite(scores)):
            raise ValueError("Input scores contain NaN or Inf values")
        probs = F.softmax(scores.float() / tau, dim=-1)
        if not torch.all(torch.isfinite(probs)):
            raise ValueError("Computed probabilities contain NaN or Inf values")
        return probs

    # Python sequence of floats
    scores_list = list(scores)
    for s in scores_list:
        if not math.isfinite(s):
            raise ValueError(f"Input score {s} is not finite")

    scores_arr = np.array(scores_list, dtype=np.float64)
    scaled = scores_arr / tau
    # Numerically stable softmax
    shifted = scaled - np.max(scaled)
    exp_shifted = np.exp(shifted)
    probs_arr = exp_shifted / np.sum(exp_shifted)

    for p in probs_arr:
        if not math.isfinite(p):
            raise ValueError(f"Computed probability {p} is not finite")

    return [float(p) for p in probs_arr]
