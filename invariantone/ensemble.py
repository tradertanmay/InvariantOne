"""
InvariantOne Multi-Checkpoint Ensemble & Epistemic Uncertainty Module.

Provides InvariantOneEnsemble for combining predictions from independently seeded
replications (Seeds 42, 43, 44) to evaluate model uncertainty, checkpoint disagreement,
and mutual information.

GOVERNANCE & STATISTICAL TERMINOLOGY:
- [ensemble_min, ensemble_max] represents observed model range / ensemble spread across seeds.
  It is NOT a formal 95% confidence interval and must not be described as such.
- Softmax confidence represents model-assigned calibrated decision probability,
  not objective correctness probability.
- Mutual information is an ensemble disagreement / epistemic-uncertainty proxy,
  not exact Bayesian epistemic uncertainty.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn

from invariantone.architectures.direct_comparative_head import DirectComparativeHead
from invariantone.calibration import calibrated_softmax
from invariantone.config import (
    FROZEN_SEEDS_HASHES,
    InvariantOneConfig,
)
from invariantone.loading.checkpoint_loader import (
    load_backbone_with_lora,
    load_comparative_head,
    resolve_checkpoint_paths,
)
from invariantone.loading.hash_verification import verify_checkpoint_hash
from invariantone.model import InvariantOne
from invariantone.result import EnsembleDecisionResult
from invariantone.runtime.batching import encode_candidate_branches
from invariantone.runtime.device import resolve_device, resolve_dtype


class InvariantOneEnsemble:
    """
    Ensemble of independently trained InvariantOne checkpoints (D5-L4 / Nop=16).

    Coordinates multi-seed inference across Seeds 42, 43, and 44 to provide
    ensemble-calibrated probabilities, checkpoint disagreement, and uncertainty metrics.
    """

    def __init__(
        self,
        models: dict[int, InvariantOne],
        seeds: list[int],
        device: torch.device,
        dtype: torch.dtype,
        execution_mode: str = "multi_model",
    ) -> None:
        self.models = models
        self.seeds = sorted(seeds)
        self.device = device
        self.dtype = dtype
        self.execution_mode = execution_mode

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "invariantone-v1",
        seeds: Sequence[int] = (42, 43, 44),
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "bfloat16",
        strict: bool = True,
        execution_mode: Literal["auto", "multi_model", "adapter_swap"] = "auto",
        checkpoint_dir: str | Path | None = None,
    ) -> InvariantOneEnsemble:
        """
        Loads an ensemble of InvariantOne checkpoints across specified seeds.

        Args:
            pretrained_model_name_or_path: Path or identifier to checkpoint directory.
            seeds: List or tuple of replication seed integers (e.g. [42, 43, 44]).
            device: Target execution device ('auto', 'cuda', 'mps', 'cpu').
            dtype: Target precision dtype ('bfloat16', 'float16', 'float32').
            strict: If True, strictly verifies cryptographic SHA-256 digests.
            execution_mode: 'multi_model' keeps separate models resident; 'adapter_swap'
                shares one backbone and swaps adapter weights; 'auto' selects based on device.
            checkpoint_dir: Explicit override for checkpoint directory.

        Returns:
            An instantiated InvariantOneEnsemble ready for inference.
        """
        resolved_dev = resolve_device(device)
        resolved_dt = resolve_dtype(dtype, resolved_dev)

        seeds_list = list(seeds)
        if not seeds_list:
            raise ValueError("seeds list cannot be empty")

        for s in seeds_list:
            if s not in FROZEN_SEEDS_HASHES:
                raise ValueError(
                    f"Unsupported seed {s}. Supported seeds: {list(FROZEN_SEEDS_HASHES.keys())}"
                )

        # Decide execution mode
        if execution_mode == "auto":
            # On CUDA with sufficient memory, multi_model keeps all resident.
            # On CPU or MPS, adapter_swap saves RAM.
            if resolved_dev.type == "cuda":
                execution_mode = "multi_model"
            else:
                execution_mode = "adapter_swap"

        models: dict[int, InvariantOne] = {}

        if execution_mode == "multi_model":
            # Load independent resident instances
            for s in seeds_list:
                m = InvariantOne.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    device=resolved_dev,
                    dtype=resolved_dt,
                    strict=strict,
                    seed=s,
                    checkpoint_dir=checkpoint_dir,
                )
                models[s] = m
        else:
            # Load first model with base backbone, and keep additional adapter/head weights
            primary_seed = seeds_list[0]
            base_model = InvariantOne.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                device=resolved_dev,
                dtype=resolved_dt,
                strict=strict,
                seed=primary_seed,
                checkpoint_dir=checkpoint_dir,
            )
            models[primary_seed] = base_model

            # For remaining seeds, construct InvariantOne referencing the shared backbone or separate head
            for s in seeds_list[1:]:
                cfg_s = InvariantOneConfig.for_seed(seed=s, strict=strict)
                lora_s, head_s = resolve_checkpoint_paths(
                    pretrained_name_or_path=checkpoint_dir or pretrained_model_name_or_path,
                    seed=s,
                )
                head_obj = load_comparative_head(
                    config=cfg_s,
                    head_path=head_s,
                    device=resolved_dev,
                    dtype=resolved_dt,
                )
                # To maintain isolation in adapter_swap mode, we create lightweight instance
                # with shared tokenizer but distinct head
                inst = InvariantOne(
                    config=cfg_s,
                    model=base_model.model,
                    head=head_obj,
                    tokenizer=base_model.tokenizer,
                    device=resolved_dev,
                    dtype=resolved_dt,
                    lora_path=lora_s,
                    head_path=head_s,
                )
                models[s] = inst

        return cls(
            models=models,
            seeds=seeds_list,
            device=resolved_dev,
            dtype=resolved_dt,
            execution_mode=execution_mode,
        )

    def decide(
        self,
        state: str,
        question: str,
        options: Sequence[str],
        branch_batch_size: int | None = None,
        abstain_threshold: float | None = None,
    ) -> EnsembleDecisionResult:
        """
        Executes ensemble decision across all checkpoints.

        Args:
            state: State description string.
            question: Decision question string.
            options: Sequence of candidate option strings (K >= 2).
            branch_batch_size: Optional chunking size for candidate encoding.
            abstain_threshold: Optional minimum confidence threshold for abstention.

        Returns:
            EnsembleDecisionResult with calibrated ensemble probabilities, disagreement,
            and uncertainty metrics.
        """
        t0 = time.perf_counter()

        k_opts = len(options)
        if k_opts < 2:
            raise ValueError(f"options must contain at least 2 candidates, got {k_opts}")

        seed_probs: dict[int, list[float]] = {}
        seed_scores: dict[int, list[float]] = {}
        seed_choices: dict[int, int] = {}

        # 1. Obtain predictions and probabilities for each seed
        for s in self.seeds:
            m = self.models[s]
            # If in adapter_swap mode with shared model, reload LoRA weights for non-primary seeds
            if self.execution_mode == "adapter_swap" and m.lora_path:
                lora_sd = torch.load(m.lora_path, map_location=self.device, weights_only=True)
                # Update LoRA linear weights in model
                m.model.load_state_dict(lora_sd, strict=False)

            scores_t = m.score(
                state=state,
                question=question,
                options=options,
                branch_batch_size=branch_batch_size,
            )
            probs_t = calibrated_softmax(scores_t, tau=m.config.calibration_temperature)

            scores_list = [float(val) for val in scores_t.cpu().tolist()]
            probs_list = [float(val) for val in probs_t.cpu().tolist()]

            seed_scores[s] = scores_list
            seed_probs[s] = probs_list
            seed_choices[s] = int(torch.argmax(scores_t).item())

        # 2. Compute ensemble mean distribution
        n_seeds = len(self.seeds)
        mean_probs: list[float] = [
            sum(seed_probs[s][k] for s in self.seeds) / n_seeds for k in range(k_opts)
        ]
        mean_scores: list[float] = [
            sum(seed_scores[s][k] for s in self.seeds) / n_seeds for k in range(k_opts)
        ]

        # Selected choice: argmax over ensemble mean probabilities
        choice_idx = int(max(range(k_opts), key=lambda k: mean_probs[k]))
        choice_str = options[choice_idx]

        # 3. Per-option spread, std, min, max
        ensemble_stds: list[float] = []
        ensemble_mins: list[float] = []
        ensemble_maxs: list[float] = []

        for k in range(k_opts):
            vals = [seed_probs[s][k] for s in self.seeds]
            min_k = min(vals)
            max_k = max(vals)
            if n_seeds > 1:
                var_k = sum((v - mean_probs[k]) ** 2 for v in vals) / (n_seeds - 1)
                std_k = math.sqrt(max(0.0, var_k))
            else:
                std_k = 0.0
            ensemble_stds.append(std_k)
            ensemble_mins.append(min_k)
            ensemble_maxs.append(max_k)

        # Selected option metrics
        sel_mean_prob = mean_probs[choice_idx]
        sel_std = ensemble_stds[choice_idx]
        sel_min = ensemble_mins[choice_idx]
        sel_max = ensemble_maxs[choice_idx]

        # 4. Choice agreement
        agree_count = sum(1 for s in self.seeds if seed_choices[s] == choice_idx)
        choice_agreement = agree_count / n_seeds

        # 5. Predictive Entropy: H(p_bar) = -sum p_bar_k * ln(p_bar_k)
        pred_entropy = 0.0
        for p in mean_probs:
            if p > 1e-12:
                pred_entropy -= p * math.log(p)

        # 6. Expected Entropy: E[H(p)] = 1/M sum_m H(p^(m))
        individual_entropies: list[float] = []
        for s in self.seeds:
            h_s = 0.0
            for p in seed_probs[s]:
                if p > 1e-12:
                    h_s -= p * math.log(p)
            individual_entropies.append(h_s)
        exp_entropy = sum(individual_entropies) / n_seeds

        # 7. Mutual Information Proxy: MI = H(p_bar) - E[H(p)]
        mutual_info = max(0.0, pred_entropy - exp_entropy)

        # 8. Abstention check
        abstained = False
        if abstain_threshold is not None:
            abstained = bool(sel_mean_prob < abstain_threshold)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return EnsembleDecisionResult(
            choice=choice_str,
            choice_index=choice_idx,
            probabilities=mean_probs,
            scores=mean_scores,
            options=list(options),
            state=state,
            question=question,
            temperature=self.models[self.seeds[0]].config.calibration_temperature,
            latency_ms=round(latency_ms, 2),
            device=str(self.device),
            model_version="1.0.0",
            abstained=abstained,
            seed_probabilities=seed_probs,
            seed_choices=seed_choices,
            ensemble_mean_probability=sel_mean_prob,
            ensemble_std=sel_std,
            ensemble_min=sel_min,
            ensemble_max=sel_max,
            choice_agreement=choice_agreement,
            predictive_entropy=pred_entropy,
            expected_entropy=exp_entropy,
            mutual_information=mutual_info,
            ensemble_stds=ensemble_stds,
            ensemble_mins=ensemble_mins,
            ensemble_maxs=ensemble_maxs,
            metadata={
                "seeds": self.seeds,
                "execution_mode": self.execution_mode,
                "individual_entropies": individual_entropies,
                "abstain_threshold": abstain_threshold,
            },
        )

    def decide_batch(
        self,
        decisions: Sequence[dict[str, Any]],
        branch_batch_size: int | None = None,
    ) -> list[EnsembleDecisionResult]:
        """
        Executes ensemble decisions across a collection of inputs.
        """
        return [
            self.decide(
                state=d["state"],
                question=d["question"],
                options=d["options"],
                branch_batch_size=branch_batch_size,
            )
            for d in decisions
        ]

    def verify_integrity(self) -> dict[int, bool]:
        """
        Verifies cryptographic SHA-256 hashes of all checkpoint weights in the ensemble.

        Returns:
            Dict mapping seed integer to boolean integrity verification status.
        """
        status: dict[int, bool] = {}
        for s in self.seeds:
            m = self.models[s]
            hashes = FROZEN_SEEDS_HASHES[s]
            if m.lora_path is None or m.head_path is None:
                status[s] = False
                continue
            lora_ok = verify_checkpoint_hash(
                m.lora_path,
                hashes["lora"],
                checkpoint_type=f"Seed {s} LoRA",
                strict=True,
            )
            head_ok = verify_checkpoint_hash(
                m.head_path,
                hashes["head"],
                checkpoint_type=f"Seed {s} Head",
                strict=True,
            )
            status[s] = bool(lora_ok and head_ok)
        return status

    def info(self) -> dict[str, Any]:
        """Returns ensemble metadata and configuration details."""
        return {
            "model_name": "InvariantOneEnsemble",
            "model_version": "1.0.0",
            "seeds": self.seeds,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "execution_mode": self.execution_mode,
            "architecture": "D5-L4 / Nop=16 Multi-Seed Ensemble",
            "calibration_temperature": 1.0091,
        }
