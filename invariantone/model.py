"""
InvariantOne: Production Decision Model.

Packages the frozen D5-L4 (N_op=16, Seed 42) research checkpoint into a clean,
production-ready Python interface.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from invariantone.architectures.direct_comparative_head import DirectComparativeHead
from invariantone.calibration import calibrated_softmax
from invariantone.config import InvariantOneConfig
from invariantone.loading.checkpoint_loader import (
    load_backbone_with_lora,
    load_comparative_head,
    resolve_checkpoint_paths,
)
from invariantone.loading.hash_verification import (
    IntegrityVerificationError,
    verify_checkpoint_hash,
)
from invariantone.result import BatchDecisionResult, DecisionResult
from invariantone.runtime.batching import encode_candidate_branches
from invariantone.runtime.device import resolve_device, resolve_dtype


class InvariantOne:
    """
    InvariantOne Decision Model (v1.0.0).

    A natural-language decision model that scores runtime-defined options directly
    rather than generating text token-by-token. Built on Qwen3.5-4B-Base with
    upper-layer LoRA adaptation and a permutation-equivariant Direct Comparative Head.
    """

    def __init__(
        self,
        config: InvariantOneConfig,
        model: nn.Module,
        head: DirectComparativeHead,
        tokenizer: Any,
        device: torch.device,
        dtype: torch.dtype,
        lora_path: Path | None = None,
        head_path: Path | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.head = head
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.lora_path = lora_path
        self.head_path = head_path

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "invariantone-v1",
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "bfloat16",
        strict: bool = True,
        seed: int = 42,
        checkpoint_dir: str | Path | None = None,
        lora_checkpoint_path: str | Path | None = None,
        head_checkpoint_path: str | Path | None = None,
    ) -> InvariantOne:
        """
        Loads InvariantOne v1 from local checkpoints or model identifier.

        Args:
            pretrained_model_name_or_path: Checkpoint directory path or model identifier ('invariantone-v1').
            device: Target execution device ('auto', 'cuda', 'mps', 'cpu').
            dtype: Target precision dtype ('bfloat16', 'float16', 'float32').
            strict: If True, enforces strict SHA-256 and parameter checks.
            seed: Checkpoint replication seed (42, 43, or 44). Defaults to primary Seed 42.
            checkpoint_dir: Explicit override for checkpoint directory.
            lora_checkpoint_path: Explicit override for LoRA weights path.
            head_checkpoint_path: Explicit override for comparative head path.

        Returns:
            An instantiated InvariantOne decision model ready for inference.
        """
        config = InvariantOneConfig.for_seed(seed=seed, strict=strict)

        # Resolve paths
        resolved_lora, resolved_head = resolve_checkpoint_paths(
            pretrained_name_or_path=checkpoint_dir or pretrained_model_name_or_path,
            lora_path=lora_checkpoint_path,
            head_path=head_checkpoint_path,
            seed=seed,
        )

        resolved_dev = resolve_device(device)
        resolved_dt = resolve_dtype(dtype, resolved_dev)

        # Load backbone + LoRA
        model, tokenizer = load_backbone_with_lora(
            config=config,
            lora_path=resolved_lora,
            device=resolved_dev,
            dtype=resolved_dt,
        )

        # Load Comparative Head
        head = load_comparative_head(
            config=config,
            head_path=resolved_head,
            device=resolved_dev,
            dtype=resolved_dt,
        )

        return cls(
            config=config,
            model=model,
            head=head,
            tokenizer=tokenizer,
            device=resolved_dev,
            dtype=resolved_dt,
            lora_path=resolved_lora,
            head_path=resolved_head,
        )

    @classmethod
    def load_assurance_mode(
        cls,
        pretrained_model_name_or_path: str | Path = "invariantone-v1",
        seeds: Sequence[int] = (42, 43, 44),
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "bfloat16",
        strict: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Loads InvariantOne in High-Assurance Mode (Seeds 42, 43, 44 multi-seed ensemble
        with comprehensive uncertainty diagnostics).

        Args:
            pretrained_model_name_or_path: Checkpoint directory path or model identifier.
            seeds: Tuple or list of replication seeds to include in ensemble.
            device: Target execution device.
            dtype: Target execution precision dtype.
            strict: If True, enforces strict SHA-256 checks across all checkpoints.
            **kwargs: Additional parameters forwarded to InvariantOneEnsemble.from_pretrained.

        Returns:
            An instantiated InvariantOneEnsemble.
        """
        from invariantone.ensemble import InvariantOneEnsemble
        return InvariantOneEnsemble.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            seeds=seeds,
            device=device,
            dtype=dtype,
            strict=strict,
            **kwargs,
        )

    def _validate_inputs(self, state: str, question: str, options: Sequence[str]) -> None:
        """Validates input arguments according to specification."""
        if not isinstance(state, str) or not state.strip():
            raise ValueError("state must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(options, (list, tuple)) or len(options) < 2:
            raise ValueError(f"options must be a sequence of at least 2 candidate strings, got {len(options) if isinstance(options, (list, tuple)) else type(options)}")
        for i, opt in enumerate(options):
            if not isinstance(opt, str) or not opt.strip():
                raise ValueError(f"Option at index {i} must be a non-empty string")

    def score(
        self,
        state: str,
        question: str,
        options: Sequence[str],
        branch_batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Computes uncalibrated scalar scores for each candidate option.

        Args:
            state: State description.
            question: Decision question.
            options: List of candidate options (K >= 2).
            branch_batch_size: Optional chunking parameter.

        Returns:
            scores: Tensor of shape [K] containing raw comparative logits.
        """
        self._validate_inputs(state, question, options)

        reps = encode_candidate_branches(
            model=self.model,
            tokenizer=self.tokenizer,
            state=state,
            question=question,
            options=options,
            device=self.device,
            branch_batch_size=branch_batch_size,
        )

        with torch.no_grad():
            scores = self.head(reps)
        return scores

    def decide(
        self,
        state: str,
        question: str,
        options: Sequence[str],
        branch_batch_size: int | None = None,
        abstain_threshold: float | None = None,
    ) -> DecisionResult:
        """
        Executes a decision over runtime-provided options.

        Args:
            state: State description string.
            question: Decision question string.
            options: Sequence of candidate option strings (K >= 2).
            branch_batch_size: Optional batch size for branch evaluation.
            abstain_threshold: Optional minimum confidence threshold. If set and
                max probability < abstain_threshold, result.abstained is set to True.

        Returns:
            DecisionResult with selected choice, calibrated probabilities, and raw scores.
        """
        t0 = time.perf_counter()

        scores = self.score(
            state=state,
            question=question,
            options=options,
            branch_batch_size=branch_batch_size,
        )

        tau = self.config.calibration_temperature
        probs_tensor = calibrated_softmax(scores, tau=tau)

        scores_list = [float(s) for s in scores.cpu().tolist()]
        probs_list = [float(p) for p in probs_tensor.cpu().tolist()]

        choice_idx = int(torch.argmax(scores).item())
        choice_str = options[choice_idx]

        confidence = max(probs_list)
        abstained = False
        if abstain_threshold is not None:
            abstained = bool(confidence < abstain_threshold)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DecisionResult(
            choice=choice_str,
            choice_index=choice_idx,
            probabilities=probs_list,
            scores=scores_list,
            options=list(options),
            state=state,
            question=question,
            temperature=tau,
            latency_ms=round(latency_ms, 2),
            device=str(self.device),
            model_version=self.config.model_version,
            abstained=abstained,
            metadata={
                "architecture": self.config.architecture,
                "training_operator_breadth": self.config.training_operator_breadth,
                "seed": self.config.primary_seed,
                "abstain_threshold": abstain_threshold,
            },
        )

    def decide_batch(
        self,
        decisions: Sequence[dict[str, Any]],
        branch_batch_size: int | None = None,
    ) -> BatchDecisionResult:
        """
        Executes decisions across multiple distinct inputs.
        Preserves individual per-request option identities and dynamic K.
        """
        t0 = time.perf_counter()
        results: list[DecisionResult] = []

        for d in decisions:
            res = self.decide(
                state=d["state"],
                question=d["question"],
                options=d["options"],
                branch_batch_size=branch_batch_size,
            )
            results.append(res)

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        return BatchDecisionResult(
            results=results,
            total_latency_ms=round(total_latency_ms, 2),
            batch_size=len(decisions),
        )

    def verify_integrity(self) -> dict[str, Any]:
        """
        Performs an audit of loaded weights, parameter counts, and commit hashes.
        """
        lora_ok = False
        head_ok = False

        if self.lora_path and self.lora_path.is_file():
            lora_ok, _ = verify_checkpoint_hash(
                self.lora_path,
                self.config.lora_sha256,
                checkpoint_type="LoRA adapter",
                strict=False,
            )

        if self.head_path and self.head_path.is_file():
            head_ok, _ = verify_checkpoint_hash(
                self.head_path,
                self.config.head_sha256,
                checkpoint_type="Comparative head",
                strict=False,
            )

        head_params = sum(p.numel() for p in self.head.parameters())
        head_params_ok = head_params == self.config.head_param_count

        all_ok = lora_ok and head_ok and head_params_ok

        return {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "architecture": self.config.architecture,
            "backbone_name": self.config.backbone_name,
            "backbone_revision": self.config.backbone_revision,
            "lora_sha256_verified": lora_ok,
            "head_sha256_verified": head_ok,
            "head_parameter_count_verified": head_params_ok,
            "head_parameter_count": head_params,
            "calibration_temperature": self.config.calibration_temperature,
            "all_checks_passed": all_ok,
        }

    def info(self) -> dict[str, Any]:
        """
        Returns stable public metadata describing the model.
        """
        return {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "research_designation": self.config.research_designation,
            "backbone": self.config.backbone_name,
            "backbone_revision": self.config.backbone_revision,
            "architecture": self.config.architecture,
            "lora_rank": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "adapted_layers": self.config.adapted_layers,
            "training_operator_breadth": self.config.training_operator_breadth,
            "calibration_temperature": self.config.calibration_temperature,
            "weights_hashes": {
                "lora_sha256": self.config.lora_sha256,
                "head_sha256": self.config.head_sha256,
            },
        }
