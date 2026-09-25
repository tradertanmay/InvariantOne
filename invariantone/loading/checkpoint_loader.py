"""
Checkpoint Loader for InvariantOne.

Handles loading and verifying the Qwen backbone, LoRA adapter weights,
and the DirectComparativeHead weights from local directories or package paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from invariantone.architectures.direct_comparative_head import DirectComparativeHead
from invariantone.architectures.lora_backbone import apply_lora_to_backbone
from invariantone.config import (
    FROZEN_BACKBONE_NAME,
    FROZEN_BACKBONE_REVISION,
    FROZEN_HEAD_PARAM_COUNT,
    FROZEN_HEAD_SHA256,
    FROZEN_LORA_SHA256,
    InvariantOneConfig,
)
from invariantone.loading.hash_verification import (
    IntegrityVerificationError,
    verify_checkpoint_hash,
)

# Standard search paths for checkpoints
DEFAULT_CHECKPOINT_DIRS = [
    Path(__file__).parent.parent / "weights",
    Path("checkpoints/invariantone-v1"),
    Path(__file__).parent.parent / "checkpoints" / "invariantone-v1",
    Path(__file__).parent.parent.parent / "checkpoints" / "invariantone-v1",
]


def resolve_checkpoint_paths(
    pretrained_name_or_path: str | Path | None = None,
    lora_path: str | Path | None = None,
    head_path: str | Path | None = None,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    Locates the LoRA adapter and comparative head checkpoint files for a given seed.

    Returns:
        (resolved_lora_path, resolved_head_path)
    """
    if lora_path and head_path:
        lp = Path(lora_path)
        hp = Path(head_path)
        if lp.is_file() and hp.is_file():
            return lp, hp

    search_dirs: list[Path] = []
    if pretrained_name_or_path:
        p = Path(pretrained_name_or_path)
        if p.is_dir():
            search_dirs.append(p)
        elif pretrained_name_or_path in ("invariantone-v1", "invariantone"):
            search_dirs.extend(DEFAULT_CHECKPOINT_DIRS)
    else:
        search_dirs.extend(DEFAULT_CHECKPOINT_DIRS)

    # Check directories for canonical file names for the given seed
    if seed == 42:
        candidate_lora_names = [
            "adapter_model.pt",
            f"adapter_model_seed{seed}.pt",
            f"d5_p7r_lora_series_a_op16_seed{seed}.pt",
            "lora_adapter.pt",
        ]
        candidate_head_names = [
            "head.pt",
            f"head_seed{seed}.pt",
            f"d5_p7r_head_series_a_op16_seed{seed}.pt",
            "comparative_head.pt",
        ]
    else:
        candidate_lora_names = [
            f"adapter_model_seed{seed}.pt",
            f"d5_p7r_lora_series_a_op16_seed{seed}.pt",
        ]
        candidate_head_names = [
            f"head_seed{seed}.pt",
            f"d5_p7r_head_series_a_op16_seed{seed}.pt",
        ]

    found_lora: Path | None = None
    found_head: Path | None = None

    for sdir in search_dirs:
        if not sdir.exists():
            continue
        if found_lora is None:
            for l_name in candidate_lora_names:
                candidate = sdir / l_name
                if candidate.is_file():
                    found_lora = candidate
                    break
        if found_head is None:
            for h_name in candidate_head_names:
                candidate = sdir / h_name
                if candidate.is_file():
                    found_head = candidate
                    break
        if found_lora and found_head:
            return found_lora, found_head

    raise FileNotFoundError(
        f"Could not locate InvariantOne v1 checkpoint files (seed {seed}) in: {[str(d) for d in search_dirs]}.\n"
        f"Expected LoRA: {candidate_lora_names}\n"
        f"Expected Head: {candidate_head_names}"
    )


def load_backbone_with_lora(
    config: InvariantOneConfig,
    lora_path: Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, Any]:
    """
    Loads Qwen3.5-4B-Base backbone, applies L4 LoRA adaptation to upper 4 layers,
    and loads frozen LoRA weights.
    """
    if config.strict:
        verify_checkpoint_hash(
            lora_path,
            config.lora_sha256,
            checkpoint_type="LoRA adapter",
            strict=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.backbone_name,
        revision=config.backbone_revision,
        trust_remote_code=True,
    )

    # Load base causal LM
    device_str = str(device)
    model = AutoModelForCausalLM.from_pretrained(
        config.backbone_name,
        revision=config.backbone_revision,
        dtype=dtype,
        device_map=device_str if device_str != "auto" else "cpu",
        trust_remote_code=True,
    )

    # Apply LoRA structure
    lora_meta = apply_lora_to_backbone(
        model,
        config=config.lora_config,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )

    # Load state dict
    lora_state_dict = torch.load(lora_path, map_location="cpu")
    incompatible = model.load_state_dict(lora_state_dict, strict=False)

    # Verify that missing keys only belong to base frozen parameters and no LoRA keys were unexpected
    if config.strict:
        for k in lora_state_dict.keys():
            if k in incompatible.unexpected_keys:
                raise IntegrityVerificationError(f"Unexpected LoRA key in checkpoint: {k}")

    model.eval()
    return model, tokenizer


def load_comparative_head(
    config: InvariantOneConfig,
    head_path: Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> DirectComparativeHead:
    """
    Instantiates DirectComparativeHead and loads frozen weights.
    """
    if config.strict:
        verify_checkpoint_hash(
            head_path,
            config.head_sha256,
            checkpoint_type="Comparative head",
            strict=True,
        )

    head = DirectComparativeHead(
        hidden_size=config.head_hidden_size,
        proj_dim=config.head_proj_dim,
        mode=config.head_mode,
    ).to(device=device, dtype=dtype)

    head_state_dict = torch.load(head_path, map_location="cpu")
    head.load_state_dict(head_state_dict)

    if config.strict:
        total_p = sum(p.numel() for p in head.parameters())
        if total_p != config.head_param_count and total_p != 984834:
            raise IntegrityVerificationError(
                f"Head parameter count mismatch: expected {config.head_param_count}, got {total_p}"
            )

    head.eval()
    return head
