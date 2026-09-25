"""
Tests for InvariantOne Model Loading and Integrity Verification.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch

from invariantone.architectures.direct_comparative_head import DirectComparativeHead
from invariantone.config import (
    FROZEN_BACKBONE_REVISION,
    FROZEN_HEAD_PARAM_COUNT,
    FROZEN_HEAD_SHA256,
    FROZEN_LORA_SHA256,
    InvariantOneConfig,
)
from invariantone.loading.checkpoint_loader import resolve_checkpoint_paths
from invariantone.loading.hash_verification import (
    IntegrityVerificationError,
    compute_file_sha256,
    verify_checkpoint_hash,
    verify_head_integrity,
)


def test_checkpoint_path_resolution():
    """Verifies that resolve_checkpoint_paths locates the v1 checkpoints."""
    lora_p, head_p = resolve_checkpoint_paths("checkpoints/invariantone-v1")
    assert lora_p.is_file(), f"LoRA checkpoint not found: {lora_p}"
    assert head_p.is_file(), f"Head checkpoint not found: {head_p}"


def test_checkpoint_hashes_strict():
    """Verifies that frozen checkpoints match the preregistered SHA-256 hashes."""
    lora_p, head_p = resolve_checkpoint_paths("checkpoints/invariantone-v1")

    ok_lora, l_sha = verify_checkpoint_hash(lora_p, FROZEN_LORA_SHA256, "LoRA", strict=True)
    assert ok_lora is True
    assert l_sha == FROZEN_LORA_SHA256

    ok_head, h_sha = verify_checkpoint_hash(head_p, FROZEN_HEAD_SHA256, "Head", strict=True)
    assert ok_head is True
    assert h_sha == FROZEN_HEAD_SHA256


def test_hash_mismatch_raises_in_strict_mode():
    """Verifies that an invalid hash raises IntegrityVerificationError under strict=True."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp:
        tmp.write(b"tampered weight data")
        tmp.flush()

        with pytest.raises(IntegrityVerificationError):
            verify_checkpoint_hash(tmp.name, FROZEN_LORA_SHA256, "LoRA", strict=True)


def test_head_parameter_count():
    """Verifies that DirectComparativeHead has exactly 722,177 trainable parameters."""
    head = DirectComparativeHead(hidden_size=2560, proj_dim=256, mode="absolute_only")
    verification = verify_head_integrity(head, expected_param_count=FROZEN_HEAD_PARAM_COUNT, strict=True)
    assert verification["status"] == "VERIFIED"
    assert verification["total_parameters"] == FROZEN_HEAD_PARAM_COUNT


def test_config_immutability():
    """Verifies that InvariantOneConfig contains frozen research parameters."""
    cfg = InvariantOneConfig()
    assert cfg.model_version == "1.0.0"
    assert cfg.architecture == "D5-L4"
    assert cfg.training_operator_breadth == 16
    assert cfg.backbone_revision == FROZEN_BACKBONE_REVISION
    assert cfg.lora_sha256 == FROZEN_LORA_SHA256
    assert cfg.head_sha256 == FROZEN_HEAD_SHA256
    assert cfg.calibration_temperature == 1.0091
    assert cfg.adapted_layers == [28, 29, 30, 31]
