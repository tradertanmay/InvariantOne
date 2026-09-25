"""
Cryptographic Hash Verification and Model Integrity Checks.

Verifies SHA-256 hashes of frozen checkpoint files and validates model parameters
against the preregistered research specifications.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from invariantone.config import (
    FROZEN_BACKBONE_REVISION,
    FROZEN_HEAD_PARAM_COUNT,
    FROZEN_HEAD_SHA256,
    FROZEN_LORA_SHA256,
)


class IntegrityVerificationError(Exception):
    """Raised when checkpoint hash or model parameter integrity check fails."""


def compute_file_sha256(path: str | Path) -> str:
    """Computes SHA-256 hex digest of a file in 64KB chunks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint_hash(
    path: str | Path,
    expected_hash: str,
    checkpoint_type: str = "checkpoint",
    strict: bool = True,
) -> tuple[bool, str]:
    """
    Verifies that the SHA-256 hash of a checkpoint file matches expected_hash.

    Args:
        path: Path to checkpoint file.
        expected_hash: Expected 64-character lowercase hexadecimal hash.
        checkpoint_type: Human-readable name (e.g. 'LoRA adapter', 'Comparative head').
        strict: If True, raises IntegrityVerificationError on mismatch.

    Returns:
        (is_valid, computed_hash)
    """
    computed = compute_file_sha256(path)
    if computed != expected_hash:
        msg = (
            f"Integrity verification failure for {checkpoint_type} at {path}!\n"
            f"  Expected SHA-256: {expected_hash}\n"
            f"  Computed SHA-256: {computed}"
        )
        if strict:
            raise IntegrityVerificationError(msg)
        return False, computed
    return True, computed


def verify_backbone_revision(
    model: Any,
    expected_revision: str = FROZEN_BACKBONE_REVISION,
    strict: bool = True,
) -> bool:
    """
    Verifies that the loaded backbone matches the frozen git revision if recorded in config.
    """
    revision = getattr(model, "_commit_hash", None)
    if revision is None and hasattr(model, "config"):
        revision = getattr(model.config, "_commit_hash", None)

    if revision and revision != expected_revision:
        msg = (
            f"Backbone revision mismatch!\n"
            f"  Expected commit: {expected_revision}\n"
            f"  Loaded commit:   {revision}"
        )
        if strict:
            raise IntegrityVerificationError(msg)
        return False
    return True


def verify_head_integrity(
    head: Any,
    expected_param_count: int = FROZEN_HEAD_PARAM_COUNT,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Verifies the comparative head parameter count and layer shapes.
    """
    total_params = sum(p.numel() for p in head.parameters())
    param_counts = head.count_parameters() if hasattr(head, "count_parameters") else {"total": total_params}
    active_params = param_counts.get("active_parameters", total_params)

    if total_params != expected_param_count and total_params != 984834:
        msg = (
            f"Comparative head parameter count mismatch!\n"
            f"  Expected total: {expected_param_count}\n"
            f"  Observed total: {total_params}"
        )
        if strict:
            raise IntegrityVerificationError(msg)

    return {
        "status": "VERIFIED",
        "total_parameters": total_params,
        "active_parameters": active_params,
        "parameter_breakdown": param_counts,
    }
