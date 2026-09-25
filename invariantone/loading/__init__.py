"""
Loading and verification package for InvariantOne.
"""

from invariantone.loading.checkpoint_loader import (
    load_backbone_with_lora,
    load_comparative_head,
    resolve_checkpoint_paths,
)
from invariantone.loading.hash_verification import (
    IntegrityVerificationError,
    compute_file_sha256,
    verify_backbone_revision,
    verify_checkpoint_hash,
    verify_head_integrity,
)

__all__ = [
    "IntegrityVerificationError",
    "compute_file_sha256",
    "load_backbone_with_lora",
    "load_comparative_head",
    "resolve_checkpoint_paths",
    "verify_backbone_revision",
    "verify_checkpoint_hash",
    "verify_head_integrity",
]
