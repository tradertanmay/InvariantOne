"""
Hardware Device and Precision Resolution for InvariantOne.
"""

from __future__ import annotations

import torch


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """
    Resolves requested device string or auto-detects optimal hardware.
    Priority order for 'auto': CUDA -> MPS -> CPU.
    """
    if isinstance(requested, torch.device):
        return requested

    if requested is None or requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Default to CPU on macOS / other systems to avoid upstream PyTorch MPS Metal shader crashes
        return torch.device("cpu")

    req_lower = requested.lower()
    if req_lower.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({requested}) but CUDA is not available!")
        return torch.device(requested)
    elif req_lower.startswith("mps"):
        if not torch.backends.mps.is_available():
            raise RuntimeError(f"Apple MPS device requested ({requested}) but MPS is not available!")
        return torch.device("mps")
    elif req_lower.startswith("cpu"):
        return torch.device("cpu")
    else:
        return torch.device(requested)


def resolve_dtype(
    requested: str | torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.dtype:
    """
    Resolves tensor dtype. Default for research reproduction is bfloat16.
    """
    if isinstance(requested, torch.dtype):
        return requested

    if requested is None or requested in ("bf16", "bfloat16"):
        return torch.bfloat16
    elif requested in ("fp16", "float16", "half"):
        return torch.float16
    elif requested in ("fp32", "float32", "float"):
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype requested: {requested}")
