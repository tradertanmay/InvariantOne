"""
Configuration specifications for InvariantOne v1.

Defines the InvariantOneConfig dataclass and Pydantic models with frozen hashes,
architecture hyperparameters, and runtime settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field

# Frozen Research Hashes and Parameters (Milestone: InvariantOne v1)
FROZEN_BACKBONE_NAME = "Qwen/Qwen3.5-4B-Base"
FROZEN_BACKBONE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
FROZEN_LORA_SHA256 = "a0000d3ec984d155fbad5abc854556904ecffe78b997900b677a8d8d5811e4d2"
FROZEN_HEAD_SHA256 = "190d867dc302a89d420d19a8374a5db252ea2e71e4f40c68064ded23fc321384"
FROZEN_CALIBRATION_TEMPERATURE = 1.0091
FROZEN_ADAPTED_LAYERS = [28, 29, 30, 31]
FROZEN_LORA_RANK = 8
FROZEN_LORA_ALPHA = 16
FROZEN_HEAD_HIDDEN_SIZE = 2560
FROZEN_HEAD_PROJ_DIM = 256
FROZEN_HEAD_MODE = "absolute_only"
FROZEN_HEAD_PARAM_COUNT = 984834
FROZEN_HEAD_ACTIVE_PARAM_COUNT = 722177


FROZEN_SEEDS_HASHES: dict[int, dict[str, str]] = {
    42: {
        "lora": "a0000d3ec984d155fbad5abc854556904ecffe78b997900b677a8d8d5811e4d2",
        "head": "190d867dc302a89d420d19a8374a5db252ea2e71e4f40c68064ded23fc321384",
    },
    43: {
        "lora": "3c0907e6b29cb0efaea891600b4fe7ac2520f0788308ef66dba412993c2f3373",
        "head": "472abd0353d7ddf5669fe1427a29b0d491802fe954f57474f5e6824fd25894fd",
    },
    44: {
        "lora": "023da5216ef707d422fc391b99042d65ed49fdbd01890abb0a66b7f18579f7d0",
        "head": "f52f0c4703ed050a37bf4b73b2600ad7599d31020caa42e7a5bb3db1fda9e8b5",
    },
}


class InvariantOneConfig(BaseModel):
    """
    Frozen production configuration for InvariantOne-v1.
    All architectural constants match the validated D5-L4 / Nop=16 checkpoint.
    """

    model_name: str = Field(default="InvariantOne", description="Public model identifier")
    model_version: str = Field(default="1.0.0", description="SemVer release version")
    research_designation: str = Field(default="D5-L4 / Nop=16 / Seed 42", description="Research lineage identifier")
    architecture: str = Field(default="D5-L4", description="Architectural configuration")
    training_operator_breadth: int = Field(default=16, description="Training relational operator breadth (N_op)")
    primary_seed: int = Field(default=42, description="Primary validation seed")

    @classmethod
    def for_seed(cls, seed: int = 42, **kwargs: Any) -> InvariantOneConfig:
        """Constructs an InvariantOneConfig initialized with verified hashes for a specific seed."""
        if seed not in FROZEN_SEEDS_HASHES:
            raise ValueError(f"Unknown seed {seed}. Supported seeds: {list(FROZEN_SEEDS_HASHES.keys())}")
        hashes = FROZEN_SEEDS_HASHES[seed]
        return cls(
            primary_seed=seed,
            research_designation=f"D5-L4 / Nop=16 / Seed {seed}",
            lora_sha256=hashes["lora"],
            head_sha256=hashes["head"],
            **kwargs,
        )

    # Backbone parameters
    backbone_name: str = Field(default=FROZEN_BACKBONE_NAME, description="Hugging Face base model identifier")
    backbone_revision: str = Field(default=FROZEN_BACKBONE_REVISION, description="Git commit SHA of frozen backbone")

    # LoRA parameters
    lora_config: Literal["L4", "L8"] = Field(default="L4", description="LoRA layer selection configuration")
    lora_r: int = Field(default=FROZEN_LORA_RANK, description="LoRA rank")
    lora_alpha: int = Field(default=FROZEN_LORA_ALPHA, description="LoRA alpha scaling factor")
    lora_dropout: float = Field(default=0.0, description="LoRA dropout during inference")
    adapted_layers: list[int] = Field(default_factory=lambda: list(FROZEN_ADAPTED_LAYERS), description="Adapted layer indices")
    lora_sha256: str = Field(default=FROZEN_LORA_SHA256, description="Expected SHA-256 of LoRA adapter checkpoint")

    # Comparative Head parameters
    head_hidden_size: int = Field(default=FROZEN_HEAD_HIDDEN_SIZE, description="Hidden dimension from backbone")
    head_proj_dim: int = Field(default=FROZEN_HEAD_PROJ_DIM, description="Projection bottleneck dimension")
    head_mode: Literal["absolute_only", "pairwise_only", "combined"] = Field(
        default=FROZEN_HEAD_MODE, description="Head scoring mode"
    )
    head_sha256: str = Field(default=FROZEN_HEAD_SHA256, description="Expected SHA-256 of comparative head checkpoint")
    head_param_count: int = Field(default=FROZEN_HEAD_PARAM_COUNT, description="Expected total head parameter count")
    head_active_param_count: int = Field(default=FROZEN_HEAD_ACTIVE_PARAM_COUNT, description="Expected active head parameter count")

    # Inference & calibration
    calibration_temperature: float = Field(default=FROZEN_CALIBRATION_TEMPERATURE, description="Frozen calibration temperature tau*")
    pooling_strategy: str = Field(default="option_span_mean", description="Option representation pooling strategy")
    dtype: str = Field(default="bfloat16", description="Target inference dtype")
    device: str = Field(default="auto", description="Execution device (cuda, mps, cpu, auto)")
    strict: bool = Field(default=True, description="Enforce strict cryptographic hash and revision verification")

    # Local checkpoint paths (optional overrides)
    checkpoint_dir: Path | None = Field(default=None, description="Custom local checkpoint directory")
    lora_checkpoint_path: Path | None = Field(default=None, description="Explicit path to LoRA checkpoint")
    head_checkpoint_path: Path | None = Field(default=None, description="Explicit path to comparative head checkpoint")

    model_config = {"extra": "ignore"}
