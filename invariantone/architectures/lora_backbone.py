"""
LoRA Backbone Adaptation Module for InvariantOne Phase 4 (D5).

Supports limited parameter adaptation of upper Qwen3.5 layers while strictly
preserving independent option scoring, exact permutation equivariance,
and mathematical Independence of Irrelevant Alternatives (IIA).

LoRA Configurations:
- L4: Layers 28..31 adapted (585,728 trainable parameters, 0.015% of backbone)
- L8: Layers 24..31 adapted (1,171,456 trainable parameters, 0.029% of backbone)

Hybrid Layer Targeting:
- self_attn layers (every 4th layer: 27, 31): q_proj, v_proj, o_proj
- linear_attn layers (intermediate Gated Delta Net: 24, 25, 26, 28, 29, 30): in_proj_qkv, out_proj
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from invariantone.architectures.direct_comparative import DirectComparativeHead

LoRAConfig = Literal["L4", "L8"]


class LoRALinear(nn.Module):
    """
    Parameter-efficient Low-Rank Adaptation (LoRA) wrapper around an existing nn.Linear.
    Computes: y = W_base(x) + (alpha / r) * (x @ A.T @ B.T).
    Base weight is completely frozen.
    B is initialized to zeros, so Delta W = 0 at initialization.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Freeze base linear parameters
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable adapter weights
        self.lora_A = nn.Parameter(
            torch.empty(r, self.in_features, dtype=base_layer.weight.dtype, device=base_layer.weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.empty(self.out_features, r, dtype=base_layer.weight.dtype, device=base_layer.weight.device)
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming uniform initialization for A, exact zeros for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass (no grads into base_layer)
        base_out = self.base_layer(x)
        # LoRA forward pass: (x @ A^T) @ B^T * scaling
        lora_in = self.dropout(x)
        lora_out = F.linear(F.linear(lora_in, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def get_adapted_layer_indices(config: LoRAConfig) -> list[int]:
    """Returns the layer indices to adapt based on configuration."""
    if config == "L4":
        return [28, 29, 30, 31]
    elif config == "L8":
        return list(range(24, 32))
    else:
        raise ValueError(f"Unknown LoRA config: {config}. Supported: 'L4', 'L8'")


def apply_lora_to_backbone(
    model: nn.Module,
    config: LoRAConfig = "L4",
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> dict[str, Any]:
    """
    Applies LoRA adapters to upper layers of Qwen3.5 backbone.
    Freezes all base parameters and tracks parameter counts.
    """
    target_layers = set(get_adapted_layer_indices(config))

    # 1. Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    num_self_attn_adapted = 0
    num_linear_attn_adapted = 0
    adapted_modules: list[str] = []

    # 2. Iterate through layers and attach LoRA
    # In Qwen: model.model.layers is the ModuleList
    layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else model.layers

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in target_layers:
            continue

        # Check for self-attention
        if hasattr(layer, "self_attn"):
            num_self_attn_adapted += 1
            for name in ["q_proj", "v_proj", "o_proj"]:
                if hasattr(layer.self_attn, name):
                    orig = getattr(layer.self_attn, name)
                    if isinstance(orig, nn.Linear):
                        lora_mod = LoRALinear(orig, r=r, alpha=alpha, dropout=dropout)
                        setattr(layer.self_attn, name, lora_mod)
                        adapted_modules.append(f"layers.{layer_idx}.self_attn.{name}")

        # Check for Gated Delta Net linear attention
        if hasattr(layer, "linear_attn"):
            num_linear_attn_adapted += 1
            for name in ["in_proj_qkv", "out_proj"]:
                if hasattr(layer.linear_attn, name):
                    orig = getattr(layer.linear_attn, name)
                    if isinstance(orig, nn.Linear):
                        lora_mod = LoRALinear(orig, r=r, alpha=alpha, dropout=dropout)
                        setattr(layer.linear_attn, name, lora_mod)
                        adapted_modules.append(f"layers.{layer_idx}.linear_attn.{name}")

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    stats = {
        "config": config,
        "target_layers": sorted(list(target_layers)),
        "num_self_attn_adapted": num_self_attn_adapted,
        "num_linear_attn_adapted": num_linear_attn_adapted,
        "adapted_modules_count": len(adapted_modules),
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_fraction": trainable_params / (trainable_params + frozen_params) if (trainable_params + frozen_params) > 0 else 0.0,
    }
    return stats


class D5System(nn.Module):
    """
    InvariantOne Phase 4 (D5) Architecture.
    Combines:
    1. Qwen3.5 backbone with upper-layer LoRA (L4 or L8)
    2. Option-span mean pooling: h_i = mean(H_i[p_len:])
    3. Parameter-shared DirectComparativeHead (mode='absolute_only')
    
    Guarantees:
    - Zero cross-option interaction: s_i = g(h_i) depends strictly on option i.
    - Exact Permutation Equivariance: s(pi(O)) = pi(s(O)).
    - Exact Candidate-Insertion Invariance: s_i(O union {d}) = s_i(O).
    - Exact IIA & Conditional-Original-Option TVD = 0.000000.
    """

    def __init__(
        self,
        backbone: nn.Module,
        lora_config: LoRAConfig = "L4",
        hidden_size: int = 2560,
        r: int = 8,
        alpha: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.lora_config = lora_config
        self.hidden_size = hidden_size
        self.r = r
        self.alpha = alpha

        # Apply LoRA to upper backbone layers
        self.lora_stats = apply_lora_to_backbone(backbone, config=lora_config, r=r, alpha=alpha)

        # Scorer: standard D0 absolute_only head
        self.head = DirectComparativeHead(
            hidden_size=hidden_size,
            proj_dim=256,
            mode="absolute_only",
        )

        # Temperature parameter (calibrated post-hoc)
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def set_temperature(self, temp: float) -> None:
        self.temperature.copy_(torch.tensor(temp, dtype=torch.float32))

    def extract_option_representation(
        self,
        input_ids: torch.Tensor,
        prefix_len: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Runs a single branch [S, Q, O_i] through the backbone,
        and mean-pools over the option token span.
        
        Args:
            input_ids: [1, seq_len] or [seq_len]
            prefix_len: Number of tokens in [S, Q] prefix
            attention_mask: Optional mask
        Returns:
            h_i: [hidden_size] option representation
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Top hidden states: [1, seq_len, hidden_size]
        last_hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None else outputs.last_hidden_state

        # Mean pool strictly over option span [prefix_len:]
        option_span = last_hidden[0, prefix_len:, :]
        if option_span.size(0) == 0:
            # Fallback to last token if span is empty
            h_i = last_hidden[0, -1, :]
        else:
            h_i = option_span.mean(dim=0)
        return h_i

    def score_options_from_representations(self, h_options: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_options: [K, hidden_size]
        Returns:
            scores: [K] raw decision scores
        """
        return self.head(h_options)

    def forward_probabilities(self, h_options: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_options: [K, hidden_size]
        Returns:
            probs: [K] calibrated choice probabilities
        """
        scores = self.score_options_from_representations(h_options)
        return F.softmax(scores / self.temperature, dim=-1)
