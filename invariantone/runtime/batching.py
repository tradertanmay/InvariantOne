"""
Batching and Dynamic-K Execution Utilities for InvariantOne.

Handles candidate branch extraction, chunking, and representation pooling
across arbitrary numbers of candidate options (K >= 2).
"""

from __future__ import annotations

from typing import Any, Sequence
import torch
import torch.nn as nn

from invariantone.formatting import compute_option_spans, pool_option_representations


def encode_candidate_branches(
    model: nn.Module,
    tokenizer: Any,
    state: str,
    question: str,
    options: Sequence[str],
    device: torch.device,
    branch_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Encodes K candidate branches [State, Question, Option_k] through the backbone
    and extracts pooled option representations.

    Args:
        model: Adapted Qwen backbone with upper-layer LoRA.
        tokenizer: Frozen Qwen tokenizer.
        state: State description string.
        question: Decision question string.
        options: List of K option strings (K >= 2).
        device: Execution device.
        branch_batch_size: Optional chunk size for large K. If None, encodes all K options simultaneously.

    Returns:
        reps: Tensor of shape [K, hidden_size]
    """
    k_opts = len(options)
    if k_opts < 2:
        raise ValueError(f"InvariantOne requires at least 2 options, got {k_opts}")

    if branch_batch_size is None or branch_batch_size >= k_opts:
        # Single forward pass for all K options
        span_info = compute_option_spans(tokenizer, state, question, options, device=device)
        inputs = span_info["inputs"]
        p_len = span_info["prefix_len"]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            top_hidden = outputs.hidden_states[-1]
            reps = pool_option_representations(top_hidden, inputs["attention_mask"], p_len)
        return reps

    # Chunked forward passes for large K
    reps_list: list[torch.Tensor] = []
    for start_idx in range(0, k_opts, branch_batch_size):
        chunk_options = options[start_idx : start_idx + branch_batch_size]
        span_info = compute_option_spans(tokenizer, state, question, chunk_options, device=device)
        inputs = span_info["inputs"]
        p_len = span_info["prefix_len"]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            top_hidden = outputs.hidden_states[-1]
            chunk_reps = pool_option_representations(top_hidden, inputs["attention_mask"], p_len)
            reps_list.append(chunk_reps)

    return torch.cat(reps_list, dim=0)
