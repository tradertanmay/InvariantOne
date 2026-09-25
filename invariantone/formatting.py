"""
Canonical Prompt Formatting and Span Logic for InvariantOne.

This module centralizes prompt construction and option-span boundary extraction,
preserving byte-for-byte and token-for-token equivalence with the frozen research
experiments (Phase 4 through FINAL-HOLDOUT-V2).
"""

from __future__ import annotations

from typing import Any, Sequence
import torch


def format_prefix(state: str, question: str) -> str:
    """
    Constructs the canonical prefix string for state and question.

    Format:
        State:\\n{state.strip()}\\n\\nQuestion:\\n{question.strip()}\\n\\nCandidate Choice:\\n
    """
    return f"State:\n{state.strip()}\n\nQuestion:\n{question.strip()}\n\nCandidate Choice:\n"


def format_candidate_prompt(state: str, question: str, option: str) -> str:
    """
    Constructs the canonical full prompt for a single candidate option branch.

    Format:
        {format_prefix(state, question)}{option.strip()}
    """
    return format_prefix(state, question) + option.strip()


def format_candidate_prompts(state: str, question: str, options: Sequence[str]) -> list[str]:
    """
    Constructs canonical prompts for all candidate options.
    """
    prefix = format_prefix(state, question)
    return [prefix + opt.strip() for opt in options]


def compute_option_spans(
    tokenizer: Any,
    state: str,
    question: str,
    options: Sequence[str],
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """
    Tokenizes canonical prompts and identifies the exact [start, end) token span
    for each option in the candidate branch.

    Returns:
        dict containing:
            - 'tokenized_inputs': BatchEncoding suitable for model(**tokenized_inputs)
            - 'prefix_len': token length of the shared prefix
            - 'spans': list of (start_idx, end_idx) for each candidate option
    """
    prefix = format_prefix(state, question)
    prefix_tokens = tokenizer(prefix, return_tensors="pt")["input_ids"]
    p_len = int(prefix_tokens.shape[1])

    prompts = [prefix + opt.strip() for opt in options]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    if device is not None:
        inputs = inputs.to(device)

    spans = []
    attn_mask = inputs["attention_mask"]
    for k in range(len(options)):
        s_len = int(attn_mask[k].sum().item())
        start = p_len
        end = s_len
        spans.append((start, end))

    return {
        "inputs": inputs,
        "prefix_len": p_len,
        "spans": spans,
    }


def pool_option_representations(
    top_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    """
    Computes option representations from the top hidden states using the exact
    option_span_mean rule with single-token fallback.

    Args:
        top_hidden_states: Tensor of shape [K, SeqLen, HiddenSize]
        attention_mask: Tensor of shape [K, SeqLen]
        prefix_len: integer token length of prefix

    Returns:
        reps: Tensor of shape [K, HiddenSize]
    """
    k_opts = top_hidden_states.shape[0]
    reps = []
    for k in range(k_opts):
        s_len = int(attention_mask[k].sum().item())
        if prefix_len < s_len:
            # option_span_mean over the tokens corresponding to Candidate Choice
            h_k = top_hidden_states[k, prefix_len:s_len, :].mean(dim=0)
        else:
            # Fallback to final non-padding token
            h_k = top_hidden_states[k, s_len - 1, :]
        reps.append(h_k)
    return torch.stack(reps, dim=0)
