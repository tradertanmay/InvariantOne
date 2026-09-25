# Model Card: InvariantOne-v1

**Release Version:** `InvariantOne-v1` (Frozen Release)  
**Date:** September 24, 2026  
**License:** Apache-2.0  
**Base Backbone:** `Qwen/Qwen3.5-4B-Base` (Revision `1001bb4d826a52d1f399e183466143f4da7b741b`, Native BF16)  
**Primary Checkpoint:** $D_5$-L4 ($N_{\text{op}}=16$, Seed 42 Primary; Replicated across Seeds 43, 44)  

---

## 1. Executive Summary & Model Overview

`InvariantOne-v1` is an order-equivariant probabilistic decision model designed for multi-candidate decision-making under complex operational constraints and relational rules. 

Standard autoregressive language models suffer from severe position, length, and order biases when evaluating multiple-choice candidate decisions. `InvariantOne-v1` solves this structural failure through a decoupled comparative architecture:
1. **Candidate-Independent Scoring:** Each candidate action is scored independently conditioned on context, eliminating candidate order sensitivity and cross-option attention artifacts.
2. **Analytic Distributional Equivariance:** Option-span pooled representations pass through a lightweight direct comparative projection head, producing logits normalized via softmax to guarantee exact permutation equivariance over candidate distributions ($\text{Mean TVD} = 0.000000$).
3. **Upper-Layer Relational Pre-Adaptation:** Parameter-efficient adaptation (LoRA on layers 28–31) trained across a diverse relational operator universe ($N_{\text{op}}=16$) instills strong, transferable relational reasoning capabilities into the comparative readout.

```
Context + Option_i ---> [Qwen3.5-4B (Frozen Layers 0-27 + LoRA Layers 28-31)]
                                     |
                          Option Span Mean Pooling
                                     |
                        Direct Comparative Head (2560 -> 256 -> 1)
                                     |
                               Logit s_i
                                     |
                           Softmax([s_1, ..., s_K])
```

---

## 2. Model Architecture & Specifications

| Parameter | Specification |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen3.5-4B-Base` (32 transformer layers, $d_{\text{model}} = 2560$) |
| **Backbone Status** | Layers 0–27 strictly frozen (zero parameter updates) |
| **Adaptation** | Low-Rank Adaptation (LoRA) on upper 4 layers (28–31): $r = 8, \alpha = 16$, modules: `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| **Active LoRA Parameters** | 1,179,648 parameters |
| **Readout Head** | `DirectComparativeHead`: Linear($2560 \to 256$) $\to$ GELU $\to$ LayerNorm($256$) $\to$ Linear($256 \to 1$) |
| **Active Head Parameters** | 722,177 parameters |
| **Total Adapted Parameters** | 1,901,825 parameters ($<0.05\%$ of base model) |
| **Pooling Method** | Option-span mean pooling over candidate option token span |
| **Scoring Scheme** | Independent candidate scoring with comparative readout: $P(y = i \mid x, \{o_k\}) = \frac{\exp(s_i / \tau^*)}{\sum_{j=1}^K \exp(s_j / \tau^*)}$ |
| **Calibration Temperature** | $\tau^* = 1.0091$ (pre-calibrated on validation split) |

### Cryptographic Checkpoint Signatures

* **Primary Checkpoint (`Seed 42`, $N_{\text{op}}=16$)**:
  - LoRA Weights: `checkpoints/invariantone-v1/adapter_model.pt` (Frozen Phase 7R research checkpoint)
    - SHA-256: `a0000d3ec984d155fbad5abc854556904ecffe78b997900b677a8d8d5811e4d2`
  - Head Weights: `checkpoints/invariantone-v1/head.pt` (Frozen Phase 7R research checkpoint)
    - SHA-256: `190d867dc302a89d420d19a8374a5db252ea2e71e4f40c68064ded23fc321384`
* **Replication Checkpoint (`Seed 43`, $N_{\text{op}}=16$)**:
  - LoRA: `checkpoints/invariantone-v1/adapter_model_seed43.pt` (Frozen Phase 7R research checkpoint)
  - Head: `checkpoints/invariantone-v1/head_seed43.pt` (Frozen Phase 7R research checkpoint)
* **Replication Checkpoint (`Seed 44`, $N_{\text{op}}=16$)**:
  - LoRA: `checkpoints/invariantone-v1/adapter_model_seed44.pt` (Frozen Phase 7R research checkpoint)
  - Head: `checkpoints/invariantone-v1/head_seed44.pt` (Frozen Phase 7R research checkpoint)

---

## 3. Verified Performance & Benchmark Evaluation

`InvariantOne-v1` was evaluated on `FINAL-HOLDOUT-V2` ($1,536$ four-choice items) under strict preregistered one-shot governance. Prior to inference, SHA-256 checksums of all strata were verified against the locked manifest (`final_holdout_v2_manifest.json`).

> **Scope Note on Candidate Options Count ($K$):**  
> The runtime supports arbitrary $K \ge 2$; the reported scientific evaluation results are for $K=4$ unless otherwise stated. Experimental generalization to arbitrary $K$ has not been independently established in the scientific benchmarks.

### Definitive Benchmark Summary

| Stratum | Evaluation Focus | $D_0$ (Baseline) | $B_1$-Raw | $B_1$-Sym (24-Perm) | `InvariantOne-v1` (Seed 42) | `InvariantOne-v1` (Mean $\pm$ SD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`FINAL-REAL`** | Real operational control tasks ($N=256$) | 89.45% | 91.41% | 92.58% | **94.14%** | **93.36% $\pm$ 1.35%** |
| **`FINAL-L1`** | Familiar-Operator Compositional Recombination ($N=256$) | 32.81% | 38.28% | 43.75% | **89.84%** | **89.32% $\pm$ 0.45%** |
| **`FINAL-L2a`** | Semantic Extrapolation ($N=256$) | 28.12% | 45.31% | 50.39% | **86.33%** | **87.76% $\pm$ 2.48%** |
| **`FINAL-L2b`** | Relational Extrapolation ($N=256$) | 22.27% | 62.50% | **65.23%** | 55.08% | 53.91% $\pm$ 2.03% |
| **`FINAL-L3`** | Double Extrapolation ($N=512$) | 25.20% | 60.94% | **65.43%** | 56.84% | 54.10% $\pm$ 4.91% |
| **`Aggregate`** | Total locked holdout pool ($N=1,536$) | 37.17% | 59.90% | 63.80% | **73.18%** (1,124/1,536) | **71.74% $\pm$ 1.97%** |

### Counterfactual Sensitivity & Minimal-Pair Tracking

| Stratum | Micro Accuracy | Minimal-Pair Accuracy | Correct Override Switch Rate |
| :--- | :---: | :---: | :---: |
| **`FINAL-L1`** | 89.84% | **87.50%** | **99.12%** |
| **`FINAL-L2a`** | 86.33% | **85.16%** | **97.32%** |
| **`FINAL-L2b`** | 55.08% | 31.25% | 39.60% |
| **`FINAL-L3`** | 56.84% | 39.45% | 53.16% |

---

## 4. Invariance Guarantees & Systems Efficiency

### Structural Invariance Audit
* **Distributional Equivariance**: Mean TVD = **$0.000000$**, Max TVD = **$0.000000$** across all 24 option permutations on 128 samples ($3,072$ evaluations).
* **Top-1 Tie-Breaking Instability**: $1.56\%$ top-choice flips under permutation, caused strictly by floating-point precision ties at decision boundaries under argmax selection. The underlying probabilistic output distribution is strictly equivariant.

### Computational Efficiency (Measured on NVIDIA A100-SXM4-40GB)
* **Benchmark Provenance Note:** A100 performance figures are from the frozen FINAL-HOLDOUT-V2 benchmarking run; release-wheel functionality was independently smoke-tested in a fresh Python 3.12 environment.
* **Forward Passes**: 4 candidate forward passes per sample vs 24 full-context forward passes for $B_1$-Sym ($6.0\times$ reduction).
* **Wall-Clock Latency**: **$83.71\text{ ms}$** vs **$425.31\text{ ms}$** ($B_1$-Sym) $\implies$ **$5.08\times$ speedup**.
* **Throughput**: **$11.95\text{ samples/sec}$** vs **$2.35\text{ samples/sec}$** ($B_1$-Sym).
* **Peak VRAM**: **$8,941.2\text{ MB}$** vs **$14,614.2\text{ MB}$** ($B_1$-Sym) $\implies$ **$38.8\%$ memory footprint reduction**.

---

## 5. Capabilities vs Frontier Boundaries

### Confirmed Capabilities
1. **Real-Task Decision Accuracy**: Outperforms unadapted baselines and 24-permutation causal scoring on real operational decision tasks ($94.14\%$).
2. **Compositional Recombination**: Excels at counterfactual state tracking and rule execution for learned operator families ($89.84\%$ accuracy, $99.12\%$ override switch rate).
3. **Semantic Domain Generalization**: Robustly transfers learned relational operators to completely novel technical and engineering domains ($86.33\%$, $+35.94$ points over $B_1$-Sym).
4. **Order-Equivariant Efficiency**: Eliminates position bias with zero prompt permutation overhead, achieving $5.08\times$ lower latency than permutation averaging.

### Localized Frontier Limitations
1. **Zero-Shot Novel Operator Grammars**: Performance drops to $55.08\%$ on $L_{2b}$ and $56.84\%$ on $L_3$. While $D_5$ rescues independent candidate comparative scoring from near-chance baseline failure ($D_0 = 25.20\%$, $\Delta = +31.64$ points), exhaustive permutation-averaged full-context causal inference ($B_1$-Sym $= 65.43\%$) retains an $8.59$-point advantage on completely novel relational operators.
2. **Mechanistic Cause**: Full-context direct-logit inference leverages self-attention across candidate option tokens in-context to perform joint comparison of unfamiliar syntactic operators, whereas the independent comparative head operates on pooled option representations.
3. **Capacity Ceiling ($N_{\text{op}}=16$ vs $N_{\text{op}}=32$)**: Training with 32 operator families saturates held-out transfer ($+1.37\%$ on $L_3$, $p=0.2075$) while causing substantial in-distribution catastrophic degradation on $L_1$ ($75.52\%$ vs $89.32\%$). $N_{\text{op}}=16$ is the frozen Pareto-optimal operating point.

---

## 6. Program Governance & Immutability

* **Status:** FROZEN & ARCHIVED.
* **Audit Trail:** Frozen `FINAL-HOLDOUT-V2` preregistered evaluation archive.
* **Governance Rule:** No further hyperparameter tuning, model re-training, or prompt tweaking shall be conducted on this model release. Any future architectural innovations directed at the novel relational operator frontier are strictly reserved for a new development cycle evaluated against `FINAL-HOLDOUT-V3`.
