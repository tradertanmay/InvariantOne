# InvariantOne

**InvariantOne is a natural-language decision model that scores runtime-defined options directly rather than generating an answer token-by-token.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/tradertanmay/InvariantOne/blob/main/LICENSE)
[![Model: D5-L4](https://img.shields.io/badge/Architecture-D5--L4%20%7C%20Nop%3D16-purple.svg)](https://huggingface.co/TanmaySah/InvariantOne-v1)
[![Equivariance: Exact](https://img.shields.io/badge/Equivariance-Exact%20by%20Construction-brightgreen.svg)](https://huggingface.co/TanmaySah/InvariantOne-v1)

---

## Overview

Many LLM-based multi-option decision pipelines evaluate candidates jointly in a single prompt, using next-token logits, sequence scoring, or generated answer tokens. Joint candidate presentation can introduce option-order sensitivity, while exhaustive permutation symmetrization requires evaluating all $K!$ candidate orders.

**InvariantOne** resolves these problems architecturally:
1. **Independent Branch Encoding:** Each candidate choice $[S, Q, O_k]$ is processed independently through an adapted `Qwen/Qwen3.5-4B-Base` backbone.
2. **Analytic Distributional Permutation Equivariance:** Option-span pooled representations pass into a lightweight direct comparative head (Projection: $\text{Linear}(2560 \to 256) \to \text{LayerNorm}(256)$; Absolute Scorer: $\text{Linear}(256 \to 256) \to \text{SiLU} \to \text{Linear}(256 \to 1)$), producing scalar scores normalized via calibrated softmax. The candidate-independent scoring architecture is exactly permutation-equivariant by construction in real arithmetic. In the finite-precision implementation audit, the measured distributional TVD was 0.000000 to six decimal places across all tested permutations. Discrete argmax selection showed a 1.56% top-choice flip rate in tied/near-tied cases due to tie-breaking behavior.
3. **Calibrated Confidence:** Output probabilities are normalized with a frozen temperature parameter ($\tau^* = 1.0091$).

---

## Architecture & Specifications

| Component | Specification |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen3.5-4B-Base` (32 transformer layers, $d_{\text{model}} = 2560$) |
| **Backbone Status** | Layers 0–27 strictly frozen (zero parameter updates) |
| **LoRA Adaptation** | Layers 28–30 (`in_proj_qkv`, `out_proj`) and layer 31 (`q_proj`, `v_proj`, `o_proj`); $r=8, \alpha=16$ |
| **Active LoRA Parameters** | 585,728 parameters |
| **Readout Head** | `DirectComparativeHead`: Projection (`Linear(2560 → 256, bias=True)` → `LayerNorm(256)`) + Absolute Scorer (`Linear(256 → 256, bias=True)` → `SiLU` → `Linear(256 → 1, bias=True)`) |
| **Active Head Parameters** | 722,177 parameters (frozen `absolute_only` inference mode: 656,128 projection + 66,049 scorer) |
| **Auxiliary Stored Head Parameters** | 262,657 parameters (`pair_net`, stored in `head.pt` checkpoint but bypassed during v1 inference) |
| **Total Stored Head Parameters** | 984,834 parameters (in checkpoint file `head.pt`) |
| **Total Active Adapted Parameters** | 1,307,905 parameters (585,728 LoRA + 722,177 Head; <0.05% of base model) |
| **Permutation Equivariance** | Mathematically exact by construction in real arithmetic (measured TVD = 0.000000 to six decimal places) |
| **Calibration Temperature** | $\tau^* = 1.0091$ (pre-calibrated temperature scaling) |

---

## Installation

```bash
# Install from PyPI
pip install invariantone

# With optional HTTP server support
pip install "invariantone[server]"

# Or install from repository
git clone https://github.com/tradertanmay/InvariantOne.git
cd InvariantOne
pip install -e .
```

---

## Quickstart

```python
from invariantone import InvariantOne

# Load frozen v1 checkpoint (auto-detects CUDA / Apple MPS / CPU)
model = InvariantOne.from_pretrained("invariantone-v1")

result = model.decide(
    state="Pressure is rising above the safe operating range.",
    question="What action should the controller take?",
    options=[
        "Close the inlet valve",
        "Increase feed pressure",
        "Maintain the current state",
        "Disable monitoring",
    ],
)

print(f"Selected: {result.choice}")
print(f"Index:    {result.choice_index}")
print(f"Probabilities: {[round(p, 4) for p in result.probabilities]}")
```

Output:
```text
Selected: Close the inlet valve
Index:    0
Probabilities: [0.7952, 0.0389, 0.1547, 0.0112]
```

---

## Dynamic Options Count ($K \ge 2$)

> **Scope Note on Candidate Options Count ($K$):**  
> The runtime supports arbitrary $K \ge 2$; the reported scientific evaluation results are for $K=4$ unless otherwise stated. Experimental generalization to arbitrary $K$ has not been independently established in the scientific benchmarks.

InvariantOne supports arbitrary numbers of runtime options:

```python
# Binary decision (K=2)
binary_res = model.decide(
    state="High packet loss observed on eth0.",
    question="Should traffic fail over to eth1?",
    options=["Fail over to eth1", "Stay on eth0"],
)

# Multi-way decision (K=5)
multi_res = model.decide(
    state="Temperature at 520 C, cooling loop nominal.",
    question="Select operational mode:",
    options=[
        "SCRAM shutdown",
        "Engage loop B",
        "Adjust control rods",
        "Vent steam",
        "Maintain baseline",
    ],
)
```

---

## Batch Decisions

Evaluate workloads with heterogeneous option counts:

```python
results = model.decide_batch([
    {
        "state": "Disk usage at 98%.",
        "question": "Action:",
        "options": ["Purge logs", "Extend disk", "Ignore"],
    },
    {
        "state": "CPU nominal.",
        "question": "Action:",
        "options": ["Scale down", "Maintain size"],
    },
])
```

---

## Command-Line Interface (CLI)

```bash
# Single decision (human-readable table)
invariantone decide \
  --state "Pressure is above limit" \
  --question "What should the system do?" \
  --option "Close inlet valve" \
  --option "Increase pressure" \
  --option "Maintain current state" \
  --option "Disable monitoring"

# Machine-readable JSON output
invariantone decide \
  --state "Pressure is above limit" \
  --question "What should the system do?" \
  --option "Close inlet valve" \
  --option "Increase pressure" \
  --json

# Verify checkpoint cryptographic integrity
invariantone verify

# Inspect model metadata
invariantone info
```

---

## HTTP Inference Service

Run the local API service:

```bash
invariantone serve --port 8000
```

Execute inference via REST:

```bash
curl -X POST http://127.0.0.1:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Pressure is rising above the safe operating range.",
    "question": "What action should the controller take?",
    "options": [
      "Close the inlet valve",
      "Increase feed pressure",
      "Maintain the current state",
      "Disable monitoring"
    ]
  }'
```

---

## Benchmark Performance Summary

Evaluated on `FINAL-HOLDOUT-V2` ($1,536$ four-choice items) under strict preregistered one-shot governance:

> **Benchmark Provenance Note:**  
> A100 performance figures are from the frozen FINAL-HOLDOUT-V2 benchmarking run; release-wheel functionality was independently smoke-tested in a fresh Python 3.12 environment.

| Evaluation Stratum | Focus | Unadapted $D_0$ | $B_1$-Raw | $B_1$-Sym ($24$-Perm) | **InvariantOne-v1** (Seed 42) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`FINAL-REAL`** | Real operational control tasks ($N=256$) | 89.45% | 91.41% | 92.58% | **94.14%** |
| **`FINAL-L1`** | Familiar-Operator Compositional Recombination ($N=256$) | 32.81% | 38.28% | 43.75% | **89.84%** |
| **`FINAL-L2a`** | Semantic Extrapolation ($N=256$) | 28.12% | 45.31% | 50.39% | **86.33%** |
| **`FINAL-L2b`** | Relational Extrapolation ($N=256$) | 22.27% | 62.50% | **65.23%** | 55.08% |
| **`FINAL-L3`** | Double Extrapolation ($N=512$) | 25.20% | 60.94% | **65.43%** | 56.84% |
| **`Aggregate`** | Total locked holdout pool ($N=1,536$) | 37.17% | 59.90% | 63.80% | **73.18%** (1,124/1,536) |

*Multi-seed aggregate mean across Seeds 42, 43, 44 is **71.74%** ($\pm 1.97\%$).*

**Permutation Equivariance:** Measured finite-precision $\text{TVD} = 0.000000$ across all 24 permutations (exact by construction in real arithmetic).  
**Inference Speedup:** $5.08\times$ faster than 24-permutation symmetrized causal baseline ($83.71\text{ ms}$ vs $425.31\text{ ms}$ on A100).

---

## Known Boundaries & Model Limitations

1. **Dynamic-$K$ Scientific Generalization:** While the runtime supports arbitrary $K \ge 2$ and exact permutation equivariance holds algebraically, empirical benchmark accuracy ($94.14\%$, $89.84\%$, etc.) was evaluated on 4-choice tasks ($K=4$). Experimental generalization to arbitrary $K$ has not been independently established in the scientific benchmarks.
2. **Novel Relational Operators:** InvariantOne excels at familiar-operator recombination ($89.84\%$) and semantic domain transfer ($86.33\%$). However, completely unseen operator grammars drop to $55.08\%$ ($L_{2b}$) and $56.84\%$ ($L_3$), where 24-permutation causal averaging remains stronger ($65.43\%$).
3. **Top-1 Argmax Tie Breaking:** Discrete argmax selection showed a 1.56% top-choice flip rate in tied/near-tied cases due to tie-breaking behavior under finite floating-point precision. The underlying continuous probabilistic output distribution is strictly permutation-equivariant in real arithmetic.
4. **No Generative Explanations:** InvariantOne is a dedicated decision model; it does not produce token-by-token natural-language rationales.
5. **Confidence Estimates:** Probabilities represent model confidence estimates and are not safety-critical certification guarantees.

---

## Documentation & Research

The v1 public release includes the runtime, tests, checkpoint hashes, model card, and release manifest. The full FINAL-HOLDOUT-V2 evaluation corpus and generation pipeline are not included in this repository.

- **GitHub Repository:** [https://github.com/tradertanmay/InvariantOne](https://github.com/tradertanmay/InvariantOne)
- **Hugging Face Model:** [https://huggingface.co/TanmaySah/InvariantOne-v1](https://huggingface.co/TanmaySah/InvariantOne-v1)
- **Model Card:** [`model_card/MODEL_CARD.md`](https://github.com/tradertanmay/InvariantOne/blob/main/model_card/MODEL_CARD.md)
- **Release Manifest:** [`INVARIANTONE_V1_RELEASE_MANIFEST.json`](https://github.com/tradertanmay/InvariantOne/blob/main/INVARIANTONE_V1_RELEASE_MANIFEST.json)
- **Reproduction Test:** [`tests/test_research_reproduction.py`](https://github.com/tradertanmay/InvariantOne/blob/main/tests/test_research_reproduction.py)
- **License Audit:** [`RELEASE_LICENSE_AUDIT.md`](https://github.com/tradertanmay/InvariantOne/blob/main/RELEASE_LICENSE_AUDIT.md)

---

## License

InvariantOne is licensed under the [Apache 2.0 License](LICENSE).
Base foundation model weights are licensed under the upstream Apache 2.0 license by Alibaba Cloud.
