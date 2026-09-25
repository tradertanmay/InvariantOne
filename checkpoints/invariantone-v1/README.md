# InvariantOne-v1 Model Checkpoint

This directory contains the frozen, validated weights and configuration for `InvariantOne-v1` (Research Designation: `D5-L4 / Nop=16 / Seed 42`).

## Contents

- `adapter_model.pt`: LoRA adapter weights for layers 28–31 ($r=8, \alpha=16$)
- `head.pt`: Direct Comparative Head weights ($722,177$ parameters)
- `invariantone_config.json`: Model configuration and verified SHA-256 hashes
- `config.json`: Hugging Face compatible metadata
- `MODEL_CARD.md`: Complete scientific model card and benchmark results
- `example.py`: Minimal inference script

## Quickstart

```python
from invariantone import InvariantOne

model = InvariantOne.from_pretrained("checkpoints/invariantone-v1")

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

print(result.choice)
print(result.probabilities)
```

## Cryptographic Hashes

- LoRA Adapter SHA-256: `a0000d3ec984d155fbad5abc854556904ecffe78b997900b677a8d8d5811e4d2`
- Comparative Head SHA-256: `190d867dc302a89d420d19a8374a5db252ea2e71e4f40c68064ded23fc321384`
- Base Model: `Qwen/Qwen3.5-4B-Base` (Revision: `1001bb4d826a52d1f399e183466143f4da7b741b`)

## Evaluation & Benchmark Provenance

- **Candidate Options Count ($K$):** The runtime supports arbitrary $K \ge 2$; the reported scientific evaluation results are for $K=4$ unless otherwise stated.
- **Benchmark Provenance:** A100 performance figures are from the frozen FINAL-HOLDOUT-V2 benchmarking run; release-wheel functionality was independently smoke-tested in a fresh Python 3.12 environment.
