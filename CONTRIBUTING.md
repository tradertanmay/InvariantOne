# Contributing to InvariantOne

Thank you for your interest in contributing to InvariantOne!

InvariantOne is an open-source, permutation-equivariant natural-language decision model built on top of `Qwen/Qwen3.5-4B-Base`.

---

## Development Setup

1. **Clone or navigate to the repository:**
   ```bash
   git clone https://github.com/tradertanmay/InvariantOne.git
   cd InvariantOne
   ```

2. **Create and activate a virtual environment (Python 3.10+):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the package in editable mode with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run the test suite:**
   ```bash
   pytest -v
   ```

5. **Lint and format code:**
   ```bash
   ruff check .
   ```

---

## Architectural & Invariance Guidelines

When making changes to the decision pipeline:

1. **Exact Permutation Equivariance:**
   Candidate options must be independently encoded through the transformer backbone. Under no circumstances should candidate options be concatenated sequentially into a single autoregressive prompt context, as that reintroduces permutation and position bias.
   
2. **Dynamic-$K$ Compatibility:**
   All scoring and ranking functions must support arbitrary option counts $K \ge 2$. Never hardcode assumptions that $K = 4$.

3. **Calibration & Uncertainty:**
   Probabilities are calibrated with temperature scaling ($\tau^* = 1.0091$). Any changes to logits must preserve calibration diagnostics or update temperature parameters systematically. Conformal prediction features are strictly research-grade.

4. **Testing Requirements:**
   - Any modification to model architectures must include tests verifying $\text{TVD} = 0.0$ across candidate permutations.
   - Run reproduction verification: `pytest tests/test_research_reproduction.py`

---

## Pull Request Process

1. Ensure all tests pass cleanly (`pytest -v`).
2. Verify code conforms to Ruff formatting (`ruff check .`).
3. Add unit or property tests for any newly added features.
4. Ensure documentation and Model Card remain synchronized.
5. All contributions are licensed under the [Apache 2.0 License](LICENSE).
