# InvariantOne v1.0.0 Release License & Redistribution Audit

**Audit Date:** September 25, 2026  
**Target Release:** `InvariantOne-v1` (Version 1.0.0)  
**Research Designation:** `D5-L4 / Nop=16 / Seed 42`  
**Audit Status:** APPROVED FOR PUBLIC OPEN-SOURCE REDISTRIBUTION

---

## 1. Scope of Audit

This audit evaluates all components of the InvariantOne v1.0.0 release packaging to verify that no copyright, licensing conflicts, or intellectual property restrictions encumber the public distribution of code, adapter weights, comparative heads, and documentation.

Components audited:
1. **Base Foundation Model Backbone**: `Qwen/Qwen3.5-4B-Base`
2. **Adapter Weights**: LoRA checkpoint (Frozen Phase 7R research checkpoint / `checkpoints/invariantone-v1/adapter_model.pt`)
3. **Head Weights**: DirectComparativeHead (Frozen Phase 7R research checkpoint / `checkpoints/invariantone-v1/head.pt`)
4. **Source Code**: Python package `invariantone`, CLI `cli/invariantone_cli.py`, and test suite
5. **Training and Evaluation Datasets**: InvariantOne synthetic operator corpora and `FINAL-HOLDOUT-V2`

---

## 2. Component-by-Component License Matrix

| Component | Upstream / Origin | Applicable License | Distribution Permitted? | Conditions / Attribution |
| :--- | :--- | :--- | :---: | :--- |
| **Backbone Model** | Alibaba Cloud (`Qwen/Qwen3.5-4B-Base`) | **Apache 2.0** | YES (Referenced) | Weights are downloaded on-demand from Hugging Face hub. Base weights are not vendored directly, avoiding base redistribution footprint while fully adhering to Apache 2.0 terms. |
| **LoRA Adapter Weights** | InvariantOne Research Team | **Apache 2.0** | YES | Delta weights trained with LoRA on upper layers (28–31). Standalone tensors contain no copyrighted upstream text. |
| **Comparative Head Weights** | InvariantOne Research Team | **Apache 2.0** | YES | Standalone PyTorch state dictionary (722,177 parameters) created entirely in-house. |
| **Repository Source Code** | InvariantOne Contributors | **Apache 2.0** | YES | Permissive open-source license. |
| **Tokenizer & Configs** | Hugging Face Hub / Qwen | **Apache 2.0** | YES | Tokenizer configuration and JSON schema metadata compatible with Apache 2.0. |
| **Training Datasets** | Procedural Relational Corpora ($N_{\text{op}}=16$) | **CC-BY-4.0 / Apache 2.0** | YES | In-house procedural and relational synthetic data generator without scraping of copyrighted third-party works. |
| **Evaluation Datasets** | `FINAL-HOLDOUT-V2` ($1,536$ examples) | **Preregistered Evaluation Archive** | Research Use Only | Locked hash manifest ensures benchmark integrity. |

---

## 3. Dependency License Compatibility

All runtime Python dependencies declared in `pyproject.toml` were audited for permissive licensing:

| Package | Minimum Version | License | Compatibility Status |
| :--- | :--- | :--- | :---: |
| `torch` | $\ge 2.1.0$ | BSD-style | Compatible |
| `transformers` | $\ge 4.45.0$ | Apache 2.0 | Compatible |
| `accelerate` | $\ge 0.30.0$ | Apache 2.0 | Compatible |
| `pydantic` | $\ge 2.0.0$ | MIT | Compatible |
| `numpy` | $\ge 1.24.0$ | BSD-3-Clause | Compatible |
| `scipy` | $\ge 1.10.0$ | BSD-3-Clause | Compatible |
| `click` | $\ge 8.0.0$ | BSD-3-Clause | Compatible |
| `rich` | $\ge 13.0.0$ | MIT | Compatible |
| `fastapi` | $\ge 0.100.0$ | MIT | Compatible |
| `uvicorn` | $\ge 0.20.0$ | BSD-3-Clause | Compatible |

**GPL / AGPL Contamination Check:** Negative. No copyleft, GPL, AGPL, or restrictive non-commercial (NC) licenses are present in the core runtime dependency tree.

---

## 4. Distribution Strategy

To adhere strictly to open-weight best practices:
1. **No Vendoring of 4B Base Weights:** InvariantOne distributes only the lightweight adapter weights ($1.1\text{ MB}$) and comparative head ($1.9\text{ MB}$), accompanied by cryptographic hashes and the explicit upstream model identifier `Qwen/Qwen3.5-4B-Base` (revision `1001bb4d826a52d1f399e183466143f4da7b741b`).
2. **Attribution:** Upstream Qwen model attribution is maintained in the Model Card, Documentation, and Package metadata.
3. **Patent Retaliation Clause:** Standard Apache 2.0 patent grants apply to all InvariantOne code and model heads.

---

## 5. Audit Conclusion

The InvariantOne v1.0.0 release is **CLEAR FOR DISTRIBUTION** under the Apache 2.0 license.
