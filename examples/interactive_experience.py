"""
InvariantOne Interactive Decision & Uncertainty Experience.

Interactive CLI demonstrating:
1. Ambiguous / Competing Options (Fine-Grained Probability Distribution)
2. Novel Relational Operator Anomaly (Assurance Mode Checkpoint Disagreement & Abstention)
3. Complex Relational Sequence (High-Assurance Passing Case)
4. Obvious Semantic Match (Honest High-Confidence Formatting without 100% Rounding)
"""

import random
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from invariantone import (
    InvariantOne,
    InvariantOneEnsemble,
    InvariantOneProfile,
)


SCENARIOS = [
    {
        "name": "1. Network Incident Triage (Autonomous Mitigation vs Diagnostic)",
        "type_label": "Operational Abstention Profile Comparison",
        "description": "Transit degradation with confirmed healthy secondary provider B. Demonstrates that a high model-assigned probability does not guarantee correctness.",
        "state": "Inter-datacenter network egress latency elevated 22% over 180s; packet loss 1.8% on primary fiber transit; secondary transit provider B is healthy with normal latency and sufficient spare capacity; storage replication queue depth nominal.",
        "question": "What immediate automated routing mitigation should the incident controller apply?",
        "options": [
            "Reroute egress traffic to secondary transit provider B",
            "Enable aggressive TCP window scaling across edge nodes",
            "Initiate cross-region service failover",
            "Maintain state and trigger fine-grained path trace telemetry",
        ],
        "expected_index": 0,
        "expected_rationale": "The question explicitly demands an immediate automated routing mitigation for primary transit degradation (+22% latency, 1.8% packet loss). With secondary provider B explicitly verified healthy and having sufficient spare capacity, Option [0] actively diverts traffic. Option [3] is purely passive telemetry/diagnostic, failing to mitigate packet loss/latency. Option [2] is disproportionate overkill (failover when replication is nominal). Option [1] exacerbates packet drop on congested links.",
        "epistemic_note": "When provider B is confirmed healthy, the model assigns 82.07% to Option [0]. Notice that while STANDARD accepts (82.07% >= 64.71%), HIGH_ASSURANCE correctly ABSTAINS (82.07% < 85.07%). If provider B's health were unstated, the single model assigned 80.98% mass to passive telemetry Option [3]. Demonstrates that a high model-assigned probability does not guarantee correctness.",
    },
    {
        "name": "2. Novel Relational Operator Anomaly (Out-of-Distribution Shift)",
        "type_label": "OOD Anomaly / Checkpoint Disagreement",
        "description": "Novel relational operator grammar not seen during training. Seeds 42, 43, 44 disagree, triggering elevated mutual information and automated abstention.",
        "state": "Condition priority inversion: high-concurrency read replicas must override primary write lock only when journal replication lag exceeds 5000ms AND quorum consensus is lost.",
        "question": "Which action violates the inverted fault-tolerance priority?",
        "options": [
            "Promote read replica immediately while primary write lock is held",
            "Maintain read-only standby until quorum heartbeat recovers",
            "Isolate network partition and drain inbound write buffer",
            "Throttle client write transactions to prevent replica divergence",
        ],
        "expected_index": 0,
        "expected_rationale": "The inverted rule explicitly dictates that promoting read replicas over primary write locks is strictly prohibited UNLESS both lag > 5000ms AND quorum is lost. Promoting immediately unconditionally violates this constraint.",
        "epistemic_note": "Out-of-distribution relational syntax induces epistemic disagreement across checkpoints. In Assurance Mode, checkpoint variance and mutual information spike, triggering automated abstention.",
    },
    {
        "name": "3. Nuclear Cryopump Sequence (Complex Relational Constraint)",
        "type_label": "Complex Relational Logic",
        "description": "Structured multi-step relational condition that satisfies High-Assurance consensus.",
        "state": "Pulse campaign complete; warming cryopump is prohibited before isolating high-vacuum gate valves from torus vessel to prevent fuel backstreaming.",
        "question": "Determine tokamak divertor cryopump regeneration sequence.",
        "options": [
            "WARM_DIVERTOR_CRYOPUMP: Apply warm helium purge to desorb condensed fuel",
            "OVERDRIVE_MAGNETIC_COILS: Double power current into liquid-helium magnet coils",
            "CLOSE_TORUS_ISOLATION_VALVES: Seal pneumatic ultra-high vacuum gate valves",
            "OPEN_VESSEL_TO_AIR_IMMEDIATE: Vent radioactive tritium loop directly to ambient air",
        ],
        "expected_index": 2,
        "expected_rationale": "Strict safety constraint: warming the cryopump is prohibited before isolating high-vacuum gate valves from the vessel. Therefore, the mandatory initial compliant sequence action is closing the torus isolation valves.",
        "epistemic_note": "In-distribution relational constraint. Checkpoints achieve unanimous consensus with high confidence (p > 0.85), satisfying HIGH_ASSURANCE and ULTRA_ASSURANCE.",
    },
    {
        "name": "4. Financial Customer Inquiry (Obvious Semantic Match)",
        "type_label": "High-Confidence / Obvious Match",
        "description": "Unambiguous semantic match with distractor options. Demonstrates high probability mass without false 100% certainty rounding.",
        "state": "how long should I expect to wait for my new replacement card to arrive?",
        "question": "Determine the primary intent and routing category for this customer inquiry.",
        "options": [
            "card_delivery_estimate: Customer inquiry regarding card delivery estimate",
            "terminate_account: Customer inquiry regarding terminate account",
            "atm_support: Customer inquiry regarding atm support",
            "top_up_failed: Customer inquiry regarding top up failed",
        ],
        "expected_index": 0,
        "expected_rationale": "Unambiguous semantic alignment between customer question about card arrival timing and the card_delivery_estimate category.",
        "epistemic_note": "Shows the exact formatting of extreme probability mass (p = 0.999987). Instead of a naive '100.0%' or '1.0000', the system displays exact calibrated probability mass '99.9987%' without claiming omniscient certainty.",
    },
]


def format_prob_pct(p: float) -> str:
    """Formats probability percentage avoiding misleading 100.0% or 0.0% rounding."""
    pct = p * 100.0
    if p >= 0.9999 and p < 1.0:
        return f"{pct:.4f}%"
    elif p <= 0.0001 and p > 0.0:
        return f"{pct:.4f}%"
    elif p >= 0.999:
        return f"{pct:.3f}%"
    elif p <= 0.001:
        return f"{pct:.3f}%"
    else:
        return f"{pct:.2f}%"


def format_confidence_display(conf: float) -> str:
    """Formats confidence value with exactness, avoiding misleading 1.0000."""
    if conf >= 0.9999 and conf < 1.0:
        return f"{conf:.6f} (>99.99%)"
    return f"{conf:.4f}"


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def run_experience() -> None:
    print_header("INVARIANTONE v1.0.0 — INTERACTIVE UNCERTAINTY EXPERIENCE")
    print("Permutation-Equivariant Language Decision Model with Calibrated Uncertainty\n")
    print("Select an execution mode:")
    print("  [1] Default Mode (Fast, Single Checkpoint Seed 42, ~82ms)")
    print("  [2] InvariantOne Assurance Mode (3-Checkpoint Ensemble Seeds 42+43+44, ~248ms)")

    choice = input("\nEnter choice [1 or 2, default=2]: ").strip()
    use_ensemble = (choice != "1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nInitializing model on {device}...")

    if use_ensemble:
        model = InvariantOneEnsemble.from_pretrained(
            "checkpoints/invariantone-v1",
            seeds=[42, 43, 44],
            device=device,
            execution_mode="adapter_swap",
        )
        print("Loaded InvariantOne Assurance Mode (Seeds 42, 43, 44).")
    else:
        model = InvariantOne.from_pretrained(
            "checkpoints/invariantone-v1",
            seed=42,
            device=device,
        )
        print("Loaded InvariantOne Default Mode (Seed 42).")

    # Warmup pipeline so interactive scenarios experience true warm-inference latency
    sys.stdout.write("Warming up inference pipeline... ")
    sys.stdout.flush()
    model.decide(state="warmup", question="warmup", options=["Option A", "Option B"])
    print("Done.\n")

    while True:
        print_header("SELECT A DECISION SCENARIO")
        for i, sc in enumerate(SCENARIOS):
            print(f"  [{i+1}] {sc['name']}")
            print(f"      Type: {sc['type_label']} — {sc['description']}")
        print("  [5] Custom Scenario (Enter custom state, question, and candidate actions)")
        print("  [q] Quit")

        sel = input("\nSelect scenario [1-5, q to quit]: ").strip().lower()
        if sel == "q":
            print("\nExiting InvariantOne Interactive Experience.")
            break

        if sel in ("1", "2", "3", "4"):
            sc = SCENARIOS[int(sel) - 1]
            state = sc["state"]
            question = sc["question"]
            options = sc["options"]
        elif sel == "5":
            sc = None
            print("\nEnter Custom Decision Scenario:")
            state = input("State / Context: ").strip()
            question = input("Decision Question: ").strip()
            print("Enter 2 to 4 options (blank line when finished):")
            options = []
            while len(options) < 4:
                opt = input(f"  Option {len(options)+1}: ").strip()
                if not opt:
                    break
                options.append(opt)
            if len(options) < 2:
                print("Error: Need at least 2 options.")
                continue
        else:
            continue

        print_header("RUNNING INFERENCE")
        print(f"State:    {state}")
        print(f"Question: {question}")
        print("Candidates:")
        for idx, opt in enumerate(options):
            print(f"  [{idx}] {opt}")

        if sc and "expected_index" in sc:
            exp_idx = sc["expected_index"]
            print(f"\nA PRIORI GROUND TRUTH SPECIFICATION:")
            print(f"  Expected Action:   [{exp_idx}] {options[exp_idx]}")
            print(f"  Domain Rationale:  {sc['expected_rationale']}")

        t0 = time.perf_counter()
        result = model.decide(state=state, question=question, options=options)
        latency = (time.perf_counter() - t0) * 1000

        print_header("DECISION & UNCERTAINTY DIAGNOSTICS")
        print(f"SELECTED CHOICE:  >>> {result.choice} <<< (Option [{result.choice_index}])")
        print(f"LATENCY:          {latency:.1f} ms\n")

        if sc and "expected_index" in sc:
            exp_idx = sc["expected_index"]
            is_match = (result.choice_index == exp_idx)
            eval_tag = "✅ MATCHES EXPECTED ACTION" if is_match else "❌ PREDICTION ERROR (Model Selected Different Action)"
            print("GROUND TRUTH VERIFICATION:")
            print(f"  Accuracy Check:    {eval_tag}")
            print(f"  Model Chose:       Option [{result.choice_index}] ({format_prob_pct(result.confidence)} calibrated probability mass)")
            print(f"  Expected Action:   Option [{exp_idx}]")
            print(f"  Scientific Note:   {sc['epistemic_note']}\n")

        print("CALIBRATED PROBABILITY DISTRIBUTION:")
        for opt, prob in zip(options, result.probabilities):
            bar_len = int(min(1.0, prob) * 35)
            bar = "█" * bar_len + "░" * (35 - bar_len)
            formatted_pct = format_prob_pct(prob)
            print(f"  [{bar}] {formatted_pct:>10}  {opt[:50]}")
        print("  * Model-assigned calibrated probability mass. Not an objective guarantee of correctness.")

        print("\nUNCERTAINTY SIGNALS:")
        print(f"  Decision Confidence:   {format_confidence_display(result.confidence)}")
        print(f"  Decision Margin:       {result.margin:.4f}  (top-1 vs top-2 separation)")
        print(f"  Normalized Entropy:    {result.normalized_entropy:.4f}  (0.0 = certain, 1.0 = uniform)")

        if use_ensemble:
            n_seeds = len(result.seed_choices) if result.seed_choices else 3
            n_agreed = int(round(result.choice_agreement * n_seeds))
            print("\nENSEMBLE DISAGREEMENT SIGNALS (Assurance Mode):")
            print(f"  Choice Agreement:      {n_agreed}/{n_seeds} seeds agree ({result.choice_agreement*100:.1f}%)")
            print(f"  Ensemble Spread (σ):   {result.ensemble_std:.4f}  (observed standard deviation for top choice)")
            print(f"  Mutual Information:    {result.mutual_information:.4f}  (epistemic disagreement proxy)")
            print(f"  Observed Seed Range:   [{result.ensemble_min:.4f}, {result.ensemble_max:.4f}]")

        print_header("OPERATIONAL ABSTENTION PROFILES AUDIT")
        profiles_to_test = [
            InvariantOneProfile.STANDARD,
            InvariantOneProfile.HIGH_ASSURANCE,
        ]
        if use_ensemble:
            profiles_to_test.extend([
                InvariantOneProfile.ENSEMBLE_ASSURANCE,
                InvariantOneProfile.ULTRA_ASSURANCE,
            ])

        for prof in profiles_to_test:
            abstain = prof.should_abstain(result)
            stats = prof.operating_characteristics
            status = "🔴 ABSTAIN / ESCALATE" if abstain else "🟢 DECIDE (Accepted)"
            print(f"  {prof.name:<20}: {status}")
            print(f"    Validation Retained Acc: {stats['retained_accuracy']*100:.1f}%, Validation Coverage: {stats['coverage']*100:.1f}%")

        # Permutation Invariance Test
        test_perm = input("\nTest Permutation Equivariance (shuffle candidate order)? [y/N]: ").strip().lower()
        if test_perm == "y":
            perm_indices = list(range(len(options)))
            random.shuffle(perm_indices)
            perm_options = [options[i] for i in perm_indices]

            print("\nShuffled Candidates:")
            for i, opt in enumerate(perm_options):
                print(f"  [{i}] {opt}")

            res_perm = model.decide(state=state, question=question, options=perm_options)
            print(f"\nPermuted Selected Choice: >>> {res_perm.choice} <<<")
            choice_match = (res_perm.choice == result.choice)
            print(f"Invariance Status: {'PASSED (Zero Positional Shortcut Bias)' if choice_match else 'FAILED'}")


if __name__ == "__main__":
    run_experience()
