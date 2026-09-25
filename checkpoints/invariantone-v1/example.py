"""
Basic Decision Example with InvariantOne v1.

Demonstrates loading the frozen model and performing a 4-option decision.
"""

from invariantone import InvariantOne


def main() -> None:
    print("Loading InvariantOne v1.0.0 (D5-L4 / Nop=16 / Seed 42)...")
    model = InvariantOne.from_pretrained("invariantone-v1", device="cpu")

    state = "Pressure is rising above the safe operating range."
    question = "What action should the controller take?"
    options = [
        "Close the inlet valve",
        "Increase feed pressure",
        "Maintain the current state",
        "Disable monitoring",
    ]

    print(f"\nState:\n  {state}")
    print(f"Question:\n  {question}")
    print("Options:")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")

    result = model.decide(
        state=state,
        question=question,
        options=options,
    )

    print("\nDecision Result:")
    print(f"  Selected Choice: {result.choice} (index {result.choice_index})")
    print(f"  Probabilities:   {[round(p, 4) for p in result.probabilities]}")
    print(f"  Raw Scores:      {[round(s, 4) for s in result.scores]}")
    print(f"  Latency:         {result.latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
