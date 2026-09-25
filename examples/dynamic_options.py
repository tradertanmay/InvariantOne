"""
Dynamic Options Example with InvariantOne v1.

Demonstrates that InvariantOne supports arbitrary dynamic option counts K >= 2:
- Binary decision (K=2)
- Ternary decision (K=3)
- 5-way decision (K=5)
- 8-way decision (K=8)
"""

from invariantone import InvariantOne


def main() -> None:
    print("Loading InvariantOne v1.0.0...")
    model = InvariantOne.from_pretrained("invariantone-v1", device="cpu")

    # 1. Binary Decision (K=2)
    print("\n--- 1. Binary Decision (K=2) ---")
    res_k2 = model.decide(
        state="System health checks indicate packet drop on eth0 exceeding 15%.",
        question="Should the gateway reroute traffic to the secondary interface eth1?",
        options=[
            "Reroute traffic to secondary interface eth1",
            "Remain on primary interface eth0 and log drops",
        ],
    )
    print(f"Choice: {res_k2.choice}")
    print(f"Probabilities: {[round(p, 4) for p in res_k2.probabilities]}")

    # 2. 5-Way Decision (K=5)
    print("\n--- 2. Five-Way Decision (K=5) ---")
    res_k5 = model.decide(
        state="Reactor core temperature is elevated at 520 C and coolant loop pressure is normal.",
        question="Which operational mode should the supervisor select?",
        options=[
            "SCRAM immediate shutdown",
            "Engage auxiliary cooling loop B",
            "Reduce control rod insertion depth",
            "Vent containment steam",
            "Maintain baseline monitoring",
        ],
    )
    print(f"Choice: {res_k5.choice} (idx {res_k5.choice_index})")
    print(f"Probabilities: {[round(p, 4) for p in res_k5.probabilities]}")


if __name__ == "__main__":
    main()
