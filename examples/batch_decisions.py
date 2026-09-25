"""
Batch Decisions Example with InvariantOne v1.

Demonstrates evaluating multiple distinct decisions in a single call,
where each decision may have a different number of options.
"""

from invariantone import InvariantOne


def main() -> None:
    print("Loading InvariantOne v1.0.0...")
    model = InvariantOne.from_pretrained("invariantone-v1", device="cpu")

    workload = [
        {
            "state": "Disk utilization on /var/log reached 98%.",
            "question": "What maintenance task should be initiated?",
            "options": [
                "Rotate and compress older log archives",
                "Increase filesystem quota allocation",
                "Ignore warning until 100% full",
            ],
        },
        {
            "state": "Memory consumption is 45% and CPU load is nominal.",
            "question": "Should auto-scaler spin up additional worker nodes?",
            "options": [
                "Provision 2 additional worker nodes",
                "Do not provision additional nodes",
            ],
        },
    ]

    print(f"\nProcessing batch of {len(workload)} decisions...")
    batch_result = model.decide_batch(workload)

    for i, res in enumerate(batch_result):
        print(f"\nDecision #{i+1}:")
        print(f"  Question: {res.question}")
        print(f"  Selected: {res.choice} (index {res.choice_index})")
        print(f"  Probabilities: {[round(p, 4) for p in res.probabilities]}")

    print(f"\nTotal Batch Latency: {batch_result.total_latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
