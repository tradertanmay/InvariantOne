"""
HTTP Client Example for InvariantOne Inference Service.

Demonstrates sending a decision request to the running FastAPI server.
"""

import json
import urllib.request


def main() -> None:
    url = "http://127.0.0.1:8000/v1/decide"
    payload = {
        "state": "Pressure is rising above the safe operating range.",
        "question": "What action should the controller take?",
        "options": [
            "Close the inlet valve",
            "Increase feed pressure",
            "Maintain the current state",
            "Disable monitoring",
        ],
    }

    print(f"Sending decision request to {url}...")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("\nResponse from InvariantOne Service:")
            print(json.dumps(data, indent=2))
    except urllib.error.URLError as e:
        print(f"Could not connect to server: {e}")
        print("Start server first via: invariantone serve --port 8000")


if __name__ == "__main__":
    main()
