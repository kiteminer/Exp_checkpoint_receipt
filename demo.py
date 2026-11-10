"""Audit committed training samples across a simulated interruption."""
import argparse
import json
from collections import Counter
from pathlib import Path

DATA = list(range(1, 17))
BATCH = 4


def train(state, samples):
    for sample in samples:
        x = sample / len(DATA)
        state["weight"] += 0.1 * (2 * x - state["weight"] * x) * x
        state["receipt"].append(sample)


def start(path):
    state = {"weight": 0.0, "receipt": [], "consumed": 0, "fetched": 0}
    for offset in (0, BATCH):
        train(state, DATA[offset:offset + BATCH])
    state["consumed"] = len(state["receipt"])
    buffer = DATA[state["consumed"]:state["consumed"] + BATCH]
    state["fetched"] = state["consumed"] + len(buffer)
    with path.open("x") as out:
        json.dump(state, out)
    print("Saved after training on 1-8. Prefetched 9-12. Run resume next.")


def resume(path):
    baseline = {"weight": 0.0, "receipt": []}
    train(baseline, DATA)
    for cursor in ("fetched", "consumed"):
        state = json.loads(path.read_text())
        train(state, DATA[state[cursor]:])
        counts = Counter(state["receipt"])
        missing = sorted(set(DATA) - counts.keys())
        repeated = sorted(key for key, count in counts.items() if count > 1)
        matches = state["receipt"] == baseline["receipt"] and state["weight"] == baseline["weight"]
        print(f"{cursor}: missing={missing} repeated={repeated} matches_baseline={matches}")
        if matches != (cursor == "consumed"):
            raise SystemExit("Unexpected audit result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resume"))
    parser.add_argument("--file", type=Path, default=Path("checkpoint.json"))
    args = parser.parse_args()
    try:
        (start if args.action == "start" else resume)(args.file)
    except FileExistsError:
        parser.exit(1, "Checkpoint already exists. Choose a fresh --file path.\n")
    except FileNotFoundError:
        parser.exit(1, "Checkpoint missing. Run start first with the same --file path.\n")
