#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tactigrip.policies import evaluate_heuristic, summarize


def pct(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


def print_summary(name: str, summary: dict[str, float]) -> None:
    print(
        f"{name:16s}  "
        f"success {pct(summary['success_rate'])}  "
        f"drop {pct(summary['drop_rate'])}  "
        f"crush {pct(summary['crush_rate'])}  "
        f"force {summary['mean_force_n']:5.2f} N  "
        f"slip {summary['avg_slip_duration_s']:5.3f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the transparent heuristic gripper baseline.")
    parser.add_argument("--episodes", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("runs/heuristic_eval.json"))
    args = parser.parse_args()

    stats = evaluate_heuristic(episodes=args.episodes, seed=args.seed)
    grouped: dict[str, list] = defaultdict(list)
    for item in stats:
        grouped[item.object_name].append(item)

    report = {
        "aggregate": summarize(stats),
        "by_object": {name: summarize(items) for name, items in grouped.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("Heuristic baseline")
    print("------------------")
    print_summary("aggregate", report["aggregate"])
    for name in sorted(report["by_object"]):
        print_summary(name, report["by_object"][name])
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

