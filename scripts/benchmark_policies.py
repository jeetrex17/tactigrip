#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tactigrip.envs.fragile_grasp_env import MODALITIES
from tactigrip.policies import EpisodeStats, summarize

from evaluate_policy import run_policy_episode


DEFAULT_MODALITIES = ("force", "force_shear", "force_acoustic", "force_accel", "force_temp", "full")
OBJECT_NAMES = ("fragile_foam", "slippery_plastic", "cool_metal")


def evaluate_policy(
    policy_path: Path,
    modalities: str,
    episodes: int,
    seed: int,
    lift_start_s: float,
    max_time_s: float,
    disturbance: bool,
    disturbance_start_s: float,
    disturbance_duration_s: float,
    disturbance_friction_scale: float,
    disturbance_slip_penalty: float,
):
    from stable_baselines3 import PPO

    model = PPO.load(policy_path)
    stats: list[EpisodeStats] = []
    for idx in range(episodes):
        stats.append(
            run_policy_episode(
                model=model,
                modalities=modalities,
                seed=seed + idx,
                object_name=OBJECT_NAMES[idx % len(OBJECT_NAMES)],
                lift_start_s=lift_start_s,
                max_time_s=max_time_s,
                max_target_force_n=8.0,
                disturbance_start_s=disturbance_start_s if disturbance else None,
                disturbance_duration_s=disturbance_duration_s if disturbance else 0.0,
                disturbance_friction_scale=disturbance_friction_scale if disturbance else 1.0,
                disturbance_slip_penalty_scale=disturbance_slip_penalty,
            )
        )
    grouped: dict[str, list[EpisodeStats]] = defaultdict(list)
    for item in stats:
        grouped[item.object_name].append(item)
    return {
        "aggregate": summarize(stats),
        "by_object": {name: summarize(items) for name, items in grouped.items()},
    }


def metric(summary: dict, key: str) -> float:
    return float(summary["aggregate"][key])


def format_rate(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Policy | Scenario | Success | Drops | Crushes | Mean Force N | Slip Duration s | Peak Slip m/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary = row["summary"]
        lines.append(
            "| "
            f"{row['modalities']} | "
            f"{row['scenario']} | "
            f"{format_rate(metric(summary, 'success_rate'))} | "
            f"{format_rate(metric(summary, 'drop_rate'))} | "
            f"{format_rate(metric(summary, 'crush_rate'))} | "
            f"{metric(summary, 'mean_force_n'):.2f} | "
            f"{metric(summary, 'avg_slip_duration_s'):.3f} | "
            f"{metric(summary, 'mean_peak_slip_velocity_m_s'):.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained tactile policies across clean and stress scenarios.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--modalities", nargs="+", choices=sorted(MODALITIES), default=list(DEFAULT_MODALITIES))
    parser.add_argument("--episodes", type=int, default=90)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lift-start", type=float, default=2.0)
    parser.add_argument("--max-time", type=float, default=7.0)
    parser.add_argument("--disturbance-start", type=float, default=3.2)
    parser.add_argument("--disturbance-duration", type=float, default=1.5)
    parser.add_argument("--disturbance-friction-scale", type=float, default=0.45)
    parser.add_argument("--disturbance-slip-penalty", type=float, default=30.0)
    parser.add_argument("--output-json", type=Path, default=Path("runs/policy_benchmark.json"))
    parser.add_argument("--output-md", type=Path, default=Path("runs/policy_benchmark.md"))
    args = parser.parse_args()

    rows: list[dict] = []
    for modalities in args.modalities:
        policy_path = args.models_dir / f"ppo_{modalities}.zip"
        if not policy_path.exists():
            print(f"skip missing policy: {policy_path}")
            continue

        for scenario, disturbance in (("clean", False), ("friction_drop", True)):
            summary = evaluate_policy(
                policy_path=policy_path,
                modalities=modalities,
                episodes=args.episodes,
                seed=args.seed,
                lift_start_s=args.lift_start,
                max_time_s=args.max_time,
                disturbance=disturbance,
                disturbance_start_s=args.disturbance_start,
                disturbance_duration_s=args.disturbance_duration,
                disturbance_friction_scale=args.disturbance_friction_scale,
                disturbance_slip_penalty=args.disturbance_slip_penalty,
            )
            rows.append(
                {
                    "modalities": modalities,
                    "policy": str(policy_path),
                    "scenario": scenario,
                    "summary": summary,
                }
            )
            print(
                f"{modalities:14s} {scenario:13s} "
                f"success={format_rate(metric(summary, 'success_rate'))} "
                f"force={metric(summary, 'mean_force_n'):.2f}N "
                f"slip={metric(summary, 'avg_slip_duration_s'):.3f}s "
                f"peak={metric(summary, 'mean_peak_slip_velocity_m_s'):.4f}m/s"
            )

    report = {
        "episodes": args.episodes,
        "seed": args.seed,
        "lift_start_s": args.lift_start,
        "max_time_s": args.max_time,
        "disturbance_start_s": args.disturbance_start,
        "disturbance_duration_s": args.disturbance_duration,
        "disturbance_friction_scale": args.disturbance_friction_scale,
        "disturbance_slip_penalty": args.disturbance_slip_penalty,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    args.output_md.write_text(markdown_table(rows))
    print(f"wrote: {args.output_json}")
    print(f"wrote: {args.output_md}")


if __name__ == "__main__":
    main()
