#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tactigrip.envs.fragile_grasp_env import MODALITIES, FragileGraspEnv
from tactigrip.policies import EpisodeStats, summarize


def infer_modalities(path: Path) -> str:
    stem = path.stem
    for name in sorted(MODALITIES, key=len, reverse=True):
        if stem.endswith(name):
            return name
    return "full"


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


def run_policy_episode(
    model,
    modalities: str,
    seed: int,
    object_name: str,
    lift_start_s: float | None,
    max_time_s: float | None,
    max_target_force_n: float,
    disturbance_start_s: float | None,
    disturbance_duration_s: float,
    disturbance_friction_scale: float,
) -> EpisodeStats:
    env = FragileGraspEnv(
        modalities=modalities,
        object_name=object_name,
        lift_start_s=lift_start_s,
        max_time_s=max_time_s,
        max_target_force_n=max_target_force_n,
        disturbance_start_s=disturbance_start_s,
        disturbance_duration_s=disturbance_duration_s,
        disturbance_friction_scale=disturbance_friction_scale,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed, options={"object_name": object_name})

    forces: list[float] = []
    slip_steps = 0
    max_slip_velocity = 0.0
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        result = env.last_result
        forces.append(result.contact.normal_force_n)
        total_reward += float(reward)
        if result.contact.slip_velocity_m_s > 0.002:
            slip_steps += 1
        max_slip_velocity = max(max_slip_velocity, result.contact.slip_velocity_m_s)

    result = env.last_result
    return EpisodeStats(
        object_name=object_name,
        success=bool(info["success"]),
        dropped=bool(info["dropped"]),
        crushed=bool(info["crushed"]),
        duration_s=result.state.time_s,
        final_height_m=result.state.object_height_m,
        max_force_n=float(np.max(forces)) if forces else 0.0,
        mean_force_n=float(np.mean(forces)) if forces else 0.0,
        avg_slip_duration_s=slip_steps * env.sim.config.dt,
        max_slip_velocity_m_s=max_slip_velocity,
        reward=total_reward,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO tactile gripper policy.")
    parser.add_argument("--policy", type=Path, default=Path("models/ppo_force.zip"))
    parser.add_argument("--modalities", choices=sorted(MODALITIES), default=None)
    parser.add_argument("--episodes", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lift-start", type=float, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--max-target-force", type=float, default=8.0)
    parser.add_argument("--disturbance-start", type=float, default=None)
    parser.add_argument("--disturbance-duration", type=float, default=0.0)
    parser.add_argument("--disturbance-friction-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install RL dependencies first: uv pip install -r requirements.txt") from exc

    modalities = args.modalities or infer_modalities(args.policy)
    model = PPO.load(args.policy)
    object_names = ("fragile_foam", "slippery_plastic", "cool_metal")
    stats = [
        run_policy_episode(
            model,
            modalities,
            args.seed + idx,
            object_names[idx % len(object_names)],
            args.lift_start,
            args.max_time,
            args.max_target_force,
            args.disturbance_start,
            args.disturbance_duration,
            args.disturbance_friction_scale,
        )
        for idx in range(args.episodes)
    ]

    grouped: dict[str, list[EpisodeStats]] = defaultdict(list)
    for item in stats:
        grouped[item.object_name].append(item)

    report = {
        "policy": str(args.policy),
        "modalities": modalities,
        "lift_start_s": args.lift_start,
        "max_time_s": args.max_time,
        "max_target_force_n": args.max_target_force,
        "disturbance_start_s": args.disturbance_start,
        "disturbance_duration_s": args.disturbance_duration,
        "disturbance_friction_scale": args.disturbance_friction_scale,
        "aggregate": summarize(stats),
        "by_object": {name: summarize(items) for name, items in grouped.items()},
    }

    output = args.output or Path(f"runs/policy_eval_{modalities}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("PPO policy")
    print("----------")
    print(f"policy:     {args.policy}")
    print(f"modalities: {modalities}")
    print_summary("aggregate", report["aggregate"])
    for name in sorted(report["by_object"]):
        print_summary(name, report["by_object"][name])
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
