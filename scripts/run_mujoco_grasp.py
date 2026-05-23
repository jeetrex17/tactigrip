#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tactigrip.backends.mujoco_grasp import run_scripted_mujoco_episode
from tactigrip.sim.gripper import default_objects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a MuJoCo contact-physics grasp-lift smoke test."
    )
    parser.add_argument("--object", choices=sorted(default_objects()), default="slippery_plastic")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--disturbance", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    _, stats = run_scripted_mujoco_episode(
        object_name=args.object,
        seed=args.seed,
        disturbance=args.disturbance,
    )

    print("MuJoCo grasp")
    print("------------")
    print(f"object:       {args.object}")
    print(f"disturbance:  {args.disturbance}")
    print(f"result:       {stats['reason']}")
    print(f"duration:     {stats['duration_s']:.2f} s")
    print(f"height:       {stats['final_height_m']:.3f} m")
    print(f"max force:    {stats['max_force_n']:.2f} N")
    print(f"mean force:   {stats['mean_force_n']:.2f} N")
    print(f"slip:         {stats['slip_duration_s']:.3f} s")
    print(f"peak slip:    {stats['peak_slip_m_s']:.4f} m/s")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"wrote:        {args.output}")


if __name__ == "__main__":
    main()
