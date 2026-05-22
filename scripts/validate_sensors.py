#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tactigrip.sim.gripper import FragileGraspSim


def run_validation(seed: int, output: Path, target_force_n: float) -> dict[str, float]:
    sim = FragileGraspSim()
    result = sim.reset(seed=seed, object_name="fragile_foam")
    rows: list[dict[str, float | bool]] = []

    while not (result.terminated or result.truncated):
        error = target_force_n - result.tactile.normal_force_n
        if error > 0.08:
            action = 0.35
        elif error < -0.08:
            action = -0.35
        else:
            action = 0.0
        result = sim.step(action)
        accel_norm = float(
            np.linalg.norm(
                [
                    result.tactile.accel_x_m_s2,
                    result.tactile.accel_y_m_s2,
                    result.tactile.accel_z_m_s2,
                ]
            )
        )
        rows.append(
            {
                "time_s": result.state.time_s,
                "normal_force_n": result.tactile.normal_force_n,
                "shear_force_n": result.tactile.shear_force_n,
                "acoustic_energy": result.tactile.acoustic_energy,
                "accel_norm_m_s2": accel_norm,
                "temperature_c": result.tactile.temperature_c,
                "slip_velocity_m_s": result.contact.slip_velocity_m_s,
                "slip_distance_m": result.state.slip_distance_m,
                "in_contact": result.contact.in_contact,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    slip_rows = [row for row in rows if float(row["slip_velocity_m_s"]) > 0.002]
    slip_onset_s = float(slip_rows[0]["time_s"]) if slip_rows else float("nan")
    peak_acoustic = max(float(row["acoustic_energy"]) for row in rows)
    peak_accel = max(float(row["accel_norm_m_s2"]) for row in rows)
    peak_force = max(float(row["normal_force_n"]) for row in rows)

    return {
        "slip_onset_s": slip_onset_s,
        "peak_acoustic": peak_acoustic,
        "peak_accel_m_s2": peak_accel,
        "peak_force_n": peak_force,
        "duration_s": rows[-1]["time_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic tactile signals in a known slip case.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-force", type=float, default=0.55)
    parser.add_argument("--output", type=Path, default=Path("runs/sensor_validation.csv"))
    args = parser.parse_args()

    metrics = run_validation(args.seed, args.output, args.target_force)

    print("Sensor validation")
    print("-----------------")
    print(f"wrote:          {args.output}")
    print(f"slip onset:     {metrics['slip_onset_s']:.3f} s")
    print(f"peak acoustic:  {metrics['peak_acoustic']:.4f}")
    print(f"peak accel:     {metrics['peak_accel_m_s2']:.4f} m/s^2")
    print(f"peak force:     {metrics['peak_force_n']:.3f} N")
    print(f"duration:       {metrics['duration_s']:.3f} s")


if __name__ == "__main__":
    main()
