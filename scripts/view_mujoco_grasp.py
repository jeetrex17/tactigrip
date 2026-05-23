#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install MuJoCo first: uv pip install mujoco") from exc

from tactigrip.backends.mujoco_grasp import MuJoCoGraspSim
from tactigrip.sim.gripper import SimConfig, default_objects


def controller_action(sim: MuJoCoGraspSim, result) -> float:
    if not result.contact.in_contact:
        return 1.0

    force_error = sim.target_normal_force_n - result.contact.normal_force_n
    return float(np.clip(0.20 * force_error, -1.0, 1.0))


def make_sim(args) -> tuple[MuJoCoGraspSim, object]:
    config = SimConfig(
        max_time_s=6.0,
        lift_start_s=0.9,
        lift_speed_m_s=0.15,
        success_lift_m=0.26,
        required_hold_s=0.35,
        disturbance_start_s=1.8 if args.disturbance else None,
        disturbance_duration_s=1.0 if args.disturbance else 0.0,
        disturbance_friction_scale=0.55 if args.disturbance else 1.0,
    )
    sim = MuJoCoGraspSim(config)
    result = sim.reset(seed=args.seed, object_name=args.object)
    return sim, result


def configure_camera(viewer) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0.0, 0.0, 0.12]
    viewer.cam.distance = 0.42
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -22.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the native MuJoCo viewer for TactiGrip.")
    parser.add_argument("--object", choices=sorted(default_objects()), default="slippery_plastic")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--disturbance", action="store_true")
    parser.add_argument("--slowdown", type=float, default=4.0)
    parser.add_argument("--show-ui", action="store_true")
    parser.add_argument("--show-contacts", action="store_true")
    args = parser.parse_args()

    sim, result = make_sim(args)

    if sim.model is None or sim.data is None:
        raise RuntimeError("MuJoCo backend did not initialize.")

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        show_left_ui=args.show_ui,
        show_right_ui=args.show_ui,
    ) as viewer:
        configure_camera(viewer)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = args.show_contacts
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = args.show_contacts

        print("MuJoCo viewer")
        print("-------------")
        print("Close the viewer window or press Ctrl-C in the terminal to stop.")
        print("Mouse drag rotates the view. Right-drag or scroll zooms/pans.")
        print("Use --show-contacts for MuJoCo contact debug markers.")
        print(f"object:      {args.object}")
        print(f"disturbance: {args.disturbance}")

        while viewer.is_running():
            step_start = time.time()
            if result.terminated or result.truncated:
                print(
                    f"episode ended: {result.info['reason']}  "
                    f"height={result.state.object_height_m:.3f}m  "
                    f"force={result.contact.normal_force_n:.2f}N"
                )
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
                break

            result = sim.step(controller_action(sim, result))
            viewer.sync()

            frame_dt = sim.config.dt * max(args.slowdown, 0.0)
            elapsed = time.time() - step_start
            if frame_dt > elapsed:
                time.sleep(frame_dt - elapsed)


if __name__ == "__main__":
    main()
