#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MODALITIES = ("force", "force_shear", "force_accel", "force_acoustic", "force_temp", "full")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on the tactile grasp-lift-hold task.")
    parser.add_argument("--modalities", choices=MODALITIES, default="full")
    parser.add_argument("--backend", choices=("fast", "mujoco"), default="fast")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--object-name", default="fragile_foam")
    parser.add_argument("--randomize-object", action="store_true")
    parser.add_argument("--lift-start", type=float, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--max-target-force", type=float, default=8.0)
    parser.add_argument("--disturbance-start", type=float, default=None)
    parser.add_argument("--disturbance-duration", type=float, default=0.0)
    parser.add_argument("--disturbance-friction-scale", type=float, default=1.0)
    parser.add_argument("--disturbance-slip-penalty", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=18)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        from tactigrip.envs.fragile_grasp_env import FragileGraspEnv
    except ImportError as exc:
        raise SystemExit("Install RL dependencies first: pip install -r requirements.txt") from exc

    def make_env():
        if args.backend == "mujoco":
            from tactigrip.envs.mujoco_grasp_env import MuJoCoGraspEnv

            if args.randomize_object:
                raise SystemExit("--randomize-object is not supported by the MuJoCo backend yet")
            env = MuJoCoGraspEnv(
                modalities=args.modalities,
                object_name=args.object_name,
                lift_start_s=args.lift_start if args.lift_start is not None else 0.9,
                max_time_s=args.max_time if args.max_time is not None else 4.5,
                max_target_force_n=args.max_target_force,
                disturbance_start_s=args.disturbance_start,
                disturbance_duration_s=args.disturbance_duration,
                disturbance_friction_scale=args.disturbance_friction_scale,
                seed=args.seed,
            )
        else:
            env = FragileGraspEnv(
                modalities=args.modalities,
                object_name=args.object_name,
                randomize_object=args.randomize_object,
                lift_start_s=args.lift_start,
                max_time_s=args.max_time,
                max_target_force_n=args.max_target_force,
                disturbance_start_s=args.disturbance_start,
                disturbance_duration_s=args.disturbance_duration,
                disturbance_friction_scale=args.disturbance_friction_scale,
                disturbance_slip_penalty_scale=args.disturbance_slip_penalty,
                seed=args.seed,
            )
        return Monitor(env)

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1 if args.verbose else 0,
        seed=args.seed,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        policy_kwargs={"net_arch": [128, 128]},
        device="cpu",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / (
        f"ppo_mujoco_{args.modalities}" if args.backend == "mujoco" else f"ppo_{args.modalities}"
    )
    callback = None
    best_dir = args.out_dir / (
        f"best_mujoco_{args.modalities}" if args.backend == "mujoco" else f"best_{args.modalities}"
    )
    if args.eval_freq > 0:
        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(best_dir),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1 if args.verbose else 0,
        )

    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(output)
    best_model = best_dir / "best_model.zip"
    if best_model.exists():
        shutil.copyfile(best_model, output.with_suffix(".zip"))
    print(f"saved: {output}.zip")


if __name__ == "__main__":
    main()
