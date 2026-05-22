#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

MODALITIES = ("force", "force_accel", "force_acoustic", "force_temp", "full")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on the tactile grasp-lift-hold task.")
    parser.add_argument("--modalities", choices=MODALITIES, default="full")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        from tactigrip.envs.fragile_grasp_env import FragileGraspEnv
    except ImportError as exc:
        raise SystemExit("Install RL dependencies first: pip install -r requirements.txt") from exc

    def make_env():
        env = FragileGraspEnv(
            modalities=args.modalities,
            randomize_object=True,
            seed=args.seed,
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
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
    model.learn(total_timesteps=args.timesteps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"ppo_{args.modalities}"
    model.save(output)
    print(f"saved: {output}.zip")


if __name__ == "__main__":
    main()
