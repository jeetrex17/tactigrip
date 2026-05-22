# TactiGrip

Multimodal tactile gripper simulation for slip-aware, force-limited manipulation.

The first task is a fragile grasp-lift-hold benchmark: a parallel-jaw gripper must lift and hold an object without dropping it or exceeding its crush force. The project compares force-only control against richer tactile observations: force, shear, acoustic slip energy, fingertip acceleration, and temperature.

## Quick Start

```bash
uv run python scripts/validate_sensors.py
uv run python scripts/evaluate_heuristic.py
```

For RL training:

```bash
uv pip install -r requirements.txt
uv run python scripts/train_ppo.py --modalities full --timesteps 200000
```

## Scope

- Core simulator runs with NumPy only.
- Gymnasium/SB3 are used for PPO training.
- Isaac Lab is a later port once a Linux + NVIDIA RTX machine is available.
