# TactiGrip

Multimodal tactile gripper simulation for slip-aware, force-limited manipulation.

The first task is a fragile grasp-lift-hold benchmark: a parallel-jaw gripper must lift and hold an object without dropping it or exceeding its crush force. The project compares normal-force-only control against richer tactile observations: shear, acoustic slip energy, fingertip acceleration, and temperature.

## What It Tests

The gripper has to lift fragile, slippery, and metallic objects through a temporary friction drop. Too little force drops the object. Too much force crushes fragile objects. The benchmark asks whether richer tactile signals help the policy stay robust when normal force alone is not enough.

## Results

PPO policies were trained with the same task settings and evaluated for 120 episodes per scenario.

| Policy | Scenario | Success | Drops | Crushes | Mean Force N | Slip Duration s | Peak Slip m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| force | clean | 100.0% | 0.0% | 0.0% | 2.46 | 0.015 | 0.0158 |
| force | friction_drop | 66.7% | 33.3% | 0.0% | 2.43 | 1.009 | 0.0167 |
| force_shear | friction_drop | 100.0% | 0.0% | 0.0% | 2.66 | 1.013 | 0.0152 |
| force_acoustic | friction_drop | 100.0% | 0.0% | 0.0% | 2.68 | 1.007 | 0.0139 |
| force_accel | friction_drop | 100.0% | 0.0% | 0.0% | 2.86 | 1.005 | 0.0115 |
| force_temp | friction_drop | 100.0% | 0.0% | 0.0% | 2.90 | 1.008 | 0.0115 |
| full | friction_drop | 100.0% | 0.0% | 0.0% | 2.89 | 1.032 | 0.0125 |

Normal-force-only control drops objects under friction disturbance. Adding tactile modalities recovers 100% success in this benchmark.

## Quick Start

```bash
uv run python scripts/validate_sensors.py
uv run python scripts/evaluate_heuristic.py
```

For RL training:

```bash
uv pip install -r requirements.txt
uv run python scripts/train_ppo.py --modalities full --randomize-object --lift-start 2.0 --max-time 7.0 --disturbance-start 3.2 --disturbance-duration 1.5 --disturbance-friction-scale 0.45 --timesteps 300000
```

Run the policy ablation:

```bash
uv run python scripts/benchmark_policies.py --episodes 120
```

## Rerun Demo

Clean gripper demo:

```bash
uv run python scripts/visualize_rerun.py --policy models/ppo_full.zip --modalities full --object slippery_plastic --seed 5 --spawn
```

Debug overlays:

```bash
uv run python scripts/visualize_rerun.py --policy models/ppo_full.zip --modalities full --object slippery_plastic --seed 5 --spawn --show-sensors --show-guides
```

## Scope

- Core simulator runs with NumPy only.
- Gymnasium/SB3 are used for PPO training.
- Rerun is used for technical telemetry and gripper visualization.
- Isaac Lab is a later port once a Linux + NVIDIA RTX machine is available.
