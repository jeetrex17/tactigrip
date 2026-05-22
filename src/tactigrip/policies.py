from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tactigrip.sim.gripper import FragileGraspSim, StepResult


@dataclass
class EpisodeStats:
    object_name: str
    success: bool
    dropped: bool
    crushed: bool
    duration_s: float
    final_height_m: float
    max_force_n: float
    mean_force_n: float
    avg_slip_duration_s: float
    max_slip_velocity_m_s: float
    reward: float


class HeuristicGripController:
    """Simple force controller used as the non-learning benchmark.

    It closes until the measured force reaches a target, holds near that target,
    and raises the target when slip evidence appears. This is intentionally
    transparent so PPO comparisons are not against an undefined baseline.
    """

    def __init__(
        self,
        initial_target_force_n: float = 2.0,
        force_margin_n: float = 0.15,
        slip_force_increment_n: float = 0.35,
        acoustic_threshold: float = 0.025,
        slip_velocity_threshold_m_s: float = 0.003,
        force_cap_ratio: float = 0.88,
    ) -> None:
        self.initial_target_force_n = initial_target_force_n
        self.force_margin_n = force_margin_n
        self.slip_force_increment_n = slip_force_increment_n
        self.acoustic_threshold = acoustic_threshold
        self.slip_velocity_threshold_m_s = slip_velocity_threshold_m_s
        self.force_cap_ratio = force_cap_ratio
        self.target_force_n = initial_target_force_n

    def reset(self) -> None:
        self.target_force_n = self.initial_target_force_n

    def act(self, result: StepResult, crush_force_n: float) -> float:
        tactile = result.tactile
        contact = result.contact
        force_cap = self.force_cap_ratio * crush_force_n

        slip_evidence = (
            tactile.acoustic_energy > self.acoustic_threshold
            or contact.slip_velocity_m_s > self.slip_velocity_threshold_m_s
        )
        if slip_evidence:
            self.target_force_n = min(force_cap, self.target_force_n + self.slip_force_increment_n)

        if tactile.normal_force_n < self.target_force_n - self.force_margin_n:
            return 1.0
        if tactile.normal_force_n > self.target_force_n + self.force_margin_n:
            return -0.25
        return 0.0


def run_episode(
    sim: FragileGraspSim,
    controller: HeuristicGripController,
    seed: int,
    object_name: str,
) -> EpisodeStats:
    result = sim.reset(seed=seed, object_name=object_name)
    controller.reset()

    forces: list[float] = []
    slip_steps = 0
    max_slip_velocity = 0.0
    total_reward = 0.0

    while not (result.terminated or result.truncated):
        action = controller.act(result, sim.object_profile.crush_force_n)
        result = sim.step(action)
        forces.append(result.contact.normal_force_n)
        total_reward += result.reward
        if result.contact.slip_velocity_m_s > 0.002:
            slip_steps += 1
        max_slip_velocity = max(max_slip_velocity, result.contact.slip_velocity_m_s)

    info = result.info
    return EpisodeStats(
        object_name=object_name,
        success=bool(info["success"]),
        dropped=bool(info["dropped"]),
        crushed=bool(info["crushed"]),
        duration_s=result.state.time_s,
        final_height_m=result.state.object_height_m,
        max_force_n=float(np.max(forces)) if forces else 0.0,
        mean_force_n=float(np.mean(forces)) if forces else 0.0,
        avg_slip_duration_s=slip_steps * sim.config.dt,
        max_slip_velocity_m_s=max_slip_velocity,
        reward=total_reward,
    )


def evaluate_heuristic(
    episodes: int = 100,
    seed: int = 7,
    object_names: tuple[str, ...] = ("fragile_foam", "slippery_plastic", "cool_metal"),
) -> list[EpisodeStats]:
    sim = FragileGraspSim()
    controller = HeuristicGripController()
    stats: list[EpisodeStats] = []
    for idx in range(episodes):
        object_name = object_names[idx % len(object_names)]
        stats.append(run_episode(sim, controller, seed + idx, object_name))
    return stats


def summarize(stats: list[EpisodeStats]) -> dict[str, float]:
    if not stats:
        raise ValueError("cannot summarize an empty stats list")
    return {
        "episodes": float(len(stats)),
        "success_rate": float(np.mean([s.success for s in stats])),
        "drop_rate": float(np.mean([s.dropped for s in stats])),
        "crush_rate": float(np.mean([s.crushed for s in stats])),
        "mean_force_n": float(np.mean([s.mean_force_n for s in stats])),
        "max_force_n": float(np.max([s.max_force_n for s in stats])),
        "avg_slip_duration_s": float(np.mean([s.avg_slip_duration_s for s in stats])),
        "mean_peak_slip_velocity_m_s": float(np.mean([s.max_slip_velocity_m_s for s in stats])),
        "max_peak_slip_velocity_m_s": float(np.max([s.max_slip_velocity_m_s for s in stats])),
        "mean_reward": float(np.mean([s.reward for s in stats])),
    }
