from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised when RL deps are absent.
    raise ImportError(
        "FragileGraspEnv requires gymnasium. Install with `pip install -r requirements.txt`."
    ) from exc

from tactigrip.sim.gripper import FragileGraspSim


MODALITIES = {
    "force": ("normal_force_n", "shear_force_n"),
    "force_acoustic": ("normal_force_n", "shear_force_n", "acoustic_energy"),
    "force_accel": (
        "normal_force_n",
        "shear_force_n",
        "accel_x_m_s2",
        "accel_y_m_s2",
        "accel_z_m_s2",
    ),
    "force_temp": ("normal_force_n", "shear_force_n", "temperature_c"),
    "full": (
        "normal_force_n",
        "shear_force_n",
        "acoustic_energy",
        "accel_x_m_s2",
        "accel_y_m_s2",
        "accel_z_m_s2",
        "temperature_c",
    ),
}


@dataclass(frozen=True)
class ObservationScales:
    jaw_gap_m: float = 0.085
    jaw_velocity_m_s: float = 0.045
    lift_height_m: float = 0.45
    slip_distance_m: float = 0.03
    force_n: float = 10.0
    acoustic: float = 0.20
    accel_m_s2: float = 1.0
    temperature_c: float = 35.0


class FragileGraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        modalities: str = "full",
        object_name: str = "fragile_foam",
        randomize_object: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if modalities not in MODALITIES:
            raise ValueError(f"unknown modalities '{modalities}', expected one of {sorted(MODALITIES)}")

        self.modalities = modalities
        self.object_name = object_name
        self.randomize_object = randomize_object
        self.sim = FragileGraspSim()
        self.scales = ObservationScales()
        self._rng = np.random.default_rng(seed)
        self._last = self.sim.reset(seed=seed, object_name=object_name)

        obs_dim = len(self._observation(self._last))
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        object_name = self.object_name
        if self.randomize_object:
            object_name = str(self._rng.choice(list(self.sim.objects)))
        if options and "object_name" in options:
            object_name = str(options["object_name"])

        self._last = self.sim.reset(seed=seed, object_name=object_name)
        return self._observation(self._last), dict(self._last.info)

    def step(self, action):
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self._last = self.sim.step(action_value)
        return (
            self._observation(self._last),
            self._last.reward,
            self._last.terminated,
            self._last.truncated,
            dict(self._last.info),
        )

    def _observation(self, result) -> np.ndarray:
        state = result.state
        contact = result.contact
        tactile = result.tactile
        obj = self.sim.object_profile
        scales = self.scales

        values = [
            state.jaw_gap_m / scales.jaw_gap_m,
            state.jaw_velocity_m_s / scales.jaw_velocity_m_s,
            state.lift_height_m / scales.lift_height_m,
            state.object_height_m / scales.lift_height_m,
            state.slip_distance_m / scales.slip_distance_m,
            float(contact.in_contact),
            contact.available_friction_n / max(contact.required_friction_n, 1e-6),
            obj.crush_force_n / scales.force_n,
        ]

        for name in MODALITIES[self.modalities]:
            raw = getattr(tactile, name)
            values.append(self._scale_tactile(name, raw))

        return np.asarray(values, dtype=np.float32)

    def _scale_tactile(self, name: str, value: float) -> float:
        scales = self.scales
        if name in {"normal_force_n", "shear_force_n"}:
            return value / scales.force_n
        if name == "acoustic_energy":
            return value / scales.acoustic
        if name.startswith("accel_"):
            return value / scales.accel_m_s2
        if name == "temperature_c":
            return value / scales.temperature_c
        raise KeyError(name)

