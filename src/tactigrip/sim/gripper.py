from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from tactigrip.sensors.tactile import SurfaceProfile, TactileReading, TactileSensorModel


@dataclass(frozen=True)
class ObjectProfile:
    name: str
    width_m: float
    mass_kg: float
    friction: float
    crush_force_n: float
    crush_time_s: float
    stiffness_n_m: float
    surface: SurfaceProfile


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.01
    max_time_s: float = 5.0
    max_gap_m: float = 0.085
    min_gap_m: float = 0.01
    max_close_speed_m_s: float = 0.045
    jaw_response_hz: float = 18.0
    contact_damping_n_s_m: float = 25.0
    slip_damping_n_s_m: float = 45.0
    lift_start_s: float = 0.8
    lift_speed_m_s: float = 0.18
    success_lift_m: float = 0.45
    required_hold_s: float = 1.0
    drop_slip_m: float = 0.03
    gravity_m_s2: float = 9.81
    force_penalty_scale: float = 0.015
    slip_penalty_scale: float = 8.0
    disturbance_start_s: float | None = None
    disturbance_duration_s: float = 0.0
    disturbance_friction_scale: float = 1.0
    disturbance_slip_penalty_scale: float = 30.0


@dataclass
class GripperState:
    time_s: float
    jaw_gap_m: float
    jaw_velocity_m_s: float
    lift_height_m: float
    object_height_m: float
    slip_distance_m: float
    crushed_time_s: float
    hold_time_s: float


@dataclass(frozen=True)
class ContactState:
    in_contact: bool
    normal_force_n: float
    shear_force_n: float
    slip_velocity_m_s: float
    available_friction_n: float
    required_friction_n: float
    compression_m: float


@dataclass(frozen=True)
class StepResult:
    state: GripperState
    contact: ContactState
    tactile: TactileReading
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, float | bool | str]


def default_objects() -> dict[str, ObjectProfile]:
    return {
        "fragile_foam": ObjectProfile(
            name="fragile_foam",
            width_m=0.052,
            mass_kg=0.12,
            friction=0.75,
            crush_force_n=5.0,
            crush_time_s=0.10,
            stiffness_n_m=2600.0,
            surface=SurfaceProfile(
                material="foam",
                roughness=0.45,
                temperature_c=25.0,
                thermal_response=0.35,
            ),
        ),
        "slippery_plastic": ObjectProfile(
            name="slippery_plastic",
            width_m=0.050,
            mass_kg=0.16,
            friction=0.35,
            crush_force_n=8.0,
            crush_time_s=0.15,
            stiffness_n_m=4200.0,
            surface=SurfaceProfile(
                material="plastic",
                roughness=0.30,
                temperature_c=23.0,
                thermal_response=0.20,
            ),
        ),
        "cool_metal": ObjectProfile(
            name="cool_metal",
            width_m=0.048,
            mass_kg=0.24,
            friction=0.50,
            crush_force_n=18.0,
            crush_time_s=0.25,
            stiffness_n_m=9000.0,
            surface=SurfaceProfile(
                material="metal",
                roughness=0.20,
                temperature_c=18.0,
                thermal_response=0.75,
            ),
        ),
    }


class FragileGraspSim:
    """Small deterministic gripper/contact simulator with synthetic tactile readings.

    The model is intentionally compact: MuJoCo/Isaac can replace the dynamics later,
    while the task contract, sensor assumptions, and benchmark metrics stay stable.
    """

    def __init__(
        self,
        config: SimConfig | None = None,
        sensor_model: TactileSensorModel | None = None,
    ) -> None:
        self.config = config or SimConfig()
        self.sensor_model = sensor_model or TactileSensorModel()
        self.objects = default_objects()
        self.object_profile = self.objects["fragile_foam"]
        self.rng = np.random.default_rng()
        self.state = self._initial_state()

    def reset(
        self,
        seed: int | None = None,
        object_name: str = "fragile_foam",
        object_profile: ObjectProfile | None = None,
    ) -> StepResult:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.object_profile = object_profile or self.objects[object_name]
        self.sensor_model.reset(self.rng)
        self.state = self._initial_state()
        contact = self._compute_contact(self.state)
        tactile = self.sensor_model.read(contact, self.object_profile.surface, self.config.dt, self.rng)
        return StepResult(
            state=self.state,
            contact=contact,
            tactile=tactile,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"object": self.object_profile.name},
        )

    def step(self, action: float) -> StepResult:
        cfg = self.config
        action = float(np.clip(action, -1.0, 1.0))
        commanded_close_speed = action * cfg.max_close_speed_m_s

        jaw_velocity = self.state.jaw_velocity_m_s
        response = np.clip(cfg.jaw_response_hz * cfg.dt, 0.0, 1.0)
        jaw_velocity += response * (commanded_close_speed - jaw_velocity)

        jaw_gap = self.state.jaw_gap_m - jaw_velocity * cfg.dt
        jaw_gap = float(np.clip(jaw_gap, cfg.min_gap_m, cfg.max_gap_m))
        if jaw_gap in (cfg.min_gap_m, cfg.max_gap_m):
            jaw_velocity = 0.0

        lift_height = self.state.lift_height_m
        if self.state.time_s >= cfg.lift_start_s:
            lift_height += cfg.lift_speed_m_s * cfg.dt

        draft_state = replace(
            self.state,
            time_s=self.state.time_s + cfg.dt,
            jaw_gap_m=jaw_gap,
            jaw_velocity_m_s=jaw_velocity,
            lift_height_m=lift_height,
        )
        contact = self._compute_contact(draft_state)

        if contact.in_contact:
            slip_distance = self.state.slip_distance_m + contact.slip_velocity_m_s * cfg.dt
            object_height = max(0.0, lift_height - slip_distance)
        else:
            slip_distance = self.state.slip_distance_m
            object_height = 0.0

        crushed_time = self.state.crushed_time_s
        if contact.normal_force_n > self.object_profile.crush_force_n:
            crushed_time += cfg.dt
        else:
            crushed_time = max(0.0, crushed_time - 2.0 * cfg.dt)

        lifted = object_height >= cfg.success_lift_m
        stable = contact.slip_velocity_m_s < 0.002 and contact.in_contact
        hold_time = self.state.hold_time_s + cfg.dt if lifted and stable else 0.0

        state = replace(
            draft_state,
            object_height_m=object_height,
            slip_distance_m=slip_distance,
            crushed_time_s=crushed_time,
            hold_time_s=hold_time,
        )
        self.state = state

        tactile = self.sensor_model.read(contact, self.object_profile.surface, cfg.dt, self.rng)
        terminated, reason = self._termination_reason(state, contact)
        truncated = state.time_s >= cfg.max_time_s and not terminated
        reward = self._reward(state, contact, terminated, reason)

        return StepResult(
            state=state,
            contact=contact,
            tactile=tactile,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "object": self.object_profile.name,
                "reason": reason,
                "success": reason == "success",
                "dropped": reason == "dropped",
                "crushed": reason == "crushed",
            },
        )

    def _initial_state(self) -> GripperState:
        return GripperState(
            time_s=0.0,
            jaw_gap_m=self.config.max_gap_m,
            jaw_velocity_m_s=0.0,
            lift_height_m=0.0,
            object_height_m=0.0,
            slip_distance_m=0.0,
            crushed_time_s=0.0,
            hold_time_s=0.0,
        )

    def _compute_contact(self, state: GripperState) -> ContactState:
        obj = self.object_profile
        cfg = self.config
        friction = self._effective_friction(state.time_s)
        compression = max(0.0, obj.width_m - state.jaw_gap_m)
        in_contact = compression > 0.0

        closing_speed = max(0.0, state.jaw_velocity_m_s)
        normal_force = 0.0
        if in_contact:
            normal_force = obj.stiffness_n_m * compression
            normal_force += cfg.contact_damping_n_s_m * closing_speed

        required_friction = obj.mass_kg * cfg.gravity_m_s2
        available_friction = 2.0 * friction * normal_force
        slip_force = max(0.0, required_friction - available_friction)
        slip_velocity = slip_force / cfg.slip_damping_n_s_m if in_contact else 0.0
        shear_force = min(required_friction / 2.0, friction * normal_force)

        return ContactState(
            in_contact=in_contact,
            normal_force_n=float(normal_force),
            shear_force_n=float(shear_force),
            slip_velocity_m_s=float(slip_velocity),
            available_friction_n=float(available_friction),
            required_friction_n=float(required_friction),
            compression_m=float(compression),
        )

    def _termination_reason(self, state: GripperState, contact: ContactState) -> tuple[bool, str]:
        if state.crushed_time_s >= self.object_profile.crush_time_s:
            return True, "crushed"
        if state.lift_height_m > 0.05 and not contact.in_contact:
            return True, "dropped"
        if contact.in_contact and state.slip_distance_m >= self.config.drop_slip_m:
            return True, "dropped"
        if state.hold_time_s >= self.config.required_hold_s:
            return True, "success"
        return False, "running"

    def _reward(
        self,
        state: GripperState,
        contact: ContactState,
        terminated: bool,
        reason: str,
    ) -> float:
        cfg = self.config
        reward = 1.5 * state.object_height_m
        target_force = self._target_normal_force_n()

        if contact.in_contact:
            force_error = abs(contact.normal_force_n - target_force)
            reward += 0.20
            reward += 0.10 * min(contact.normal_force_n / max(target_force, 1e-6), 1.0)
            reward -= 0.05 * force_error
            safe_force = 0.80 * self.object_profile.crush_force_n
            excess_safe_force = max(0.0, contact.normal_force_n - safe_force)
            reward -= 0.35 * excess_safe_force * excess_safe_force
        else:
            closing_range = max(1e-6, cfg.max_gap_m - self.object_profile.width_m)
            approach = 1.0 - np.clip(
                (state.jaw_gap_m - self.object_profile.width_m) / closing_range,
                0.0,
                1.0,
            )
            reward += 0.15 * approach
            reward -= 0.03
            if state.lift_height_m > 0.0:
                reward -= 0.40

        reward -= cfg.force_penalty_scale * contact.normal_force_n
        reward -= cfg.slip_penalty_scale * contact.slip_velocity_m_s
        if self._disturbance_active(state.time_s) and contact.in_contact:
            required = max(contact.required_friction_n, 1e-6)
            friction_margin = (contact.available_friction_n - contact.required_friction_n) / required
            crush_headroom = np.clip(
                (self.object_profile.crush_force_n - contact.normal_force_n)
                / max(0.20 * self.object_profile.crush_force_n, 1e-6),
                0.0,
                1.0,
            )
            reward += 0.15 * float(crush_headroom * np.clip(friction_margin, -1.0, 1.0))
            reward -= cfg.disturbance_slip_penalty_scale * contact.slip_velocity_m_s

        if contact.normal_force_n > self.object_profile.crush_force_n:
            reward -= 1.0
        if terminated:
            if reason == "success":
                reward += 20.0
            elif reason == "dropped":
                reward -= 15.0
            elif reason == "crushed":
                reward -= 18.0
        return float(reward)

    def _target_normal_force_n(self) -> float:
        obj = self.object_profile
        needed = obj.mass_kg * self.config.gravity_m_s2 / max(2.0 * obj.friction, 1e-6)
        return float(min(0.8 * obj.crush_force_n, 1.25 * needed + 0.20))

    def _effective_friction(self, time_s: float) -> float:
        if self._disturbance_active(time_s):
            return self.object_profile.friction * self.config.disturbance_friction_scale
        return self.object_profile.friction

    def _disturbance_active(self, time_s: float) -> bool:
        cfg = self.config
        if cfg.disturbance_start_s is None:
            return False
        disturbance_end = cfg.disturbance_start_s + cfg.disturbance_duration_s
        return cfg.disturbance_start_s <= time_s <= disturbance_end
