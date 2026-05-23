from __future__ import annotations

from dataclasses import replace

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised when MuJoCo is absent.
    raise ImportError(
        "MuJoCo backend requires `mujoco`. Install with `uv pip install mujoco`."
    ) from exc

from tactigrip.sensors.tactile import TactileSensorModel
from tactigrip.sim.gripper import (
    ContactState,
    GripperState,
    ObjectProfile,
    SimConfig,
    StepResult,
    default_objects,
)


JAW_THICKNESS_M = 0.016


def _rgba(material: str) -> tuple[float, float, float, float]:
    if "foam" in material:
        return 0.35, 0.70, 1.00, 1.0
    if "plastic" in material:
        return 0.20, 0.85, 0.70, 1.0
    if "metal" in material:
        return 0.75, 0.78, 0.82, 1.0
    return 0.45, 0.70, 0.95, 1.0


def _mjcf(profile: ObjectProfile) -> str:
    color = _rgba(profile.name)
    half_width = 0.5 * profile.width_m
    half_depth = 0.035
    half_height = 0.032

    return f"""
<mujoco model="tactigrip">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <default>
    <geom contype="1" conaffinity="1" solref="0.006 1" solimp="0.90 0.95 0.001"
          friction="0.8 0.006 0.0001"/>
    <joint damping="1.5" armature="0.002"/>
  </default>

  <asset>
    <material name="floor_mat" rgba="0.12 0.13 0.15 1"/>
    <material name="rail_mat" rgba="0.45 0.50 0.55 1"/>
    <material name="jaw_mat" rgba="0.90 0.58 0.18 1"/>
    <material name="pad_mat" rgba="0.04 0.04 0.05 1"/>
    <material name="object_mat" rgba="{color[0]} {color[1]} {color[2]} {color[3]}"/>
  </asset>

  <worldbody>
    <light name="key" pos="0 -0.5 1.0" dir="0 0.5 -1"/>
    <camera name="demo" pos="0.25 -0.38 0.28" xyaxes="0.84 0.54 0 -0.24 0.38 0.89"/>
    <geom name="floor" type="plane" size="0.6 0.6 0.02" material="floor_mat"/>
    <geom name="table" type="box" pos="0 0 -0.006" size="0.16 0.12 0.006"
          material="floor_mat" contype="1" conaffinity="1"/>

    <body name="object" pos="0 0 {half_height + 0.001:.6f}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="{half_width:.6f} {half_depth:.6f} {half_height:.6f}"
            mass="{profile.mass_kg:.6f}" material="object_mat"
            friction="{profile.friction:.6f} 0.008 0.0001"/>
    </body>

    <body name="lift_stage" pos="0 0 0.090">
      <joint name="lift_z" type="slide" axis="0 0 1" range="0 0.55" damping="9.0"/>
      <geom name="palm" type="box" pos="0 -0.065 0" size="0.080 0.012 0.040"
            material="rail_mat" contype="0" conaffinity="0"/>
      <geom name="rail" type="box" pos="0 -0.055 0.068" size="0.105 0.007 0.007"
            material="rail_mat" contype="0" conaffinity="0"/>

      <body name="left_jaw">
        <joint name="left_slide" type="slide" axis="1 0 0" range="-0.060 -0.026" damping="5.0"/>
        <geom name="left_jaw_body" type="box" pos="0 0 0" size="0.008 0.050 0.080"
              material="jaw_mat" contype="0" conaffinity="0"/>
        <geom name="left_pad" type="box" pos="0.006 0 -0.025" size="0.004 0.040 0.045"
              material="pad_mat" friction="{profile.friction:.6f} 0.010 0.0001"/>
      </body>

      <body name="right_jaw">
        <joint name="right_slide" type="slide" axis="1 0 0" range="0.026 0.060" damping="5.0"/>
        <geom name="right_jaw_body" type="box" pos="0 0 0" size="0.008 0.050 0.080"
              material="jaw_mat" contype="0" conaffinity="0"/>
        <geom name="right_pad" type="box" pos="-0.006 0 -0.025" size="0.004 0.040 0.045"
              material="pad_mat" friction="{profile.friction:.6f} 0.010 0.0001"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="lift_position" joint="lift_z" kp="550" forcerange="-80 80" ctrlrange="0 0.55"/>
    <position name="left_position" joint="left_slide" kp="850" forcerange="-60 60"
              ctrlrange="-0.060 -0.026"/>
    <position name="right_position" joint="right_slide" kp="850" forcerange="-60 60"
              ctrlrange="0.026 0.060"/>
  </actuator>
</mujoco>
"""


class MuJoCoGraspSim:
    """MuJoCo-backed grasp-lift-hold simulator.

    This backend keeps the public task contract aligned with ``FragileGraspSim``
    but replaces the hand-written contact model with MuJoCo rigid-body contact.
    """

    def __init__(
        self,
        config: SimConfig | None = None,
        sensor_model: TactileSensorModel | None = None,
        substeps: int = 10,
    ) -> None:
        self.config = config or SimConfig(max_time_s=6.0, lift_start_s=0.9, lift_speed_m_s=0.15)
        self.sensor_model = sensor_model or TactileSensorModel()
        self.objects = default_objects()
        self.object_profile = self.objects["fragile_foam"]
        self.rng = np.random.default_rng()
        self.substeps = substeps
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.target_gap_m = self.config.max_gap_m
        self.target_lift_m = 0.0
        self.state = self._initial_state()
        self._ids: dict[str, int] = {}

    def reset(
        self,
        seed: int | None = None,
        object_name: str = "fragile_foam",
        object_profile: ObjectProfile | None = None,
    ) -> StepResult:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.object_profile = object_profile or self.objects[object_name]
        self.model = mujoco.MjModel.from_xml_string(_mjcf(self.object_profile))
        self.data = mujoco.MjData(self.model)
        self._cache_ids()
        self.sensor_model.reset(self.rng)
        self.target_gap_m = self.config.max_gap_m
        self.target_lift_m = 0.0
        self._write_initial_qpos()
        mujoco.mj_forward(self.model, self.data)

        self.state = self._read_state(crushed_time_s=0.0, hold_time_s=0.0, slip_distance_m=0.0)
        contact = self._read_contact(self.state)
        tactile = self.sensor_model.read(
            contact,
            self.object_profile.surface,
            self.config.dt,
            self.rng,
        )
        return StepResult(
            state=self.state,
            contact=contact,
            tactile=tactile,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"object": self.object_profile.name, "backend": "mujoco"},
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

    def step(self, close_command: float) -> StepResult:
        if self.model is None or self.data is None:
            raise RuntimeError("reset() must be called before step().")

        cfg = self.config
        close_command = float(np.clip(close_command, -1.0, 1.0))
        self.target_gap_m -= close_command * cfg.max_close_speed_m_s * cfg.dt
        self.target_gap_m = float(np.clip(self.target_gap_m, cfg.min_gap_m, cfg.max_gap_m))

        if self.data.time >= cfg.lift_start_s:
            self.target_lift_m += cfg.lift_speed_m_s * cfg.dt
            self.target_lift_m = min(self.target_lift_m, cfg.success_lift_m + 0.10)

        self._apply_controls()
        for _ in range(self.substeps):
            self._apply_disturbance_friction()
            mujoco.mj_step(self.model, self.data)

        previous = self.state
        state = self._read_state(
            crushed_time_s=previous.crushed_time_s,
            hold_time_s=previous.hold_time_s,
            slip_distance_m=previous.slip_distance_m,
        )
        contact = self._read_contact(state)

        crushed_time = state.crushed_time_s
        if contact.normal_force_n > self.object_profile.crush_force_n:
            crushed_time += cfg.dt
        else:
            crushed_time = max(0.0, crushed_time - 2.0 * cfg.dt)

        slip_distance = previous.slip_distance_m + contact.slip_velocity_m_s * cfg.dt
        lifted = state.object_height_m >= cfg.success_lift_m
        stable = contact.in_contact and contact.slip_velocity_m_s < 0.004
        hold_time = previous.hold_time_s + cfg.dt if lifted and stable else 0.0

        state = replace(
            state,
            crushed_time_s=crushed_time,
            hold_time_s=hold_time,
            slip_distance_m=slip_distance,
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
                "backend": "mujoco",
                "reason": reason,
                "success": reason == "success",
                "dropped": reason == "dropped",
                "crushed": reason == "crushed",
            },
        )

    def _cache_ids(self) -> None:
        assert self.model is not None
        names = [
            "lift_z",
            "left_slide",
            "right_slide",
            "object_free",
            "object",
            "object_geom",
            "left_pad",
            "right_pad",
        ]
        self._ids = {}
        for name in names:
            kind = mujoco.mjtObj.mjOBJ_BODY if name == "object" else mujoco.mjtObj.mjOBJ_GEOM
            if name.endswith("_slide") or name == "lift_z" or name == "object_free":
                kind = mujoco.mjtObj.mjOBJ_JOINT
            entity_id = mujoco.mj_name2id(self.model, kind, name)
            if entity_id < 0:
                raise RuntimeError(f"MuJoCo model is missing expected entity '{name}'")
            self._ids[name] = entity_id

    def _write_initial_qpos(self) -> None:
        assert self.model is not None and self.data is not None
        self.data.qpos[self._qadr("lift_z")] = 0.0
        self.data.qpos[self._qadr("left_slide")] = self._left_target(self.config.max_gap_m)
        self.data.qpos[self._qadr("right_slide")] = self._right_target(self.config.max_gap_m)

        object_qadr = self._qadr("object_free")
        self.data.qpos[object_qadr : object_qadr + 7] = np.array(
            [0.0, 0.0, 0.034, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.data.ctrl[:] = [
            0.0,
            self._left_target(self.config.max_gap_m),
            self._right_target(self.config.max_gap_m),
        ]

    def _apply_controls(self) -> None:
        assert self.data is not None
        self.data.ctrl[0] = self.target_lift_m
        self.data.ctrl[1] = self._left_target(self.target_gap_m)
        self.data.ctrl[2] = self._right_target(self.target_gap_m)

    def _apply_disturbance_friction(self) -> None:
        assert self.model is not None and self.data is not None
        friction = self.object_profile.friction
        if self._disturbance_active(self.data.time):
            friction *= self.config.disturbance_friction_scale
        for geom_name in ("object_geom", "left_pad", "right_pad"):
            self.model.geom_friction[self._ids[geom_name], 0] = friction

    def _read_state(
        self,
        crushed_time_s: float,
        hold_time_s: float,
        slip_distance_m: float,
    ) -> GripperState:
        assert self.model is not None and self.data is not None
        lift = float(self.data.qpos[self._qadr("lift_z")])
        left = float(self.data.qpos[self._qadr("left_slide")])
        right = float(self.data.qpos[self._qadr("right_slide")])
        left_v = float(self.data.qvel[self._vadr("left_slide")])
        right_v = float(self.data.qvel[self._vadr("right_slide")])

        object_body = self._ids["object"]
        object_z = float(self.data.xpos[object_body, 2])
        object_height = max(0.0, object_z - 0.034)
        jaw_gap = max(0.0, right - left - JAW_THICKNESS_M)
        jaw_velocity = -0.5 * (right_v - left_v)

        return GripperState(
            time_s=float(self.data.time),
            jaw_gap_m=float(jaw_gap),
            jaw_velocity_m_s=float(jaw_velocity),
            lift_height_m=lift,
            object_height_m=float(object_height),
            slip_distance_m=slip_distance_m,
            crushed_time_s=crushed_time_s,
            hold_time_s=hold_time_s,
        )

    def _read_contact(self, state: GripperState) -> ContactState:
        assert self.model is not None and self.data is not None
        object_geom = self._ids["object_geom"]
        pad_geoms = {self._ids["left_pad"], self._ids["right_pad"]}

        normal_force = 0.0
        shear_force = 0.0
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            if object_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == object_geom else contact.geom1
            if other not in pad_geoms:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, idx, force)
            normal_force += abs(float(force[0]))
            shear_force += float(np.linalg.norm(force[1:3]))

        in_contact = normal_force > 0.02
        object_vz = float(self.data.cvel[self._ids["object"], 5])
        lift_vz = float(self.data.qvel[self._vadr("lift_z")])
        slip_velocity = max(0.0, lift_vz - object_vz) if in_contact else 0.0
        required_friction = self.object_profile.mass_kg * self.config.gravity_m_s2
        available_friction = self._effective_friction(state.time_s) * normal_force
        compression = max(0.0, self.object_profile.width_m - state.jaw_gap_m)

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
        if state.lift_height_m > 0.04 and state.object_height_m < 0.015:
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
            reward += 0.2
            force_error = abs(contact.normal_force_n - target_force)
            reward += 0.10 * min(contact.normal_force_n / max(target_force, 1e-6), 1.0)
            reward -= 0.04 * force_error
            reward -= 0.015 * contact.normal_force_n
            reward -= 8.0 * contact.slip_velocity_m_s
        else:
            closing_range = max(1e-6, cfg.max_gap_m - self.object_profile.width_m)
            approach = 1.0 - np.clip(
                (state.jaw_gap_m - self.object_profile.width_m) / closing_range,
                0.0,
                1.0,
            )
            reward += 0.35 * approach
            reward -= 0.03
            if state.lift_height_m > 0.0:
                reward -= 0.75
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
        needed = obj.mass_kg * self.config.gravity_m_s2 / max(obj.friction, 1e-6)
        return float(min(0.80 * obj.crush_force_n, 1.25 * needed + 0.20))

    @property
    def target_normal_force_n(self) -> float:
        return self._target_normal_force_n()

    def _effective_friction(self, time_s: float) -> float:
        if self._disturbance_active(time_s):
            return self.object_profile.friction * self.config.disturbance_friction_scale
        return self.object_profile.friction

    def _disturbance_active(self, time_s: float) -> bool:
        cfg = self.config
        if cfg.disturbance_start_s is None:
            return False
        return (
            cfg.disturbance_start_s
            <= time_s
            <= cfg.disturbance_start_s + cfg.disturbance_duration_s
        )

    def _qadr(self, joint_name: str) -> int:
        assert self.model is not None
        return int(self.model.jnt_qposadr[self._ids[joint_name]])

    def _vadr(self, joint_name: str) -> int:
        assert self.model is not None
        return int(self.model.jnt_dofadr[self._ids[joint_name]])

    @staticmethod
    def _left_target(gap_m: float) -> float:
        return -0.5 * (gap_m + JAW_THICKNESS_M)

    @staticmethod
    def _right_target(gap_m: float) -> float:
        return 0.5 * (gap_m + JAW_THICKNESS_M)


def run_scripted_mujoco_episode(
    object_name: str = "slippery_plastic",
    seed: int = 5,
    disturbance: bool = False,
) -> tuple[StepResult, dict[str, float | bool | str]]:
    config = SimConfig(
        max_time_s=6.0,
        lift_start_s=0.9,
        lift_speed_m_s=0.15,
        success_lift_m=0.26,
        required_hold_s=0.35,
        disturbance_start_s=1.8 if disturbance else None,
        disturbance_duration_s=1.0 if disturbance else 0.0,
        disturbance_friction_scale=0.55 if disturbance else 1.0,
    )
    sim = MuJoCoGraspSim(config)
    result = sim.reset(seed=seed, object_name=object_name)
    forces: list[float] = []
    slip_steps = 0
    peak_slip = 0.0

    while not (result.terminated or result.truncated):
        target_force = sim._target_normal_force_n()
        if not result.contact.in_contact:
            command = 1.0
        else:
            force_error = target_force - result.contact.normal_force_n
            command = float(np.clip(0.20 * force_error, -1.0, 1.0))
        result = sim.step(command)
        forces.append(result.contact.normal_force_n)
        if result.contact.slip_velocity_m_s > 0.004:
            slip_steps += 1
        peak_slip = max(peak_slip, result.contact.slip_velocity_m_s)

    stats = {
        "success": bool(result.info["success"]),
        "dropped": bool(result.info["dropped"]),
        "crushed": bool(result.info["crushed"]),
        "reason": str(result.info["reason"]),
        "duration_s": result.state.time_s,
        "final_height_m": result.state.object_height_m,
        "max_force_n": float(np.max(forces)) if forces else 0.0,
        "mean_force_n": float(np.mean(forces)) if forces else 0.0,
        "slip_duration_s": slip_steps * config.dt,
        "peak_slip_m_s": peak_slip,
    }
    return result, stats
