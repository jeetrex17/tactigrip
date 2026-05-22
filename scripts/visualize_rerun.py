#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from tactigrip.envs.fragile_grasp_env import MODALITIES, FragileGraspEnv
from tactigrip.policies import HeuristicGripController
from tactigrip.sim.gripper import FragileGraspSim, ObjectProfile, SimConfig, StepResult


COLORS = {
    "foam": [90, 180, 255, 255],
    "plastic": [70, 210, 180, 255],
    "metal": [180, 190, 205, 255],
    "jaw": [235, 165, 55, 255],
    "pad": [35, 35, 42, 255],
    "palm": [105, 115, 125, 255],
    "rail": [155, 165, 175, 255],
    "force": [90, 220, 120, 255],
    "sensor_force": [90, 220, 120, 255],
    "sensor_mic": [230, 90, 210, 255],
    "sensor_imu": [80, 170, 255, 255],
    "sensor_temp": [255, 180, 70, 255],
    "target": [90, 220, 120, 120],
    "warning": [245, 90, 70, 255],
    "contact": [255, 210, 65, 230],
    "path": [120, 170, 255, 255],
    "table": [70, 72, 78, 190],
    "floor": [80, 80, 80, 130],
}

VISUAL_Z_SCALE = 0.25


def make_dashboard_blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                contents="/world/**",
                name="Gripper Lab",
                background=[16, 18, 22],
                line_grid=rrb.LineGrid3D(visible=True, spacing=0.025, color=[90, 95, 105, 90]),
                eye_controls=rrb.EyeControls3D(
                    position=(0.30, -0.42, 0.28),
                    look_target=(0.0, 0.0, 0.13),
                    eye_up=(0.0, 0.0, 1.0),
                ),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/signals",
                    contents=["/signals/normal_force_n", "/signals/shear_force_n"],
                    name="Grip Force",
                    axis_y=rrb.ScalarAxis(range=(0.0, 8.0)),
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                rrb.TimeSeriesView(
                    origin="/signals",
                    contents=["/signals/slip_velocity_m_s", "/signals/acoustic_energy"],
                    name="Slip Cues",
                    axis_y=rrb.ScalarAxis(range=(0.0, 0.12)),
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                rrb.TimeSeriesView(
                    origin="/signals",
                    contents=["/signals/object_height_m", "/signals/lift_target_m"],
                    name="Lift Tracking",
                    axis_y=rrb.ScalarAxis(range=(0.0, 1.0)),
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                rrb.TimeSeriesView(
                    origin="/signals",
                    contents=["/signals/crush_margin_n", "/signals/friction_margin_n"],
                    name="Safety Margins",
                    axis_y=rrb.ScalarAxis(range=(-1.5, 8.0)),
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                row_shares=[0.25, 0.25, 0.25, 0.25],
                name="Telemetry",
            ),
            column_shares=[0.72, 0.28],
            name="TactiGrip Dashboard",
        ),
        rrb.TimePanel(expanded=True, timeline="sim_time"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.BlueprintPanel(state="collapsed"),
        collapse_panels=True,
    )


def log_signal_styles() -> None:
    styles = {
        "normal_force_n": ([90, 220, 120], "normal force"),
        "shear_force_n": ([120, 170, 255], "shear force"),
        "acoustic_energy": ([230, 90, 210], "acoustic slip"),
        "accel_x_m_s2": ([80, 170, 255], "fingertip accel x"),
        "slip_velocity_m_s": ([245, 90, 70], "slip velocity"),
        "object_height_m": ([90, 180, 255], "object height"),
        "lift_target_m": ([245, 210, 80], "lift target"),
        "jaw_gap_m": ([235, 165, 55], "jaw gap"),
        "crush_margin_n": ([90, 220, 120], "crush margin"),
        "friction_margin_n": ([245, 170, 70], "friction margin"),
        "disturbance_active": ([245, 90, 70], "friction drop"),
    }
    for entity, (color, name) in styles.items():
        rr.log(f"signals/{entity}", rr.SeriesLines(colors=[color], names=[name]), static=True)


def log_box(path: str, center, size, color) -> None:
    rr.log(
        path,
        rr.Boxes3D(
            centers=[center],
            half_sizes=[[0.5 * size[0], 0.5 * size[1], 0.5 * size[2]]],
            colors=[color],
        ),
    )


def log_sensor_point(
    path: str,
    position,
    color,
    label: str,
    show_labels: bool,
    radius: float = 0.005,
) -> None:
    rr.log(
        path,
        rr.Points3D(
            [position],
            colors=[color],
            labels=[label] if show_labels else None,
            show_labels=show_labels,
            radii=[radius],
        ),
    )


def object_color(profile: ObjectProfile) -> list[int]:
    if "foam" in profile.name:
        return COLORS["foam"]
    if "plastic" in profile.name:
        return COLORS["plastic"]
    if "metal" in profile.name:
        return COLORS["metal"]
    return COLORS["foam"]


def disturbance_active(config: SimConfig, time_s: float) -> bool:
    if config.disturbance_start_s is None:
        return False
    return config.disturbance_start_s <= time_s <= config.disturbance_start_s + config.disturbance_duration_s


def log_gripper(
    result: StepResult,
    profile: ObjectProfile,
    config: SimConfig,
    object_z: float,
    visual_lift_height: float,
    show_sensors: bool,
    show_forces: bool,
    show_disturbance: bool,
    show_labels: bool,
) -> None:
    state = result.state
    contact = result.contact
    jaw_gap = state.jaw_gap_m
    lift_z = visual_lift_height + 0.09
    jaw_thickness = 0.018
    jaw_depth = 0.115
    jaw_height = 0.18
    pad_size = [0.008, 0.080, 0.095]
    left_x = -0.5 * jaw_gap
    right_x = 0.5 * jaw_gap
    palm_y = -0.070

    log_box("world/gripper/palm", [0.0, palm_y, lift_z], [0.17, 0.026, 0.090], COLORS["palm"])
    log_box("world/gripper/rail", [0.0, palm_y + 0.008, lift_z + 0.078], [0.20, 0.012, 0.012], COLORS["rail"])
    log_box("world/gripper/left_jaw/body", [left_x, 0.0, lift_z], [jaw_thickness, jaw_depth, jaw_height], COLORS["jaw"])
    log_box("world/gripper/right_jaw/body", [right_x, 0.0, lift_z], [jaw_thickness, jaw_depth, jaw_height], COLORS["jaw"])
    log_box(
        "world/gripper/left_jaw/fingertip_pad",
        [left_x + 0.006, 0.0, object_z],
        pad_size,
        COLORS["pad"],
    )
    log_box(
        "world/gripper/right_jaw/fingertip_pad",
        [right_x - 0.006, 0.0, object_z],
        pad_size,
        COLORS["pad"],
    )

    if show_sensors:
        sensor_z = object_z + 0.038
        sensor_specs = [
            ("force", COLORS["sensor_force"], -0.030, "F"),
            ("mic", COLORS["sensor_mic"], -0.010, "mic"),
            ("imu", COLORS["sensor_imu"], 0.010, "imu"),
            ("temp", COLORS["sensor_temp"], 0.030, "temp"),
        ]
        for name, color, y, label in sensor_specs:
            log_sensor_point(
                f"world/gripper/left_jaw/sensors/{name}",
                [left_x + 0.014, y, sensor_z],
                color,
                label,
                show_labels,
                radius=0.004,
            )
            log_sensor_point(
                f"world/gripper/right_jaw/sensors/{name}",
                [right_x - 0.014, y, sensor_z],
                color,
                label,
                show_labels,
                radius=0.004,
            )

    if show_forces:
        force_len = min(0.055, 0.006 + 0.010 * contact.normal_force_n)
        arrow_color = COLORS["warning"] if contact.slip_velocity_m_s > 0.002 else COLORS["force"]
        rr.log(
            "world/gripper/normal_force",
            rr.Arrows3D(
                origins=[[left_x + 0.016, 0.045, object_z], [right_x - 0.016, 0.045, object_z]],
                vectors=[[force_len, 0.0, 0.0], [-force_len, 0.0, 0.0]],
                colors=[arrow_color, arrow_color],
                labels=["left force", "right force"] if show_labels else None,
                show_labels=show_labels,
                radii=[0.0025, 0.0025],
            ),
        )

    if contact.in_contact:
        slip_color = COLORS["warning"] if contact.slip_velocity_m_s > 0.002 else COLORS["contact"]
        log_box(
            "world/contact/left_patch",
            [-0.5 * profile.width_m, 0.0, object_z],
            [0.004, 0.050, 0.050],
            slip_color,
        )
        log_box(
            "world/contact/right_patch",
            [0.5 * profile.width_m, 0.0, object_z],
            [0.004, 0.050, 0.050],
            slip_color,
        )

    if show_forces and contact.slip_velocity_m_s > 0.002:
        slip_len = min(0.12, 3.0 * contact.slip_velocity_m_s)
        rr.log(
            "world/contact/slip_arrow",
            rr.Arrows3D(
                origins=[[0.0, -0.060, object_z + 0.045]],
                vectors=[[0.0, 0.0, -slip_len]],
                colors=[COLORS["warning"]],
                labels=["slip"] if show_labels else None,
                show_labels=show_labels,
                radii=[0.003],
            ),
        )

    if show_disturbance:
        beacon_color = COLORS["warning"] if disturbance_active(config, state.time_s) else [90, 95, 105, 150]
        log_box("world/disturbance_beacon", [0.115, -0.085, 0.080], [0.015, 0.015, 0.160], beacon_color)


def log_scene(
    result: StepResult,
    profile: ObjectProfile,
    config: SimConfig,
    path_points: list[list[float]],
    show_sensors: bool,
    show_forces: bool,
    show_disturbance: bool,
    show_guides: bool,
    show_labels: bool,
) -> None:
    state = result.state
    contact = result.contact
    tactile = result.tactile
    obj_width = profile.width_m
    obj_depth = 0.070 if profile.name != "cool_metal" else 0.058
    obj_height = 0.065
    visual_lift_height = state.lift_height_m * VISUAL_Z_SCALE
    object_z = max(0.035, state.object_height_m * VISUAL_Z_SCALE + 0.035)
    path_points.append([0.0, 0.0, object_z])

    rr.set_time("sim_time", duration=state.time_s)

    log_box("world/floor", [0.0, 0.0, -0.018], [0.46, 0.34, 0.018], COLORS["floor"])
    log_box("world/table", [0.0, 0.0, -0.006], [0.30, 0.20, 0.012], COLORS["table"])
    log_box("world/object", [0.0, 0.0, object_z], [obj_width, obj_depth, obj_height], object_color(profile))
    log_gripper(
        result,
        profile,
        config,
        object_z,
        visual_lift_height,
        show_sensors,
        show_forces,
        show_disturbance,
        show_labels,
    )

    if show_guides:
        rr.log(
            "world/lift_target",
            rr.LineStrips3D(
                [[[0.0, 0.07, 0.0], [0.0, 0.07, visual_lift_height]]],
                colors=[COLORS["target"]],
                radii=[0.003],
            ),
        )
        rr.log(
            "world/object_path",
            rr.LineStrips3D([path_points], colors=[COLORS["path"]], radii=[0.002]),
        )
        rr.log(
            "world/lift_guide",
            rr.LineStrips3D(
                [[[0.0, 0.085, 0.0], [0.0, 0.085, max(0.01, visual_lift_height)]]],
                colors=[COLORS["target"]],
                radii=[0.0015],
            ),
        )

    rr.log("signals/normal_force_n", rr.Scalars(tactile.normal_force_n))
    rr.log("signals/shear_force_n", rr.Scalars(tactile.shear_force_n))
    rr.log("signals/acoustic_energy", rr.Scalars(tactile.acoustic_energy))
    rr.log("signals/accel_x_m_s2", rr.Scalars(tactile.accel_x_m_s2))
    rr.log("signals/slip_velocity_m_s", rr.Scalars(contact.slip_velocity_m_s))
    rr.log("signals/object_height_m", rr.Scalars(state.object_height_m))
    rr.log("signals/lift_target_m", rr.Scalars(state.lift_height_m))
    rr.log("signals/jaw_gap_m", rr.Scalars(state.jaw_gap_m))
    rr.log("signals/crush_margin_n", rr.Scalars(profile.crush_force_n - contact.normal_force_n))
    rr.log("signals/friction_margin_n", rr.Scalars(contact.available_friction_n - contact.required_friction_n))
    rr.log("signals/disturbance_active", rr.Scalars(1.0 if disturbance_active(config, state.time_s) else 0.0))


def run_policy(args) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install RL dependencies first: uv pip install -r requirements.txt") from exc

    model = PPO.load(args.policy)
    env = FragileGraspEnv(
        modalities=args.modalities,
        object_name=args.object,
        lift_start_s=args.lift_start,
        max_time_s=args.max_time,
        disturbance_start_s=args.disturbance_start,
        disturbance_duration_s=args.disturbance_duration,
        disturbance_friction_scale=args.disturbance_friction_scale,
        disturbance_slip_penalty_scale=args.disturbance_slip_penalty,
        seed=args.seed,
    )
    obs, _ = env.reset(seed=args.seed, options={"object_name": args.object})
    path_points: list[list[float]] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        log_scene(
            env.last_result,
            env.sim.object_profile,
            env.sim.config,
            path_points,
            args.show_sensors,
            args.show_forces,
            args.show_disturbance,
            args.show_guides,
            args.show_labels,
        )
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
    log_scene(
        env.last_result,
        env.sim.object_profile,
        env.sim.config,
        path_points,
        args.show_sensors,
        args.show_forces,
        args.show_disturbance,
        args.show_guides,
        args.show_labels,
    )


def run_heuristic(args) -> None:
    sim = FragileGraspSim(
        SimConfig(
            lift_start_s=args.lift_start,
            max_time_s=args.max_time,
            disturbance_start_s=args.disturbance_start,
            disturbance_duration_s=args.disturbance_duration,
            disturbance_friction_scale=args.disturbance_friction_scale,
            disturbance_slip_penalty_scale=args.disturbance_slip_penalty,
        )
    )
    controller = HeuristicGripController()
    result = sim.reset(seed=args.seed, object_name=args.object)
    controller.reset()
    path_points: list[list[float]] = []
    while not (result.terminated or result.truncated):
        log_scene(
            result,
            sim.object_profile,
            sim.config,
            path_points,
            args.show_sensors,
            args.show_forces,
            args.show_disturbance,
            args.show_guides,
            args.show_labels,
        )
        result = sim.step(controller.act(result, sim.object_profile.crush_force_n))
    log_scene(
        result,
        sim.object_profile,
        sim.config,
        path_points,
        args.show_sensors,
        args.show_forces,
        args.show_disturbance,
        args.show_guides,
        args.show_labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a tactile grasp episode in Rerun.")
    parser.add_argument("--policy", type=Path, default=Path("models/ppo_full.zip"))
    parser.add_argument("--modalities", choices=sorted(MODALITIES), default="full")
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--object", default="slippery_plastic")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--lift-start", type=float, default=2.0)
    parser.add_argument("--max-time", type=float, default=7.0)
    parser.add_argument("--disturbance-start", type=float, default=3.2)
    parser.add_argument("--disturbance-duration", type=float, default=1.5)
    parser.add_argument("--disturbance-friction-scale", type=float, default=0.45)
    parser.add_argument("--disturbance-slip-penalty", type=float, default=30.0)
    parser.add_argument("--show-sensors", action="store_true")
    parser.add_argument("--show-forces", action="store_true")
    parser.add_argument("--show-disturbance", action="store_true")
    parser.add_argument("--show-guides", action="store_true")
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    blueprint = None if args.no_dashboard else make_dashboard_blueprint()
    rr.init("tactigrip_gripper_demo", spawn=args.spawn, default_blueprint=blueprint)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save, default_blueprint=blueprint)
    if blueprint is not None:
        rr.send_blueprint(blueprint)
    log_signal_styles()
    rr.log(
        "world",
        rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        static=True,
    )

    if args.heuristic:
        run_heuristic(args)
    else:
        run_policy(args)

    if args.save is not None:
        print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
