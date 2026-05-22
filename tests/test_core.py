from __future__ import annotations

import unittest

import numpy as np

try:
    from tactigrip.envs.fragile_grasp_env import FragileGraspEnv
except ImportError:  # pragma: no cover
    FragileGraspEnv = None

from tactigrip.policies import HeuristicGripController, run_episode
from tactigrip.sim.gripper import ContactState, FragileGraspSim, GripperState, SimConfig


class FragileGraspSimTest(unittest.TestCase):
    def test_contact_force_increases_when_jaws_close(self) -> None:
        sim = FragileGraspSim(SimConfig(lift_start_s=3.0, max_time_s=7.0))
        result = sim.reset(seed=1, object_name="fragile_foam")

        max_force = 0.0
        contact_seen = False
        for _ in range(220):
            result = sim.step(0.5)
            max_force = max(max_force, result.contact.normal_force_n)
            contact_seen = contact_seen or result.contact.in_contact

        self.assertTrue(contact_seen)
        self.assertGreater(max_force, 1.0)

    def test_low_force_grasp_produces_slip_signal(self) -> None:
        sim = FragileGraspSim(SimConfig(lift_start_s=3.0, max_time_s=7.0))
        result = sim.reset(seed=2, object_name="fragile_foam")
        peak_acoustic = 0.0
        slip_seen = False

        for _ in range(500):
            error = 0.55 - result.tactile.normal_force_n
            action = 0.35 if error > 0.08 else -0.35 if error < -0.08 else 0.0
            result = sim.step(action)
            peak_acoustic = max(peak_acoustic, result.tactile.acoustic_energy)
            slip_seen = slip_seen or result.contact.slip_velocity_m_s > 0.002
            if result.terminated:
                break

        self.assertTrue(slip_seen)
        self.assertGreater(peak_acoustic, 0.02)

    def test_object_does_not_lift_without_contact(self) -> None:
        sim = FragileGraspSim()
        result = sim.reset(seed=4, object_name="fragile_foam")

        while not (result.terminated or result.truncated):
            result = sim.step(-1.0)

        self.assertTrue(result.info["dropped"])
        self.assertFalse(result.contact.in_contact)
        self.assertAlmostEqual(result.state.object_height_m, 0.0)

    def test_contact_reward_beats_no_contact_reward(self) -> None:
        sim = FragileGraspSim(SimConfig(lift_start_s=3.0, max_time_s=7.0))
        no_contact = sim.reset(seed=5, object_name="fragile_foam")

        result = no_contact
        for _ in range(260):
            result = sim.step(0.35)
            if result.contact.in_contact:
                break

        self.assertTrue(result.contact.in_contact)
        self.assertGreater(result.reward, no_contact.reward)

    def test_disturbance_reward_penalizes_slip_recovery(self) -> None:
        state = GripperState(
            time_s=0.75,
            jaw_gap_m=0.051,
            jaw_velocity_m_s=0.0,
            lift_height_m=0.2,
            object_height_m=0.1,
            slip_distance_m=0.0,
            crushed_time_s=0.0,
            hold_time_s=0.0,
        )
        contact = ContactState(
            in_contact=True,
            normal_force_n=0.4,
            shear_force_n=0.1,
            slip_velocity_m_s=0.02,
            available_friction_n=0.2,
            required_friction_n=1.0,
            compression_m=0.001,
        )
        base_cfg = dict(
            disturbance_start_s=0.5,
            disturbance_duration_s=1.0,
            disturbance_friction_scale=0.4,
        )
        low_penalty = FragileGraspSim(SimConfig(**base_cfg, disturbance_slip_penalty_scale=0.0))
        high_penalty = FragileGraspSim(SimConfig(**base_cfg, disturbance_slip_penalty_scale=50.0))

        low_reward = low_penalty._reward(state, contact, terminated=False, reason="running")
        high_reward = high_penalty._reward(state, contact, terminated=False, reason="running")

        self.assertLess(high_reward, low_reward - 0.5)

    def test_heuristic_can_lift_fragile_object(self) -> None:
        sim = FragileGraspSim()
        controller = HeuristicGripController()
        stats = run_episode(sim, controller, seed=3, object_name="fragile_foam")

        self.assertTrue(stats.success)
        self.assertFalse(stats.dropped)
        self.assertFalse(stats.crushed)

    @unittest.skipIf(FragileGraspEnv is None, "gymnasium is not installed")
    def test_env_action_commands_force_target(self) -> None:
        env = FragileGraspEnv(modalities="force", object_name="fragile_foam", lift_start_s=3.0)
        obs, _ = env.reset(seed=6)

        for _ in range(180):
            obs, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
            if terminated or truncated:
                break

        self.assertTrue(env.last_result.contact.in_contact)
        self.assertGreater(env.last_result.contact.normal_force_n, 0.5)
        self.assertEqual(obs.dtype, np.float32)

    @unittest.skipIf(FragileGraspEnv is None, "gymnasium is not installed")
    def test_observation_does_not_leak_contact_state(self) -> None:
        env = FragileGraspEnv(modalities="force", object_name="fragile_foam")
        obs, _ = env.reset(seed=7)

        proprio_count = 4
        force_count = 1
        self.assertEqual(obs.shape[0], proprio_count + force_count)


if __name__ == "__main__":
    unittest.main()
