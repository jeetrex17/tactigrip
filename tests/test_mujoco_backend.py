from __future__ import annotations

import unittest

try:
    import numpy as np

    from tactigrip.backends.mujoco_grasp import MuJoCoGraspSim, run_scripted_mujoco_episode
    from tactigrip.envs.mujoco_grasp_env import MuJoCoGraspEnv
except ImportError:  # pragma: no cover
    MuJoCoGraspSim = None
    MuJoCoGraspEnv = None
    run_scripted_mujoco_episode = None


@unittest.skipIf(MuJoCoGraspSim is None, "mujoco is not installed")
class MuJoCoGraspBackendTest(unittest.TestCase):
    def test_jaw_closure_produces_mujoco_contact_force(self) -> None:
        sim = MuJoCoGraspSim()
        result = sim.reset(seed=11, object_name="fragile_foam")
        max_force = 0.0

        for _ in range(160):
            result = sim.step(1.0)
            max_force = max(max_force, result.contact.normal_force_n)
            if result.contact.in_contact and result.contact.normal_force_n > 0.5:
                break

        self.assertTrue(result.contact.in_contact)
        self.assertGreater(max_force, 0.5)
        self.assertEqual(result.info["backend"], "mujoco")

    def test_scripted_mujoco_episode_lifts_object(self) -> None:
        _, stats = run_scripted_mujoco_episode(object_name="slippery_plastic", seed=5)

        self.assertTrue(stats["success"])
        self.assertFalse(stats["dropped"])
        self.assertGreater(stats["final_height_m"], 0.25)
        self.assertGreater(stats["max_force_n"], 1.0)

    def test_mujoco_gym_env_steps_with_force_target_action(self) -> None:
        env = MuJoCoGraspEnv(modalities="force_acoustic", object_name="fragile_foam")
        obs, info = env.reset(seed=12)
        self.assertEqual(info["backend"], "mujoco")
        self.assertEqual(obs.dtype, np.float32)

        for _ in range(60):
            obs, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
            if terminated or truncated:
                break

        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(obs.shape, env.observation_space.shape)


if __name__ == "__main__":
    unittest.main()
