from __future__ import annotations

import unittest

from tactigrip.policies import HeuristicGripController, run_episode
from tactigrip.sim.gripper import FragileGraspSim


class FragileGraspSimTest(unittest.TestCase):
    def test_contact_force_increases_when_jaws_close(self) -> None:
        sim = FragileGraspSim()
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
        sim = FragileGraspSim()
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

    def test_heuristic_can_lift_fragile_object(self) -> None:
        sim = FragileGraspSim()
        controller = HeuristicGripController()
        stats = run_episode(sim, controller, seed=3, object_name="fragile_foam")

        self.assertTrue(stats.success)
        self.assertFalse(stats.dropped)
        self.assertFalse(stats.crushed)


if __name__ == "__main__":
    unittest.main()
