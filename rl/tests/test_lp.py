"""Solve the published factory layout through the same path LP used by PPO."""

import unittest

import numpy as np

from smat_rl.lp import OriginalSlotLP, SlotPathWorkers


class TerminalLPTests(unittest.TestCase):
    def test_original_factory_cost_and_certificate(self):
        instance = OriginalSlotLP()
        identity = np.arange(1068, dtype=np.int32)
        workers = SlotPathWorkers(instance, instance.seed_flow, workers=1, cuda_device=None)
        self.addCleanup(workers.shutdown)

        result = workers.submit(identity).result()
        self.assertEqual(result["status"], "optimal_complete_path_LP", result.get("error"))
        self.assertEqual(result["represented_machines"], len(identity))
        self.assertEqual(result["all_operations"], 3866)
        np.testing.assert_array_equal(result["device_at_slot"], identity)
        self.assertEqual(result["changed_slots"], 0)
        self.assertAlmostEqual(instance.reference_cost, 5_803_123_608.498, delta=0.1)
        # Both the realized primal cost and its certified lower bound describe the
        # full transport problem. Compare them at the solver's certified precision.
        tolerance = instance.reference_cost * 1e-7
        self.assertAlmostEqual(result["cost"], instance.reference_cost, delta=tolerance)
        self.assertLessEqual(result["primal_infeasibility"], 1e-5)
        self.assertLessEqual(result["global_relative_gap_bound"], 1e-7)
        self.assertLessEqual(result["dual_lower_bound"], result["cost"] + 0.1)
        self.assertLessEqual(result["cost"] - result["dual_lower_bound"], tolerance)
