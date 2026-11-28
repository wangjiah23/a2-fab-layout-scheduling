"""Run scene changes through the public model inputs, including the group's layout.csv."""

import csv
import unittest
from copy import copy
from pathlib import Path

import numpy as np

from smat_rl.lp.instance import OriginalSlotLP
from smat_rl.scene import apply_scenario, initial_permutation


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = OriginalSlotLP()

    def setUp(self):
        self.lp = copy(self.base)
        self.scene = dict(
            horizon_min=10080,
            effective_time_fraction=0.85,
            new_routes=[],
            demands=[
                dict(part=p, route="r_" + p.removeprefix("part_"), demand_lots=d)
                for p, d in self.base.meta["demands"].items()
            ],
        )

    def test_reference_and_changed_mix(self):
        self.assertTrue(apply_scenario(self.lp, self.scene))
        self.scene["demands"][0]["demand_lots"] -= 5
        self.scene["demands"][4]["demand_lots"] += 5
        self.assertFalse(apply_scenario(self.lp, self.scene))
        for field in ("flow_weights", "static_graph"):
            self.assertFalse(np.array_equal(self.base.data[field], self.lp.data[field]))
        np.testing.assert_equal(len(self.lp.operations["group_ids"]), 3866)

    def test_new_route_replaces_built_in_edges_and_rejects_overload(self):
        group = self.base.meta["groups"]
        self.scene.update(
            new_routes=[
                dict(route="custom", step=2, step_type="delay", stnfam="", time_min_per_lot=30),
                dict(
                    route="custom", step=3, step_type="process", stnfam=group[1].lower(), time_min_per_lot=6
                ),
                dict(route="custom", step=1, step_type="process", stnfam=group[0], time_min_per_lot=4),
            ],
            demands=[dict(part="A", route="custom", demand_lots=8)],
        )
        self.assertFalse(apply_scenario(self.lp, self.scene))
        np.testing.assert_array_equal(self.lp.operations["group_ids"], [0, 1])
        np.testing.assert_array_equal(self.lp.operations["tau"], [4, 6])
        self.assertEqual(self.lp.data["flow_weights"].sum(), 8)
        self.assertEqual(self.lp.data["flow_weights"][0, 1], 8)
        self.scene["demands"][0]["demand_lots"] = 1e7
        with self.assertRaisesRegex(ValueError, "可用机时"):
            apply_scenario(copy(self.base), self.scene)

    def test_pieces_scale_only_piece_and_batch_operations(self):
        self.scene["demands"][0]["pieces_per_lot"] = 12
        apply_scenario(self.lp, self.scene)
        old = self.base.operations
        first = slice(0, old["product_offsets"][1])
        np.testing.assert_allclose(
            self.lp.operations["tau"][first],
            old["tau"][first] * np.where(old["piece_scaled"][first], 12 / 25, 1),
        )

    def test_team_layout_resolves_physical_slots(self):
        sample = (
            Path(__file__).resolve().parents[2] / "sample_app/周报演示应用/导入示例/03_启发式全厂布局.csv"
        )
        with sample.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ("x", "y", "width", "height"):
                row[key] = float(row[key])
        eq = self.base.meta["equipment"]
        layout = initial_permutation(rows, eq)
        np.testing.assert_array_equal(np.sort(layout), np.arange(1068))
        self.assertEqual(eq[layout[int(rows[0]["new_slot"])]]["name"], rows[0]["equipment_name"])
        rows[0]["x"] += 100
        with self.assertRaisesRegex(ValueError, "既有机位"):
            initial_permutation(rows, eq)
