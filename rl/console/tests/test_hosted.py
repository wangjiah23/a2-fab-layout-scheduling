import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app import Job, select_gpu


class HostedTests(unittest.TestCase):
    def test_job_contains_only_production_inputs(self):
        demand = dict(part="part_1", route="r_1", demand_lots=4)
        job = dict(schema="fab-job-v1", method="rl", scenario=dict(demands=[demand]))
        self.assertEqual(Job.model_validate(job).scenario.demands[0].pieces_per_lot, 25)
        for key in ("host", "checkpoint", "gpu"):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                Job.model_validate({**job, key: "other"})
        for field in (dict(rails=[]), dict(initial_layout=[])):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                Job.model_validate({**job, "scenario": {**job["scenario"], **field}})

    @patch("app.subprocess.check_output")
    def test_choose_idle_gpu_with_most_headroom(self, query):
        query.side_effect = [
            "occupied\n",
            "occupied,24000,0\nbusy,23000,97\ntoo-small,500,0\nsmall,4096,0\nlarge,14500,3\n",
        ]
        self.assertEqual(select_gpu(), "large")
        query.side_effect = ["occupied\n", "occupied,23000,0\nbusy,19000,88\n"]
        self.assertEqual(select_gpu(), "")
