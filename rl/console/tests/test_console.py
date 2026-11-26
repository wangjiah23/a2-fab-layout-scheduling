import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import connection
import results
import worker


def isolated_worker(home):
    # Use the real manager and process groups, with a disposable payload and host resources.
    worker.ROOT = home
    worker.JOBS = home / "jobs"
    worker.CODE = home / "code"
    worker.READY = home / "deployed"
    worker.ACCOUNT_LOCK = Path(os.environ["A2_TEST_ACCOUNT"]) / "active.lock"
    worker.__file__ = __file__
    worker.resources = lambda: dict(ram_free=8192, disk_free=16000, cpus=4, gpus=[], warning="")
    if not Path("/proc/sys/kernel/random/boot_id").exists():

        def process_stamp(pid):
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "state=,lstart="], capture_output=True, text=True
            )
            fields = result.stdout.split()
            return " ".join(fields[1:]) if fields and "Z" not in fields[0] else None

        worker.stamp = process_stamp

    def steps():
        yield "fixture", [sys.executable, "-c", "import time; time.sleep(.15)"]

    worker.setup_steps = steps
    payload = """
import subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(sys.argv[1]).write_text(str(child.pid))
child.wait()
"""
    worker.model_command = lambda *_: [sys.executable, "-c", payload, str(home / "child.pid")]
    if (home / "launcher-exits").exists():
        worker.model_command = lambda *_: [
            sys.executable,
            "-c",
            payload.replace("child.wait()", ""),
            str(home / "detached-child.pid"),
        ]
    if (home / "over-budget").exists():
        worker.group_usage = lambda *_: (2048, 0)


class ConsoleTests(unittest.TestCase):
    def test_owned_job_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            env = {**os.environ, "A2_TEST_ACCOUNT": str(home / "account-locks")}

            def rpc(action, root=home, **fields):
                reply = subprocess.check_output(
                    [sys.executable, __file__, "--rpc"],
                    input=json.dumps(dict(action=action, **fields)),
                    text=True,
                    env={**env, "A2_TEST_WORKER_HOME": str(root)},
                )
                return json.loads(reply)

            def wait(predicate):
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    status = rpc("status")
                    if predicate(status["state"]):
                        return status
                    time.sleep(0.15)
                self.fail(f"Timed out: {status['state']}")

            setup = rpc("start", kind="setup")["id"]
            self.assertTrue(wait(lambda s: s["status"] == "completed")["ready"])
            params = dict(kind="train", device="cpu", ram_gib=1, vram_gib=0, cycles=1, batch=2, workers=1)
            job = rpc("start", **params)["id"]
            child = wait(lambda s: "rss" in s and (home / "child.pid").is_file())["state"]["child"][0]
            self.assertEqual(os.getpgid(int((home / "child.pid").read_text())), child)
            self.assertIn("error", rpc("start", **params))
            other_root = home / "另一个部署目录 with spaces"
            self.assertIn("error", rpc("start", root=other_root, kind="setup"))
            self.assertEqual(rpc("status", root=other_root)["jobs"], [])
            self.assertIn("error", rpc("stop", job=setup))
            other = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
            )
            try:
                self.assertEqual(rpc("stop", job=job)["id"], job)
                wait(lambda s: s["status"] == "stopped")
                self.assertFalse(worker.group_alive(child))
                self.assertIsNone(other.poll())
                self.assertIn("error", rpc("result", job=job))
                (home / "over-budget").touch()
                rpc("start", **params)
                state = wait(lambda s: s["status"] == "failed")["state"]
                self.assertIn("超过任务资源预算", state["detail"])
                (home / "over-budget").unlink()
                (home / "launcher-exits").touch()
                rpc("start", **params)
                wait(lambda s: s["status"] == "completed")
                pid = int((home / "detached-child.pid").read_text())
                status = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
                self.assertTrue(not status.stdout.strip() or status.stdout.strip().startswith("Z"))
            finally:
                other.terminate()
                other.wait()

    def test_gpu_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = dict(index=2, uuid="GPU-test", free=10000, total=24000)
            host = dict(gpus=[gpu], ram_free=4096, cpus=4)
            req = dict(kind="train", device=gpu["uuid"], ram_gib=2, vram_gib=6, workers=2, cycles=1, batch=8)
            with (
                patch.object(worker, "READY", Path(tmp) / "ready"),
                patch.object(worker, "resources", return_value=host),
            ):
                worker.READY.touch()
                plan = worker.validate(req)
                self.assertEqual(plan["vram_mib"], 6144)
                self.assertAlmostEqual(plan["allocator_fraction"], 6144 / 24000)
                for change in ({"ram_gib": 5}, {"vram_gib": 10}, {"device": "unknown"}, {"workers": 5}):
                    with self.assertRaises(ValueError):
                        worker.validate({**req, **change})

    def test_unknown_host_requires_matching_fingerprint(self):
        key = Mock()
        key.asbytes.return_value = b"fixture key"
        for pin in ("", "SHA256:different"):
            with self.assertRaises(ValueError):
                connection.VerifiedHost(pin).missing_host_key(None, "localhost", key)
        connection.VerifiedHost(connection.fingerprint(key)).missing_host_key(None, "localhost", key)

    def test_form_integers(self):
        self.assertEqual(connection.address("127.0.0.1", " 2202 "), ("127.0.0.1", 2202))
        self.assertEqual(connection.whole_number("0", "内存预算"), 0)
        for bad in ("", "-1", "1.8", "0xff", True):
            with self.assertRaises(ValueError):
                connection.whole_number(bad, "训练轮数")

    def test_remote_root_uses_server_home_and_keeps_literal_paths(self):
        home = "/home/fab"
        for raw, expected in [
            ("", "/home/fab/.local/share/a2-fab-rl"),
            ("~/PPO 试验", "/home/fab/PPO 试验"),
            ("$HOME/projects", "/home/fab/projects"),
            ("course/A2", "/home/fab/course/A2"),
            ("/mnt/data/team's run", "/mnt/data/team's run"),
        ]:
            self.assertEqual(str(connection.deployment_root(raw, home)), expected)
        for invalid in ("/", "~other/private", "/a\nb"):
            with self.assertRaises(ValueError):
                connection.deployment_root(invalid, home)

    def test_uploaded_helper_and_models_follow_selected_root(self):
        import runpy

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "自选目录"
            helpers = root / "helpers"
            helpers.mkdir(parents=True)
            script = helpers / "worker-test.py"
            script.write_bytes(Path(worker.__file__).read_bytes())
            loaded = runpy.run_path(str(script))
            self.assertEqual(loaded["ROOT"], root)
            (root / "models").mkdir()
            policy = root / "models" / "cycle120.pt"
            policy.touch()
            self.assertEqual(loaded["checkpoint_path"]("models/cycle120.pt"), policy)
            with self.assertRaises(ValueError):
                loaded["checkpoint_path"]("../models/cycle120.pt")

    def test_counts_only_owned_processes_on_selected_gpu(self):
        processes = "20 20 2048\n21 20 4096\n22 22 9216"
        gpu_rows = [
            ["20", "GPU-A", "100"],
            ["21", "GPU-A", "25"],
            ["21", "GPU-B", "7"],
            ["22", "GPU-A", "1100"],
        ]
        with (
            patch.object(worker, "capture", return_value=processes),
            patch.object(worker, "gpu_query", return_value=gpu_rows),
        ):
            self.assertEqual(worker.group_usage(20, "GPU-A"), (6, 125))

    def test_completed_job_survives_concurrent_status_read(self):
        finished = {"status": "completed"}
        with (
            patch.object(worker, "read", side_effect=[{"status": "running"}, [123, "old"], finished]),
            patch.object(worker, "active_identity", return_value=False),
        ):
            self.assertEqual(worker.reconcile(Path("job")), finished)

    def test_result_contract(self):
        meta = json.loads(results.META.read_text())
        device = list(range(len(meta["equipment"])))
        layout = dict(
            device_at_slot=device,
            cost=meta["reference_cost"],
            global_relative_gap_bound=0,
            primal_infeasibility=0,
            status="optimal_complete_path_LP",
        )

        def package(**overrides):
            return results.result_package(dict(layout={**layout, **overrides}, elapsed=5))

        self.assertEqual(len(package()["layout"]), 1068)
        self.assertEqual(package()["metrics"]["moved"], 0)
        for change in (
            {"device_at_slot": [0] * len(device)},
            {"global_relative_gap_bound": 0.1},
            {"status": "failed"},
        ):
            with self.assertRaises(ValueError):
                package(**change)


if __name__ == "__main__" and os.environ.get("A2_TEST_WORKER_HOME") and len(sys.argv) > 1:
    isolated_worker(Path(os.environ["A2_TEST_WORKER_HOME"]))
    if sys.argv[1] == "--work":
        worker.execute(worker.job_dir(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "--rpc":
        try:
            print(json.dumps(worker.handle(json.load(sys.stdin))))
        except ValueError as error:
            print(json.dumps(dict(error=str(error))))
elif __name__ == "__main__":
    unittest.main()
