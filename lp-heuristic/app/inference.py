# -*- coding: utf-8 -*-
"""推理（训练）模块：数据准备、E1→E3 调度、实时进度、结果保存。"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .rails import generate_rails

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_ROOT = PROJECT_ROOT / "data"
RESULT_ROOT = PROJECT_ROOT / "result"
SMAT_FILE = (DATA_ROOT / "SMAT2022" / "General Data" / "dataset 4"
             / "SMAT_2022_Model_Data_-_LVHM_E.xlsx")

# 结果文件清单：E1/E3 产物 -> result 目录内的文件名
_RESULT_FILES = {
    "layout.csv": "e3/layout.csv",
    "transport_flow.csv": "e1/transport_flow.csv",
    "device_utilization.csv": "e1/device_utilization.csv",
    "from_to_bay.csv": "e1/from_to_bay.csv",
}

_EPOCH_RE = re.compile(r"epoch\s+(\d+)\s+完成")


def prepare_data_root(run_dir: Path, order_file: Path) -> Path:
    """复制内置 LVHM 生产数据到临时目录，并用上传的订单覆盖 order.txt。"""
    tmp = run_dir / "tmp_data"
    src_lvhm = DATA_ROOT / "SMT2020_LVHM"
    dst_lvhm = tmp / "SMT2020_LVHM"
    shutil.copytree(src_lvhm, dst_lvhm)
    shutil.copyfile(order_file, dst_lvhm / "order.txt")
    return tmp


def run_stage(argv: list[str], on_line: Callable[[str], None]) -> int:
    """用 subprocess 运行一个脚本阶段，逐行回调输出，返回退出码。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, *argv],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.strip():
            on_line(line)
    proc.wait()
    return proc.returncode


class InferenceWorker(threading.Thread):
    """后台训练线程：E1（参考流量 LP）→ E3（SA+LNS+warm-start LP 启发式）。

    通过 ``self.queue`` 向主线程发消息：
        ("progress", stage, pct, line)  实时进度（stage: "E1"/"E3"/"save"）
        ("done", result_dir, summary)   训练完成
        ("error", message)              训练失败
    """

    def __init__(self, order_file: str, max_epochs: int = 100,
                 iter_per_epoch: int = 200, lp_every: int = 10,
                 warm_start: bool = True):
        super().__init__(daemon=True)
        self.order_file = Path(order_file)
        self.max_epochs = max_epochs
        self.iter_per_epoch = iter_per_epoch
        self.lp_every = lp_every
        self.warm_start = warm_start
        self.queue: queue.Queue[tuple] = queue.Queue()

    # -- 工具 --------------------------------------------------------------
    def _emit(self, msg: tuple) -> None:
        self.queue.put(msg)

    def _summary(self, path: Path) -> dict[str, Any]:
        import json
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # -- 主流程 ------------------------------------------------------------
    def run(self) -> None:
        try:
            self._run_training()
        except Exception as exc:  # noqa: BLE001
            self._emit(("error", f"{type(exc).__name__}: {exc}"))

    def _run_training(self) -> None:
        if not self.order_file.exists():
            self._emit(("error", f"订单文件不存在：{self.order_file}"))
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULT_ROOT / stamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1) 数据准备
        self._emit(("progress", "prepare", 0, "准备数据：复制内置生产数据并替换订单..."))
        data_root = prepare_data_root(run_dir, self.order_file)
        smat = str(SMAT_FILE)

        # 2) E1：参考流量 LP
        e1_out = run_dir / "e1"
        e1_argv = [
            str(SCRIPTS_DIR / "e1_reference_lp.py"),
            "--data-root", str(data_root),
            "--smat-file", smat,
            "--production-model", "LVHM",
            "--output-dir", str(e1_out),
        ]
        self._emit(("progress", "E1", 1, "阶段 E1：求解参考布局流量 LP ..."))
        last_line = ""

        def _on_e1(line: str) -> None:
            nonlocal last_line
            last_line = line
            self._emit(("progress", "E1", 5, line))

        rc = run_stage(e1_argv, _on_e1)
        if rc != 0:
            self._emit(("error", f"E1 阶段失败（退出码 {rc}）：{last_line}"))
            return

        # 3) E3：全局布局-流量联合启发式
        e3_out = run_dir / "e3"
        e3_argv = [
            str(SCRIPTS_DIR / "e3_heuristic.py"),
            "--data-root", str(data_root),
            "--smat-file", smat,
            "--production-model", "LVHM",
            "--e1-dir", str(e1_out),
            "--output-dir", str(e3_out),
            "--max-epochs", str(self.max_epochs),
            "--iter-per-epoch", str(self.iter_per_epoch),
            "--lp-every", str(self.lp_every),
        ]
        if self.warm_start:
            e3_argv.append("--warm-start")
        else:
            e3_argv.append("--no-warm-start")

        self._emit(("progress", "E3", 20, "阶段 E3：布局-流量联合启发式（SA + LNS + LP）..."))

        def _on_e3(line: str) -> None:
            self._emit(("progress", "E3", None, line))
            m = _EPOCH_RE.search(line)
            if m and self.max_epochs > 0:
                epoch = int(m.group(1))
                pct = 20 + 78 * (epoch / self.max_epochs)
                self._emit(("progress", "E3", pct, line))

        rc = run_stage(e3_argv, _on_e3)
        if rc != 0:
            self._emit(("error", f"E3 阶段失败（退出码 {rc}）"))
            return

        # 4) 汇总结果到 result/<run>/
        self._emit(("progress", "save", 99, "整理并保存结果..."))
        for out_name, rel in _RESULT_FILES.items():
            src = run_dir / rel
            if src.exists():
                shutil.copyfile(src, run_dir / out_name)

        # 5) 生成走廊（轨道）几何与物流强度数据 rails.csv
        self._emit(("progress", "save", 100, "生成走廊通道数据..."))
        try:
            n_rails = generate_rails(
                SMAT_FILE,
                e1_out / "selected_equipment_ports.csv",
                run_dir / "transport_flow.csv",
                run_dir / "rails.csv",
            )
            self._emit(("progress", "save", 100, f"走廊通道数据：{n_rails} 条轨道"))
        except Exception as exc:  # noqa: BLE001
            self._emit(("progress", "save", 100, f"走廊通道数据生成失败：{exc}"))

        e1_summary = self._summary(e1_out / "e1_summary.json")
        e3_summary = self._summary(e3_out / "e3_summary.json")
        summary = {
            "run_id": stamp,
            "order_file": self.order_file.name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "reference_cost": e3_summary.get("reference_cost"),
            "optimized_cost": e3_summary.get("optimized_cost"),
            "cost_reduction_pct": e3_summary.get("cost_reduction_pct"),
            "moved": e3_summary.get("moved"),
            "equipment_count": e3_summary.get("equipment_count"),
            "e1": {k: e1_summary.get(k) for k in
                   ("objective_value", "equipment_count", "operation_count", "solver_status")},
            "e3": {k: e3_summary.get(k) for k in ("lp_count", "elapsed_sec")},
        }
        import json
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        self._emit(("progress", "save", 100, "完成。"))
        self._emit(("done", str(run_dir), summary))
