# -*- coding: utf-8 -*-
"""生成走廊（轨道）几何与物流强度数据。

从 SMAT2022 读取轨道端点坐标，结合设备端口到轨道的映射与设备间物流，
输出 ``rails.csv``（每条轨道的线段几何 + 物流强度），供展示页绘制通道热力图。
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = PROJECT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from e0_data_audit import load_smat, numeric  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def generate_rails(smat_file: Path, selected_ports_csv: Path,
                   flow_csv: Path, out_csv: Path) -> int:
    """生成 rails.csv，返回写入的轨道边数量。"""
    smat = load_smat(smat_file)

    # 轨道节点坐标
    addr: dict[str, tuple[float, float]] = {}
    for row in smat["Address"]:
        name = row.get("NAME")
        x, y = numeric(row.get("POSITION_X")), numeric(row.get("POSITION_Y"))
        if name and x is not None and y is not None:
            addr[str(name)] = (x, y)

    # 轨道边几何
    geom: dict[str, tuple[float, float, float, float]] = {}
    for row in smat["Rail"]:
        name = row.get("NAME")
        src, tgt = row.get("FROM_NODE"), row.get("TO_NODE")
        if name and src in addr and tgt in addr:
            geom[str(name)] = (*addr[str(src)], *addr[str(tgt)])

    # 设备 -> 端口所在轨道
    equip_rail: dict[str, str] = {}
    for row in _read_csv(selected_ports_csv):
        eq = row.get("equipment_name", "")
        rail = row.get("rail_name", "")
        if eq and rail:
            equip_rail[eq] = rail

    # 设备间物流 -> 轨道物流强度（源/目标端口所在轨道各累加一次）
    rail_flow: dict[str, float] = defaultdict(float)
    for row in _read_csv(flow_csv):
        s, t = row.get("source_equipment", ""), row.get("target_equipment", "")
        f = numeric(row.get("flow_lots")) or 0.0
        rs, rt = equip_rail.get(s), equip_rail.get(t)
        if rs in geom:
            rail_flow[rs] += f
        if rt in geom:
            rail_flow[rt] += f

    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rail_name", "x1", "y1", "x2", "y2", "flow"])
        for name, (x1, y1, x2, y2) in geom.items():
            writer.writerow([name, x1, y1, x2, y2, rail_flow.get(name, 0.0)])

    return len(geom)


if __name__ == "__main__":
    ROOT = PROJECT_ROOT
    smat = ROOT / "data" / "SMAT2022" / "General Data" / "dataset 4" / "SMAT_2022_Model_Data_-_LVHM_E.xlsx"
    ports = ROOT / "output" / "e1_reference_lp" / "selected_equipment_ports.csv"
    flows = ROOT / "output" / "e1_reference_lp" / "transport_flow.csv"
    out = ROOT / "result" / "heuristic" / "rails.csv"
    n = generate_rails(smat, ports, flows, out)
    print("生成 rails.csv：{} 条轨道边 -> {}".format(n, out))
