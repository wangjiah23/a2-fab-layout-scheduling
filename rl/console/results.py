"""Convert a certified E3 inference to the existing fab-result-v1 workbench format."""

import json
import math
from pathlib import Path

META = Path(__file__).resolve().parents[1] / "data/original_slot_instance/instance.json"


def result_package(reply):
    meta = json.loads(META.read_text(encoding="utf-8"))
    result = reply["layout"]
    machines = meta["equipment"]
    permutation = result["device_at_slot"]
    if (
        len(permutation) != len(machines)
        or any(type(x) is not int for x in permutation)
        or sorted(permutation) != list(range(len(machines)))
    ):
        raise ValueError("推理结果中设备与槽位未形成完整的一一对应。")
    cost, gap, residual = (
        float(result[k]) for k in ("cost", "global_relative_gap_bound", "primal_infeasibility")
    )
    if (
        result.get("status") != "optimal_complete_path_LP"
        or not all(map(math.isfinite, (cost, gap, residual)))
        or cost < 0
        or not 0 <= gap <= 1e-7
        or not 0 <= residual <= 1e-5
    ):
        raise ValueError("终局运输 LP 的证书无效。")
    elapsed = float(reply["elapsed"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("远程任务耗时无效。")
    layout = []
    for slot, device in enumerate(permutation):
        machine, target = machines[device], machines[slot]
        dims = ("width", "height")
        if any(round(machine[k], 1) != round(target[k], 1) for k in dims):
            raise ValueError("布局存在尺寸不兼容的设备。")
        layout.append(
            dict(
                equipment_name=machine["name"],
                tool_group=machine["group"],
                bay=target["bay"],
                x=target["position_x"],
                y=target["position_y"],
                width=target["width"],
                height=target["height"],
                original_slot=device,
                new_slot=slot,
            )
        )
    scene = dict(
        factory_id="SMAT2022-LVHM_E",
        scope="full",
        horizon_min=meta["horizon_min"],
        effective_time_fraction=meta["eta"],
        new_routes=[],
        demands=[
            dict(part=p, route="r_" + p.removeprefix("part_"), demand_lots=d)
            for p, d in meta["demands"].items()
        ],
    )
    return dict(
        schema="fab-result-v1",
        method="rl",
        scenario=scene,
        layout=layout,
        metrics=dict(
            reference_cost=meta["reference_cost"],
            optimized_cost=cost,
            elapsed_sec=elapsed,
            moved=sum(slot != device for slot, device in enumerate(permutation)),
            lp_count=1,
            global_relative_gap_bound=gap,
            primal_infeasibility=residual,
        ),
    )
