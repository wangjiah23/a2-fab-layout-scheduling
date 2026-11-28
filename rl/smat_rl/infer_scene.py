"""Infer one submitted factory layout and certify both alternatives on its own orders."""

import argparse
import csv
import io
import json
import time
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from .env import OriginalSlotEnv, configure_slot_compatibility
from .lp.instance import OriginalSlotLP
from .lp.paths import PathLP
from .model import SlotActorCritic
from .placement import GroupDecisionTargets, GroupSlotPlan
from .rollout import collect_policy
from .runtime import json_save
from .scene import apply_scenario, initial_permutation, layout_rows


@torch.inference_mode()
def policy_layout(lp, saved, device):
    env = OriginalSlotEnv(lp, evaluate_terminal=False)
    config = saved["config"]
    plan = GroupSlotPlan(env)
    targets = GroupDecisionTargets(env, plan, device)
    with np.errstate(divide="ignore"):
        targets.log_completion_counts = torch.as_tensor(np.log(plan.completion_counts), device=device)
    model = SlotActorCritic(env, routing_actor=config.get("actor_features") == "routing").to(device).eval()
    # Instance buffers encode the submitted routes. Only learned actor parameters transfer.
    for name, parameter in model.named_parameters():
        if not name.startswith("critic."):
            parameter.copy_(saved["model"][name])
    actions, layout, _ = collect_policy(
        model,
        env,
        plan,
        targets,
        1,
        torch.Generator(device=device).manual_seed(config["seed"]),
        config["temperature"],
        greedy=True,
        completion_prior=config["completion_prior"],
    )
    return actions[0], layout[0], np.flatnonzero(~plan.is_decision)


def write_deliverables(directory, package, allocation, lp, placement):
    rows = package["layout"]
    equipment = lp.meta["equipment"]
    locations = np.argsort(placement)
    distances = lp.data["distances"]
    flows = np.zeros(distances.shape, dtype=np.float64)
    for _, path, lots in allocation["paths"]:
        np.add.at(flows, (path[:-1], path[1:]), lots)
    bay_flow = defaultdict(float)
    cost = 0.0
    links = []
    for a, b in zip(*np.nonzero(flows)):
        origin, dest = int(locations[a]), int(locations[b])
        lots, distance = float(flows[a, b]), float(distances[origin, dest])
        bay_a, bay_b = equipment[origin]["bay"], equipment[dest]["bay"]
        links.append((equipment[a]["name"], equipment[b]["name"], lots, distance, bay_a, bay_b))
        bay_flow[bay_a, bay_b] += lots
        cost += lots * distance
    if not np.isclose(cost, package["metrics"]["optimized_cost"], rtol=1e-7, atol=0.1):
        raise RuntimeError("导出的逐弧物流与终局运输成本不一致。")
    capacity = lp.meta["horizon_min"] * lp.meta["eta"]

    def csv_file(archive, name, fields, items):
        with (
            archive.open(name, "w") as raw,
            io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream,
        ):
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(items)

    with ZipFile(directory / "simulation-input.zip.tmp", "w", ZIP_DEFLATED) as archive:
        csv_file(archive, "layout.csv", rows[0].keys(), (r.values() for r in rows))
        csv_file(
            archive,
            "transport_flow.csv",
            [
                "source_equipment",
                "target_equipment",
                "flow_lots",
                "path_distance",
                "source_bay",
                "target_bay",
            ],
            links,
        )
        csv_file(
            archive,
            "device_utilization.csv",
            ["equipment_name", "available_time_min", "load_min", "utilization"],
            ((m["name"], capacity, load, load / capacity) for m, load in zip(equipment, allocation["loads"])),
        )
        csv_file(
            archive,
            "from_to_bay.csv",
            ["source_bay", "target_bay", "flow_lots"],
            ((a, b, lots) for (a, b), lots in sorted(bay_flow.items())),
        )
        archive.writestr("summary.json", json.dumps(package["metrics"], ensure_ascii=False, allow_nan=False))
        archive.writestr("result.json", json.dumps(package, ensure_ascii=False, allow_nan=False))
    (directory / "simulation-input.zip.tmp").replace(directory / "simulation-input.zip")


def solve(directory, checkpoint, device):
    started = time.monotonic()
    scene = json.loads((directory / "request.json").read_bytes())["scenario"]

    def phase(label):
        json_save(directory / "progress.json", {"phase": label})

    phase("校验场景、机位与产能")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lp = OriginalSlotLP()
    configure_slot_compatibility(lp, "footprint")
    if saved["config"]["instance_fingerprint"] != lp.fingerprint:
        raise RuntimeError("服务配置的权重与已部署 E3 工厂不匹配。")
    initial = initial_permutation(scene.get("initial_layout"), lp.meta["equipment"])
    original = apply_scenario(lp, scene)
    lp.validate_layout(initial)
    phase("PPO 正在逐台放置设备")
    actions, proposed, automatic = policy_layout(lp, saved, device)
    lp.validate_layout(proposed)
    kwargs = dict(seed_flow=lp.seed_flow if original else None, columns_per_product=3)
    master = PathLP(lp, **kwargs)
    pricer = None
    if device.type == "cuda":
        from .lp.gpu_path_pricing import CudaPathPricer

        try:
            pricer = CudaPathPricer(
                master.distance,
                master.product_offsets,
                master.operation_offsets,
                master.eligible_machines,
                master.tau,
                device=0,
            )
        except Exception as exc:
            print(f"CUDA pricing unavailable: {exc}", flush=True)

    def certify(layout):
        inverse = np.argsort(layout).astype(np.int32)
        try:
            pricing = (lambda _distance, dual: pricer.price(dual, inverse)) if pricer is not None else None
            return master.solve(layout, pricing=pricing, include_allocation=True)
        except RuntimeError as exc:
            print(f"Retrying this scenario with a cold CPU path master: {exc}", flush=True)
            return PathLP(lp, seed_flow=kwargs["seed_flow"]).solve(layout, include_allocation=True)

    phase("计算本次输入布局的运输 LP")
    baseline = certify(initial)
    phase("校验 PPO 布局，比较运输成本")
    candidate = certify(proposed)
    keep = candidate["cost"] >= baseline["cost"] - max(0.01, abs(baseline["cost"]) * 1e-7)
    allocation, selected = (baseline, initial) if keep else (candidate, proposed)
    rows = layout_rows(lp.meta["equipment"], selected, initial)
    metrics = dict(
        reference_cost=baseline["cost"],
        optimized_cost=allocation["cost"],
        policy_cost=candidate["cost"],
        moved=int(np.count_nonzero(initial != selected)),
        lp_count=2,
        global_relative_gap_bound=allocation["global_relative_gap_bound"],
        primal_infeasibility=allocation["primal_infeasibility"],
        elapsed_sec=time.monotonic() - started,
    )
    if baseline["cost"] > 0:
        metrics["cost_reduction_pct"] = 100 * (1 - allocation["cost"] / baseline["cost"])
    package = dict(
        schema="fab-result-v1",
        method="rl",
        policy_device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        scenario=scene,
        layout=rows,
        metrics=metrics,
        selection="initial" if keep else "policy",
        replay=dict(
            slots=actions.tolist(),
            automatic_steps=automatic.tolist(),
            cycle=saved["cycle"],
            checkpoint=checkpoint.name,
        ),
    )
    if scene.get("initial_layout"):
        package["reference_layout"] = layout_rows(lp.meta["equipment"], initial, initial)
    if keep:
        package["policy_layout"] = layout_rows(lp.meta["equipment"], proposed, initial)
    phase("导出布局、动作记录与物流分配")
    write_deliverables(directory, package, allocation, lp, selected)
    json_save(directory / "result.json", package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    try:
        solve(args.job, args.checkpoint, torch.device(args.device))
    except ValueError as exc:
        json_save(args.job / "error.json", {"detail": str(exc)})
        raise


if __name__ == "__main__":
    main()
