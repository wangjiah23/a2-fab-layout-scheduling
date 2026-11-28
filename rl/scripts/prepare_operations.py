"""Rebuild the path LP's compact operation arrays with the upstream E1 parser."""

import argparse
import json
import sys
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np


def main(source_repo, instance):
    sys.path.insert(0, str(source_repo.resolve() / "scripts"))
    from e0_data_audit import read_tsv
    from e1_reference_lp import build_operations

    meta = json.loads((instance / "instance.json").read_text(encoding="utf-8"))
    by_group = defaultdict(list)
    for machine, equipment in enumerate(meta["equipment"]):
        by_group[equipment["group"]].append(machine)
    directory = source_repo / "data" / "SMT2020_LVHM"
    smt = {
        "directory": directory,
        "parts": read_tsv(directory / "part.txt"),
        "orders": read_tsv(directory / "order.txt"),
    }
    operations, demands, warnings = build_operations(smt, by_group, meta["horizon_min"], meta["demand_scale"])
    if warnings or len(operations) != meta["physical_steps"] or demands != meta["demands"]:
        raise ValueError(f"Upstream routes disagree with the slot instance: {warnings}")

    groups = {name: i for i, name in enumerate(meta["groups"])}
    group_ids = [groups[op["stnfam"]] for op in operations]
    times = [op["tau_per_lot"] for op in operations]
    routes = [(name, sum(1 for _ in ops)) for name, ops in groupby(operations, key=lambda op: op["part"])]
    if len(routes) != len(demands):
        raise ValueError("Expected one contiguous route per product")
    destination = instance / "operations.npz"
    np.savez_compressed(
        destination,
        group_ids=np.asarray(group_ids, dtype="<i2"),
        tau=np.asarray(times, dtype="<f8"),
        piece_scaled=np.asarray([op["ptper"] in {"per_piece", "per_batch"} for op in operations]),
        product_offsets=np.asarray([0, *np.cumsum([n for _, n in routes])], dtype="<i4"),
        demands=np.asarray([demands[p] for p, _ in routes], dtype="<f8"),
    )
    print(f"{len(routes)} products, {len(operations)} operations -> {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo", type=Path, required=True, help="Local checkout of Stelle7th/Course-Design-SMT"
    )
    parser.add_argument(
        "--instance",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "original_slot_instance",
    )
    args = parser.parse_args()
    main(args.source_repo, args.instance)
