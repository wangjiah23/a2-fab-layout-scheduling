"""Prepare a workbench scenario on the installed SMAT2022 plant."""

from collections import defaultdict

import numpy as np


def group_index(meta):
    return {name.casefold(): i for i, name in enumerate(meta["groups"])}


def initial_permutation(rows, equipment):
    """Resolve a layout.csv by physical coordinates; a supplied slot ID must agree."""
    count = len(equipment)
    if rows is None:
        return np.arange(count, dtype=np.int32)
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError(f"起始布局须包含全部 {count} 台设备，每台恰好一次。")
    by_name = {m["name"]: i for i, m in enumerate(equipment)}
    xy = np.array([(m["position_x"], m["position_y"]) for m in equipment])
    layout = np.full(count, -1, np.int32)
    seen = set()
    for row in rows:
        name = row["equipment_name"]
        if name not in by_name or name in seen:
            raise ValueError(f"起始布局设备编号未知或重复：{name}")
        seen.add(name)
        machine = by_name[name]
        x, y = row["x"], row["y"]
        match = np.flatnonzero(np.max(np.abs(xy - [x, y]), axis=1) <= 0.01)
        if len(match) != 1:
            raise ValueError(f"{name} 的坐标未对应既有机位。新增机位或轨道需建立新的工厂实例。")
        slot = int(match[0])
        target, source = equipment[slot], equipment[machine]
        if layout[slot] != -1 or row.get("new_slot") not in (None, "", slot, str(slot)):
            raise ValueError(f"{name} 的坐标、机位编号或机位占用有冲突。")
        if row.get("tool_group") and row["tool_group"].casefold() != source["group"].casefold():
            raise ValueError(f"{name} 的设备组与工厂设备清单不符。")
        if row.get("bay") and row["bay"] != target["bay"]:
            raise ValueError(f"{name} 的 Bay 与坐标不符。")
        for key in ("width", "height"):
            dim = round(target[key], 1)
            if round(source[key], 1) != dim or (row.get(key) is not None and round(row[key], 1) != dim):
                raise ValueError(f"{name} 与该机位的宽高不兼容。")
        layout[slot] = machine
    return layout


def apply_scenario(lp, scene):
    """Rebuild routes, LP capacities, directed flow weights and GCN features in memory."""
    meta, data = lp.meta, lp.data
    groups = group_index(meta)
    original = lp.operations
    # The release publishes its product offsets in the original part-table order.
    catalog = {}
    for i, part in enumerate(meta["demands"]):
        lo, hi = original["product_offsets"][i : i + 2]
        catalog["r_" + part.removeprefix("part_")] = (
            original["group_ids"][lo:hi],
            original["tau"][lo:hi],
            original["piece_scaled"][lo:hi],
        )
    new_routes = defaultdict(dict)
    for step in scene["new_routes"]:
        route, pos = step["route"], step["step"]
        if route in catalog or pos in new_routes[route]:
            raise ValueError(f"路线 {route} 与已有路线重名，或工序 {pos} 重复。")
        group = groups.get(step["stnfam"].casefold())
        if step["step_type"] == "process" and group is None:
            raise ValueError(f"路线 {route} 的设备族 {step['stnfam']} 无法映射到本厂。")
        new_routes[route][pos] = (group, step["time_min_per_lot"]) if step["step_type"] == "process" else None
    for route, positions in new_routes.items():
        ops = [v for _, v in sorted(positions.items()) if v is not None]
        if not ops:
            raise ValueError(f"路线 {route} 至少需要一道加工工序。")
        ids, tau = zip(*ops)
        catalog[route] = np.asarray(ids, np.int16), np.asarray(tau, np.float64), None

    time_capacity = scene["horizon_min"] * scene["effective_time_fraction"]
    group_load = np.zeros(len(groups))
    group_parts = np.zeros(len(groups))
    flow = np.zeros_like(data["flow_weights"])
    ids, times, loads, offsets = [], [], [], [0]
    parts = set()
    for product in scene["demands"]:
        part, route, lots = product["part"], product["route"], product["demand_lots"]
        if part in parts or route not in catalog:
            raise ValueError(f"产品 {part} 重复或尚未定义路线 {route}。")
        parts.add(part)
        if not lots:
            continue
        g, tau, piece_scaled = catalog[route]
        pieces = product.get("pieces_per_lot", 25)
        if piece_scaled is not None and pieces != 25:
            tau = np.where(piece_scaled, tau * pieces / 25, tau)
        if offsets[-1] + len(g) > 30000:
            raise ValueError("一个任务最多可展开 30,000 道加工工序。")
        ids.append(g)
        times.append(tau)
        loads.append(lots)
        offsets.append(offsets[-1] + len(g))
        np.add.at(group_load, g, tau * lots)
        np.add.at(flow, (g[:-1], g[1:]), lots)
        group_parts[np.unique(g)] += 1
    if not ids:
        raise ValueError("至少需要一个有正需求的产品。")
    utilization = group_load / (data["inventory"].astype(np.float64) * time_capacity)
    peak = int(utilization.argmax())
    if not np.isfinite(utilization).all() or utilization[peak] > 1 + 1e-10:
        raise ValueError(
            f"订单超过设备组 {meta['groups'][peak]} 的可用机时，负荷为 {utilization[peak]:.1%}。"
        )
    operations = dict(
        group_ids=np.concatenate(ids),
        tau=np.concatenate(times),
        demands=np.asarray(loads),
        product_offsets=np.asarray(offsets, np.int32),
    )
    same_routes = all(np.array_equal(original[k], v) for k, v in operations.items())
    same_capacity = (
        scene["horizon_min"] == meta["horizon_min"] and scene["effective_time_fraction"] == meta["eta"]
    )
    if same_routes and same_capacity:
        return True
    static = data["static_graph"].copy()
    static[:, 3] = utilization
    static[:, 4] = group_parts / len(ids)
    lp.operations = operations
    lp.data = {**data, "flow_weights": flow, "static_graph": static}
    lp.meta = {**meta, "horizon_min": scene["horizon_min"], "eta": scene["effective_time_fraction"]}
    return False


def layout_rows(equipment, permutation, initial):
    origin = np.argsort(initial)
    return [
        dict(
            equipment_name=equipment[m]["name"],
            tool_group=equipment[m]["group"],
            bay=s["bay"],
            x=s["position_x"],
            y=s["position_y"],
            width=s["width"],
            height=s["height"],
            original_slot=int(origin[m]),
            new_slot=i,
        )
        for i, (m, s) in enumerate(zip(permutation, equipment))
    ]
