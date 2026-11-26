# -*- coding: utf-8 -*-
"""结果可视化：设备布局 / 物流关系 / 设备使用频率热力图 / 通道物流频率热力图。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GRID_SIZE = 140  # 热力图网格分辨率


# ------------------------------------------------------------------ 数据加载
def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_result(result_dir: Path) -> dict[str, Any]:
    """读取 result 目录，返回已 join 好坐标、便于绘图的结构化数据。"""
    layout_path = result_dir / "layout.csv"
    flow_path = result_dir / "transport_flow.csv"
    util_path = result_dir / "device_utilization.csv"
    bay_path = result_dir / "from_to_bay.csv"
    rails_path = result_dir / "rails.csv"
    summary_path = result_dir / "summary.json"

    layout_rows = _read_csv(layout_path) if layout_path.exists() else []
    flow_rows = _read_csv(flow_path) if flow_path.exists() else []
    util_rows = _read_csv(util_path) if util_path.exists() else []
    bay_rows = _read_csv(bay_path) if bay_path.exists() else []
    rails_rows = _read_csv(rails_path) if rails_path.exists() else []

    pos: dict[str, tuple[float, float]] = {}
    bay_of: dict[str, str] = {}
    group_of: dict[str, str] = {}
    for r in layout_rows:
        name = r.get("equipment_name", "")
        if not name:
            continue
        pos[name] = (_num(r.get("x")), _num(r.get("y")))
        bay_of[name] = r.get("bay", "")
        group_of[name] = r.get("tool_group", "")

    names = list(pos.keys())
    xs = np.array([pos[n][0] for n in names], dtype=float)
    ys = np.array([pos[n][1] for n in names], dtype=float)
    bays = [bay_of.get(n, "") for n in names]
    groups = [group_of.get(n, "") for n in names]

    flows: list[tuple[float, float, float, float, float]] = []
    for r in flow_rows:
        s, t = r.get("source_equipment", ""), r.get("target_equipment", "")
        if s in pos and t in pos:
            flows.append((pos[s][0], pos[s][1], pos[t][0], pos[t][1],
                          _num(r.get("flow_lots"))))

    util_map = {r.get("equipment_name", ""): _num(r.get("utilization"))
                for r in util_rows}
    # 与 names 对齐的设备使用频率（缺失设备计 0）
    util_full = np.array([util_map.get(n, 0.0) for n in names], dtype=float)

    # 走廊（轨道）几何 + 物流强度
    rails = [(_num(r.get("x1")), _num(r.get("y1")),
              _num(r.get("x2")), _num(r.get("y2")), _num(r.get("flow")))
             for r in rails_rows]

    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    return {
        "names": names, "x": xs, "y": ys, "bays": bays, "groups": groups,
        "flows": flows, "util_full": util_full, "rails": rails,
        "summary": summary,
    }


# ------------------------------------------------------------------ 网格栅格化
def _category_index(labels: list[str]) -> np.ndarray:
    uniq = sorted(set(labels))
    idx = {b: i for i, b in enumerate(uniq)}
    return np.array([idx[b] for b in labels])


def _no_axes(ax) -> None:
    """隐藏坐标轴刻度与标签，仅保留图形主体。"""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def _segments_grid(rails: list[tuple[float, float, float, float, float]]
                   ) -> tuple[np.ndarray, list[float]]:
    """把轨道线段栅格化到规则网格，物流强度沿线段累加。"""
    active = [r for r in rails if r[4] > 0]
    if not active:
        return np.zeros((GRID_SIZE, GRID_SIZE)), [0, 1, 0, 1]
    xs = np.concatenate([[r[0], r[2]] for r in active])
    ys = np.concatenate([[r[1], r[3]] for r in active])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0
    cell = max(xmax - xmin, ymax - ymin) / GRID_SIZE
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for x1, y1, x2, y2, v in active:
        length = max(abs(x2 - x1), abs(y2 - y1))
        n = max(2, int(length / cell) + 1)
        for t in np.linspace(0.0, 1.0, n):
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            xi = int(np.clip((px - xmin) / (xmax - xmin) * (GRID_SIZE - 1),
                             0, GRID_SIZE - 1))
            yi = int(np.clip((py - ymin) / (ymax - ymin) * (GRID_SIZE - 1),
                             0, GRID_SIZE - 1))
            grid[yi, xi] += v
    return grid, [xmin, xmax, ymin, ymax]


def _imshow_heatmap(ax, grid: np.ndarray, extent: list[float], cmap: str,
                    title: str, label: str) -> None:
    img = ax.imshow(grid, extent=extent, origin="lower", cmap=cmap,
                    interpolation="nearest", aspect="auto")
    ax.set_title(title)
    _no_axes(ax)
    cb = ax.figure.colorbar(img, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(label)


# ------------------------------------------------------------------ 绘图函数
def draw_layout(ax, data, title: str = "设备布局", by: str = "bay"):
    if len(data["x"]) == 0:
        ax.text(0.5, 0.5, "无布局数据", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        _no_axes(ax)
        return
    if by == "utilization":
        # 设备使用频率热力图：以利用率为颜色叠加在布局上
        sc = ax.scatter(data["x"], data["y"], c=data["util_full"], cmap="YlOrRd",
                        s=18, linewidths=0.2, edgecolors="white")
        ax.set_title(title)
        _no_axes(ax)
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("设备使用频率（利用率）")
    else:
        labels = data["groups"] if by == "group" else data["bays"]
        ci = _category_index(labels)
        cmap = "turbo" if by == "group" else "tab20"
        sc = ax.scatter(data["x"], data["y"], c=ci, cmap=cmap, s=18,
                        linewidths=0.2, edgecolors="white")
        ax.set_title(title)
        _no_axes(ax)
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("设备组 类别" if by == "group" else "Bay 类别")


def draw_flows(ax, data, max_lines: int = 300):
    flows = sorted(data["flows"], key=lambda f: -f[4])[:max_lines]
    if not flows:
        return
    wmax = max((f[4] for f in flows), default=1.0)
    for sx, sy, tx, ty, w in flows:
        lw = 0.2 + 2.8 * (w / wmax)
        ax.plot([sx, tx], [sy, ty], color="C0", linewidth=lw, alpha=0.35, zorder=1)
    ax.set_title("设备布局与物流关系")
    _no_axes(ax)


def draw_channel_heatmap(ax, data):
    rails = data.get("rails", [])
    if not rails:
        ax.text(0.5, 0.5, "无走廊数据", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("通道物流频率热力图")
        _no_axes(ax)
        return
    grid, extent = _segments_grid(rails)
    # 物流强度跨度悬殊，取对数提升整体可读性
    plot_grid = np.log1p(grid)
    _imshow_heatmap(ax, plot_grid, extent, "viridis",
                    "通道物流频率热力图", "物流强度 (log)")


# ------------------------------------------------------------------ 画布封装
class ResultCanvas:
    """matplotlib 嵌入 Tk 的画布，含勾选控制面板与动态重绘。"""

    def __init__(self, master):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.data: dict[str, Any] | None = None

        container = ttk.Frame(master)
        container.pack(fill="both", expand=True)

        ctrl = ttk.LabelFrame(container, text="展示图层")
        ctrl.pack(side="left", fill="y", padx=6, pady=6)

        self.var_layout = tk.BooleanVar(value=True)
        self.var_flows = tk.BooleanVar(value=False)
        self.var_channel = tk.BooleanVar(value=False)

        ttk.Checkbutton(ctrl, text="设备布局", variable=self.var_layout,
                        command=self.redraw).pack(anchor="w", padx=8, pady=4)
        ttk.Checkbutton(ctrl, text="设备间物流关系", variable=self.var_flows,
                        command=self.redraw).pack(anchor="w", padx=8, pady=4)
        ttk.Checkbutton(ctrl, text="通道物流频率热力图", variable=self.var_channel,
                        command=self.redraw).pack(anchor="w", padx=8, pady=4)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=6, pady=6)
        ttk.Label(ctrl, text="布局着色依据").pack(anchor="w", padx=8)
        self.var_color_by = tk.StringVar(value="Bay")
        for text in ("Bay", "设备组", "设备使用频率"):
            ttk.Radiobutton(ctrl, text=text, variable=self.var_color_by,
                            value=text, command=self.redraw).pack(
                                anchor="w", padx=16, pady=2)

        self.fig = Figure(figsize=(8, 5.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=container)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, container)
        self.toolbar.update()

    def set_data(self, data: dict[str, Any]) -> None:
        self.data = data
        self.redraw()

    def redraw(self) -> None:
        self.fig.clear()
        if self.data is None:
            self.canvas.draw_idle()
            return

        layout_on = self.var_layout.get()
        flows_on = self.var_flows.get()
        channel_on = self.var_channel.get()
        by_map = {"Bay": "bay", "设备组": "group", "设备使用频率": "utilization"}
        by = by_map.get(self.var_color_by.get(), "bay")

        base_on = layout_on or flows_on
        panels = []
        if base_on:
            panels.append("base")
        if channel_on:
            panels.append("channel")

        if not panels:
            self.canvas.draw_idle()
            return

        n = len(panels)
        axes = self.fig.subplots(1, n, squeeze=False)[0] if n > 1 else [self.fig.subplots(1, 1)]
        if n == 1:
            axes = [self.fig.subplots(1, 1)]

        for ax, kind in zip(axes, panels):
            if kind == "base":
                if layout_on:
                    draw_layout(ax, self.data, title="设备布局", by=by)
                if flows_on:
                    draw_flows(ax, self.data)
            elif kind == "channel":
                draw_channel_heatmap(ax, self.data)

        self.fig.tight_layout()
        self.canvas.draw_idle()
