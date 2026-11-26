# -*- coding: utf-8 -*-
"""结果可视化：设备布局 / 物流关系 / 设备使用频率热力图 / 通道物流频率热力图。"""

from __future__ import annotations

import csv
import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize, to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from .theme import INK, LINE, MUTED, PANEL, PAPER

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
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
    rails_path = result_dir / "rails.csv"

    layout_rows = _read_csv(layout_path) if layout_path.exists() else []
    flow_rows = _read_csv(flow_path) if flow_path.exists() else []
    util_rows = _read_csv(util_path) if util_path.exists() else []
    rails_rows = _read_csv(rails_path) if rails_path.exists() else []

    pos: dict[str, tuple[float, float]] = {}
    size_of: dict[str, tuple[float, float]] = {}
    bay_of: dict[str, str] = {}
    group_of: dict[str, str] = {}
    for r in layout_rows:
        name = r.get("equipment_name", "")
        if not name:
            continue
        pos[name] = (_num(r.get("x"), np.nan), _num(r.get("y"), np.nan))
        size_of[name] = (_num(r.get("width")), _num(r.get("height")))
        bay_of[name] = r.get("bay", "")
        group_of[name] = r.get("tool_group", "")

    names = list(pos.keys())
    xs = np.array([pos[n][0] for n in names], dtype=float)
    ys = np.array([pos[n][1] for n in names], dtype=float)
    sizes = np.array([size_of[n] for n in names], dtype=float).reshape(len(names), 2)
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
    for filename in ("summary.json", "e3_summary.json"):
        try:
            record = json.loads((result_dir / filename).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record:
            summary = record
            break

    return {
        "names": names, "x": xs, "y": ys, "bays": bays, "groups": groups,
        "width": sizes[:, 0], "height": sizes[:, 1],
        "flows": flows, "util_full": util_full, "rails": rails,
        "summary": summary,
    }


# ------------------------------------------------------------------ 网格栅格化
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


# ------------------------------------------------------------------ 绘图样式
_CATEGORY_COLORS = np.array([to_rgb(color) for color in (
    "#4A8D82", "#6688B1", "#BA7858", "#809C57", "#AA7E9D",
    "#B59654", "#528D9F", "#8C80AB", "#5E9367", "#AA7778",
    "#6C91A6", "#A19A57", "#785E9E", "#448475", "#AD7591",
    "#858A54", "#647C95", "#BD8859", "#509694", "#7F719B",
)])
_UTIL_CMAP = LinearSegmentedColormap.from_list(
    "equipment_utilization", ["#E9F3E9", "#9ECAB5", "#438D79", "#124A40"]
)
_CHANNEL_CMAP = LinearSegmentedColormap.from_list(
    "transport_intensity", ["#F5F9F6", "#CDE3D8", "#79B6AA", "#3C8290", "#173B54"]
)


def _style_axis(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=7)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.025)
    _no_axes(ax)
    for spine in ax.spines.values():
        spine.set_color(LINE)
        spine.set_linewidth(0.65)


def _colorbar(ax, mappable, label, **kwargs):
    cax = ax.inset_axes([1.018, 0.035, 0.029, 0.93])
    cax.set_label("<colorbar>")
    bar = ax.figure.colorbar(mappable, cax=cax, **kwargs)
    bar.set_label(label, color=MUTED, fontsize=9)
    bar.ax.tick_params(labelsize=8, colors=MUTED, length=2.5)
    bar.outline.set_edgecolor(LINE)
    return bar


def _categories(labels):
    unique = sorted(set(labels), key=lambda text: [int(t) if t.isdecimal() else t
                                                   for t in re.split(r"([0-9]+)", text)])
    index = {label: i for i, label in enumerate(unique)}
    positions = np.arange(len(unique))
    palette = _CATEGORY_COLORS[positions % len(_CATEGORY_COLORS)]
    lightening = 0.32 * (positions // len(_CATEGORY_COLORS)) / max(
        1, (len(unique) - 1) // len(_CATEGORY_COLORS)
    )
    palette += (1.0 - palette) * lightening[:, None]
    colors = ListedColormap(palette)
    limits = BoundaryNorm(np.arange(len(unique) + 1) - 0.5, colors.N)
    return np.array([index[name] for name in labels]), unique, colors, limits


def draw_layout(ax, data, title: str = "设备布局", by: str = "bay", *, show_colorbar: bool = True):
    _style_axis(ax, title)
    x, y = data["x"], data["y"]
    if len(x) == 0:
        ax.text(0.5, 0.5, "无布局数据", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
        return

    if data.get("rails"):
        segments = np.asarray(data["rails"])[:, :4].reshape(-1, 2, 2)
        ax.add_collection(LineCollection(segments, colors="#C0D2C8", linewidths=0.48, zorder=0))

    if by == "utilization":
        values, cmap, norm = data["util_full"], _UTIL_CMAP, Normalize(0, 1)
    else:
        labels = data["groups"] if by == "group" else data["bays"]
        values, categories, cmap, norm = _categories(labels)

    width = data.get("width", np.zeros_like(x))
    height = data.get("height", np.zeros_like(y))
    sized = np.isfinite(x) & np.isfinite(y) & np.isfinite(width) & np.isfinite(height) & (width > 0) & (height > 0)
    mappable = None
    if sized.any():
        centers = np.column_stack((x[sized], y[sized]))
        half_size = np.column_stack((width[sized], height[sized])) / 2
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        vertices = centers[:, None, :] + half_size[:, None, :] * corners
        blocks = PolyCollection(vertices, cmap=cmap, norm=norm, edgecolors="#375454",
                                linewidths=0.18, zorder=3)
        blocks.set_array(values[sized])
        ax.add_collection(blocks)
        mappable = blocks

    points = ~sized
    if points.any():
        dots = ax.scatter(x[points], y[points], c=values[points], cmap=cmap, norm=norm,
                          s=15, edgecolors=PAPER, linewidths=0.25, zorder=3)
        if mappable is None:
            mappable = dots
    ax.autoscale_view()

    if not show_colorbar:
        return
    if by == "utilization":
        _colorbar(ax, mappable, "设备利用率", ticks=np.linspace(0, 1, 5),
                  format=PercentFormatter(xmax=1, decimals=0))
    else:
        shown = np.linspace(0, len(categories) - 1, min(7, len(categories)), dtype=int)
        bar = _colorbar(ax, mappable, "设备组" if by == "group" else "Bay", ticks=shown)
        bar.set_ticklabels([categories[i] or "未标注" for i in shown])


def draw_flows(ax, data, max_lines: int = 300):
    flows = sorted(data["flows"], key=lambda f: -f[4])[:max_lines]
    if not flows:
        return
    _style_axis(ax, "设备布局与物流关系")
    rows = np.asarray(flows)
    peak = max(rows[:, 4].max(), 1e-12)
    intensity = np.clip(rows[:, 4] / peak, 0, 1)
    colors = np.tile([0.18, 0.39, 0.53, 1.0], (len(rows), 1))
    colors[:, 3] = 0.065 + 0.25 * intensity
    ax.add_collection(LineCollection(rows[:, :4].reshape(-1, 2, 2),
                                     colors=colors, linewidths=0.18 + 1.15 * intensity,
                                     zorder=2))
    ax.autoscale_view()


def draw_channel_heatmap(ax, data):
    _style_axis(ax, "通道物流频率热力图")
    rails = data.get("rails", [])
    if not rails:
        ax.text(0.5, 0.5, "无走廊数据", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
        return
    grid, extent = _segments_grid(rails)
    # 物流强度跨度悬殊，取对数提升整体可读性
    plot_grid = np.log1p(grid)
    img = ax.imshow(plot_grid, extent=extent, origin="lower", cmap=_CHANNEL_CMAP,
                    interpolation="nearest")
    segments = np.asarray(rails)[:, :4].reshape(-1, 2, 2)
    ax.add_collection(LineCollection(segments, colors="#B8D0C8", linewidths=0.35,
                                     alpha=0.50, zorder=1), autolim=False)
    _colorbar(ax, img, "物流强度 ln(1+x)")


# ------------------------------------------------------------------ 画布封装
class ResultCanvas:
    """Matplotlib 画布及图层面板。"""

    def __init__(self, master):
        self.data: dict[str, Any] | None = None
        self.stacked = True
        container = ttk.Frame(master)
        container.pack(fill="both", expand=True)

        ctrl = ttk.Frame(container, style="Sidebar.TFrame", width=210, padding=12)
        ctrl.pack(side="left", fill="y", padx=(0, 11))
        ctrl.pack_propagate(False)
        ttk.Label(ctrl, text="展示图层", style="PanelHeading.TLabel").pack(anchor="w", pady=(0, 9))

        self.var_layout = tk.BooleanVar(value=True)
        self.var_flows = tk.BooleanVar(value=False)
        self.var_channel = tk.BooleanVar(value=False)
        for label, var in (
            ("设备布局", self.var_layout),
            ("设备间物流关系", self.var_flows),
            ("通道物流频率热力图", self.var_channel),
        ):
            ttk.Checkbutton(ctrl, text=label, variable=var,
                            command=self.redraw).pack(anchor="w", pady=2)

        ttk.Separator(ctrl).pack(fill="x", pady=14)
        ttk.Label(ctrl, text="布局着色依据", style="PanelHeading.TLabel").pack(anchor="w", pady=(0, 9))
        self.var_color_by = tk.StringVar(value="Bay")
        for label in ("Bay", "设备组", "设备使用频率"):
            ttk.Radiobutton(ctrl, text=label, variable=self.var_color_by,
                            value=label, command=self.redraw).pack(anchor="w", pady=2)

        plot = ttk.Frame(container, style="Card.TFrame", padding=7)
        plot.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(8, 5.5), dpi=100, facecolor=PAPER)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        widget = self.canvas.get_tk_widget()
        widget.configure(highlightthickness=0, borderwidth=0, background=PAPER)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        self.toolbar.configure(background=PANEL, borderwidth=0)
        for item in self.toolbar.winfo_children():
            item.configure(background=PANEL)
        self.toolbar.pack(side="bottom", fill="x")
        widget.pack(side="top", fill="both", expand=True)
        widget.bind("<Configure>", self._resize_plots, add="+")

    def _resize_plots(self, event):
        stacked = event.width < 880
        changed = stacked != self.stacked
        self.stacked = stacked
        if changed and self.var_channel.get() and (self.var_layout.get() or self.var_flows.get()):
            self.redraw()
        elif self.fig.axes:
            self.fig.tight_layout(pad=1.15)

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

        panels = []
        if layout_on or flows_on:
            panels.append("base")
        if channel_on:
            panels.append("channel")
        if not panels:
            self.canvas.draw_idle()
            return

        rows, cols = (len(panels), 1) if self.stacked else (1, len(panels))
        axes = self.fig.subplots(rows, cols, squeeze=False).ravel()
        for ax, kind in zip(axes, panels):
            if kind == "base":
                if layout_on:
                    draw_layout(ax, self.data, by=by)
                if flows_on:
                    draw_flows(ax, self.data)
            else:
                draw_channel_heatmap(ax, self.data)
        self.fig.tight_layout(pad=1.15)
        self.canvas.draw_idle()
