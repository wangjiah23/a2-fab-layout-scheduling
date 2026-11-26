
"""Compare a saved optimization result with the workshop's original layout."""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .rails import load_smat, numeric
from .theme import ACCENT, BACKGROUND, ERROR, INK, LINE, LOG_FONT, MUTED, NAV, PANEL, PAPER, SELECTED, SMALL_FONT, UI_FONT
from .visualization import draw_layout

_MOVE_COLOR = "#C66E26"


def load_reference_layout(smat_file: Path, result: dict) -> dict:
    """Read the original slots for exactly the equipment in the loaded result."""
    equipment = {
        str(row["NAME"]).strip(): row
        for row in load_smat(smat_file)["Equipment"]
        if row.get("NAME")
    }
    names = result["names"]
    missing = [name for name in names if name.strip() not in equipment]
    if missing:
        sample = "、".join(missing[:4])
        raise ValueError(f"内置原始厂房中找不到 {len(missing)} 台设备（如 {sample}）。请使用本厂的结果。")
    rows = [equipment[name.strip()] for name in names]
    position = np.array([[numeric(row.get("POSITION_X")), numeric(row.get("POSITION_Y"))]
                         for row in rows], dtype=float)
    if not np.isfinite(position).all():
        raise ValueError("原始厂房的部分对应设备缺少有效坐标。")
    dimensions = np.array([[numeric(row.get("WIDTH")), numeric(row.get("HEIGHT"))]
                           for row in rows], dtype=float)

    return {
        "names": names, "x": position[:, 0], "y": position[:, 1],
        "width": dimensions[:, 0], "height": dimensions[:, 1],
        "bays": [str(row.get("BAY") or "") for row in rows],
        # Device group names follow the solver's mapping in both panels.
        "groups": result["groups"], "rails": result["rails"],
    }


def _cost(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def transport_metrics(summary: dict) -> tuple[float | None, float | None, float | None]:
    """Get a pair of archived objective values and the reduction as a fraction."""
    records = (summary, summary.get("result"))
    first_pair = (None, None)
    for record in records:
        if not isinstance(record, dict):
            continue
        before = _cost(record.get("reference_cost"))
        after = _cost(record.get("optimized_cost"))
        if before is not None and after is not None:
            reduction = (before - after) / before if before > 0 else None
            return before, after, reduction
        if first_pair == (None, None):
            first_pair = (before, after)
    return first_pair[0], first_pair[1], None


def _metric(parent, column, label, value, detail, emphasized=False, color=INK):
    background = SELECTED if emphasized else PAPER
    tile = tk.Frame(parent, background=background, highlightbackground=LINE, highlightthickness=1)
    tile.grid(row=0, column=column, padx=(0, 9) if column < 2 else 0, sticky="nsew")
    tk.Label(tile, text=label, background=background, foreground=MUTED,
             font=SMALL_FONT).pack(anchor="w", padx=12, pady=(8, 0))
    tk.Label(tile, text=value, background=background, foreground=color,
             font=(LOG_FONT[0], 17, "bold")).pack(anchor="w", padx=12, pady=(0, 1))
    tk.Label(tile, text=detail, background=background, foreground=MUTED,
             font=SMALL_FONT).pack(anchor="w", padx=12, pady=(0, 7))


def _objective_text(value):
    return "未记录" if value is None else f"{value:,.2f}"


def show_comparison(parent, result_dir: Path, result: dict, smat_file: Path) -> tk.Toplevel:
    if not np.isfinite(result["x"]).all() or not np.isfinite(result["y"]).all():
        raise ValueError("优化结果中有设备缺少有效坐标。")
    original = load_reference_layout(smat_file, result)
    optimized = dict(result)
    for side in ("width", "height"):
        sizes = result[side]
        optimized[side] = np.where(np.isfinite(sizes) & (sizes > 0), sizes, original[side])
    moved_distances = np.hypot(optimized["x"] - original["x"], optimized["y"] - original["y"])
    moved = np.count_nonzero(moved_distances > 1e-4)
    before, after, reduction = transport_metrics(result["summary"])

    win = tk.Toplevel(parent)
    win.title("布局优化对比")
    win.configure(background=BACKGROUND)
    win.transient(parent.winfo_toplevel())
    width = min(1180, win.winfo_screenwidth() - 60)
    height = min(680, win.winfo_screenheight() - 95)
    win.geometry(f"{width}x{height}")
    win.minsize(790, 470)
    win.bind("<Escape>", lambda _: win.destroy())

    mast = tk.Frame(win, background=NAV)
    mast.pack(fill="x")
    tk.Label(mast, text="布局优化对比", background=NAV, foreground=PAPER,
             font=(UI_FONT[0], 14, "bold")).pack(side="left", padx=16, pady=9)
    tk.Label(mast, text=result_dir.name, background=NAV, foreground="#CFE2D7",
             font=SMALL_FONT).pack(side="right", padx=16)

    body = ttk.Frame(win, padding=(14, 12))
    body.pack(fill="both", expand=True)
    metrics = ttk.Frame(body)
    metrics.pack(fill="x", pady=(0, 9))
    for column in range(3):
        metrics.columnconfigure(column, weight=1, uniform="metric")
    _metric(metrics, 0, "原始布局 · 加权搬运距离", _objective_text(before), "原始距离单位 × lot")
    _metric(metrics, 1, "优化后布局 · 加权搬运距离", _objective_text(after), "原始距离单位 × lot")
    ratio = f"优化后为原始值的 {after / before:.2%}" if reduction is not None else "需有两项有效目标值且原始值大于零"
    _metric(metrics, 2, "相对降幅", "—" if reduction is None else f"{reduction:.2%}",
            ratio, emphasized=True, color=ERROR if reduction is not None and reduction < 0 else ACCENT)

    note = "目标值来自该次求解汇总，按轨道距离乘搬运批量后求和。降幅＝（原始值−优化后值）÷原始值。"
    ttk.Label(body, text=note, foreground=MUTED, font=SMALL_FONT).pack(side="bottom", anchor="w", pady=(8, 0))

    plot = ttk.Frame(body, style="Card.TFrame", padding=7)
    plot.pack(fill="both", expand=True)
    figure = Figure(figsize=(11, 4.2), dpi=100, facecolor=PAPER)
    axes = figure.subplots(1, 2, sharex=True, sharey=True)
    figure.subplots_adjust(left=0.025, right=0.975, bottom=0.055, top=0.90, wspace=0.07)
    draw_layout(axes[0], original, title="原始布局", by="group", show_colorbar=False)
    draw_layout(axes[1], optimized, title="优化后布局", by="group", show_colorbar=False)
    context = f"同尺度 · 按设备组着色 · 对比 {len(result['names']):,} 台设备 · {moved:,} 台坐标变化"
    if moved:
        selected = int(np.argmax(moved_distances))
        for axis, layout in zip(axes, (original, optimized)):
            axis.scatter(layout["x"][selected], layout["y"][selected], marker="s",
                         s=77, linewidths=1.5, facecolors="none", edgecolors=_MOVE_COLOR, zorder=4)
        name = result["names"][selected]
        old_bay = original["bays"][selected] or "—"
        new_bay = optimized["bays"][selected] or "—"
        context += f"\n橙框标示设备 {name}（{old_bay} → {new_bay}）"
    ttk.Label(plot, text=context, style="File.TLabel", justify="left").pack(anchor="w", padx=8, pady=(5, 2))
    canvas = FigureCanvasTkAgg(figure, plot)
    widget = canvas.get_tk_widget()
    widget.configure(background=PAPER, borderwidth=0, highlightthickness=0)
    toolbar = NavigationToolbar2Tk(canvas, plot, pack_toolbar=False)
    toolbar.configure(background=PANEL, borderwidth=0)
    for child in toolbar.winfo_children():
        child.configure(background=PANEL)
    toolbar.pack(side="bottom", fill="x")
    widget.pack(fill="both", expand=True)

    canvas.draw_idle()
    win.lift()
    return win
