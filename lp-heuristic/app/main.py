# -*- coding: utf-8 -*-
"""应用主入口：推理 / 加载展示 / 接口说明 三个 Tab 的桌面程序。"""

from __future__ import annotations

import json
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import docs_text  # noqa: E402
from app.inference import InferenceWorker, RESULT_ROOT, SMAT_FILE  # noqa: E402
from app.theme import (ACCENT, ERROR, HEADING_FONT, INK, LOG_FONT, MUTED, NAV,
                       PANEL, PAPER, SMALL_FONT, UI_FONT, apply_theme)  # noqa: E402
from app.visualization import ResultCanvas, load_result  # noqa: E402
from app.comparison import show_comparison  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "output"


def prepare_sample_result() -> Path:
    """确保 result/ 下存在一个 sample 示例结果（来自现有 output/），便于首次演示。"""
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    sample = RESULT_ROOT / "sample"
    sample.mkdir(parents=True, exist_ok=True)

    layout_src = OUTPUT_ROOT / "e3_global" / "layout.csv"
    e1_dir = OUTPUT_ROOT / "e1_reference_lp"
    if layout_src.exists() and not (sample / "layout.csv").exists():
        shutil.copyfile(layout_src, sample / "layout.csv")
    for name in ("transport_flow.csv", "device_utilization.csv", "from_to_bay.csv"):
        src = e1_dir / name
        if src.exists() and not (sample / name).exists():
            shutil.copyfile(src, sample / name)

    summary_src = OUTPUT_ROOT / "e3_global" / "e3_summary.json"
    if summary_src.exists() and not (sample / "summary.json").exists():
        try:
            data = json.loads(summary_src.read_text(encoding="utf-8"))
            data["run_id"] = "sample"
            data["order_file"] = "内置示例订单"
            (sample / "summary.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return sample


# ------------------------------------------------------------------ 推理 Tab
class InferenceTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=(16, 13))
        self.order_file: Path | None = None
        self.worker: InferenceWorker | None = None

        settings = ttk.Frame(self, style="Sidebar.TFrame", padding=13, width=268)
        settings.pack(side="left", fill="y", padx=(0, 12))
        settings.pack_propagate(False)

        ttk.Label(settings, text="订单文件", style="PanelHeading.TLabel").pack(anchor="w")
        ttk.Button(settings, text="上传订单文件", command=self.choose_order).pack(fill="x", pady=(8, 6))
        self.order_label = ttk.Label(settings, text="未选择文件", style="Panel.TLabel",
                                     foreground=MUTED, font=SMALL_FONT, wraplength=240)
        self.order_label.pack(fill="x")
        ttk.Separator(settings).pack(fill="x", pady=15)

        ttk.Label(settings, text="训练参数", style="PanelHeading.TLabel").pack(anchor="w")
        self.max_epochs = tk.StringVar(value="100")
        self.iter_per_epoch = tk.StringVar(value="200")
        self.lp_every = tk.StringVar(value="10")
        self.warm_start = tk.BooleanVar(value=True)
        fields = ttk.Frame(settings, style="Sidebar.TFrame", borderwidth=0)
        fields.pack(fill="x", pady=(7, 5))
        fields.columnconfigure(0, weight=1)
        for row, (label, var) in enumerate([
            ("max_epochs", self.max_epochs),
            ("iter_per_epoch", self.iter_per_epoch),
            ("lp_every", self.lp_every),
        ]):
            ttk.Label(fields, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(fields, textvariable=var, width=8).grid(row=row, column=1, sticky="e")
        ttk.Checkbutton(settings, text="warm-start（持久化 HiGHS 热求解）",
                        variable=self.warm_start).pack(anchor="w", pady=(3, 13))
        ttk.Button(settings, text="开始训练", command=self.start,
                   style="Accent.TButton").pack(fill="x")

        monitor = ttk.Frame(self, style="Card.TFrame", padding=14)
        monitor.pack(side="left", fill="both", expand=True)
        self.stage_label = ttk.Label(monitor, text="就绪", style="Status.TLabel", foreground=MUTED)
        self.stage_label.pack(anchor="w")
        self.progress = ttk.Progressbar(monitor, maximum=100)
        self.progress.pack(fill="x", pady=(8, 15))

        ttk.Label(monitor, text="训练日志", style="Log.TLabel").pack(anchor="w", pady=(0, 6))
        log_frame = ttk.Frame(monitor, style="Card.TFrame")
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=1, state="disabled", wrap="word", font=LOG_FONT,
                           background=PANEL, foreground=INK, borderwidth=0, highlightthickness=0,
                           padx=10, pady=9, spacing2=2, spacing3=2)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)


    def choose_order(self):
        path = filedialog.askopenfilename(
            title="选择订单文件",
            filetypes=[("订单文件", "*.txt"), ("所有文件", "*.*")])
        if path:
            self.order_file = Path(path)
            self.order_label.config(text=str(self.order_file), foreground=INK)

    def _log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        if self.order_file is None:
            messagebox.showwarning("提示", "请先上传订单文件")
            return
        try:
            max_epochs = int(self.max_epochs.get())
            iter_per_epoch = int(self.iter_per_epoch.get())
            lp_every = int(self.lp_every.get())
        except ValueError:
            messagebox.showerror("错误", "训练参数必须为整数")
            return

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress["value"] = 0
        self.stage_label.config(text="正在启动训练...", foreground=INK)

        self.worker = InferenceWorker(
            str(self.order_file), max_epochs, iter_per_epoch, lp_every,
            self.warm_start.get())
        self.worker.start()
        self.after(120, self._poll)

    def _poll(self):
        worker = self.worker
        if worker is None:
            return
        try:
            while True:
                msg = worker.queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, stage, pct, line = msg
                    if pct is not None:
                        self.progress["value"] = pct
                    self.stage_label.config(text=f"阶段 {stage}")
                    self._log(line)
                elif kind == "done":
                    _, result_dir, summary = msg
                    self.progress["value"] = 100
                    self.stage_label.config(text="训练完成", foreground=ACCENT)
                    self._log("训练完成，结果已保存到：" + result_dir)
                    if summary:
                        red = summary.get("cost_reduction_pct")
                        self._log("成本降幅：{:.2f}%".format(red) if red is not None
                                  else "成本降幅：见 summary.json")
                        self._log("移动设备：{}".format(summary.get("moved")))
                    messagebox.showinfo("完成", "训练完成，结果已保存到：\n" + result_dir)
                    self.worker = None
                    return
                elif kind == "error":
                    _, text = msg
                    self.stage_label.config(text="训练失败", foreground=ERROR)
                    self._log("[错误] " + text)
                    messagebox.showerror("错误", text)
                    self.worker = None
                    return
        except Exception:
            pass
        if worker.is_alive():
            self.after(120, self._poll)


# ------------------------------------------------------------------ 加载 Tab
class ViewTab(ttk.Frame):
    def __init__(self, master, initial_dir: Path | None):
        super().__init__(master, padding=(16, 13))
        self.result_dir: Path | None = initial_dir
        self.comparison_window: tk.Toplevel | None = None

        top = ttk.Frame(self, style="Card.TFrame", padding=(10, 7))
        top.pack(fill="x", pady=(0, 11))
        top.columnconfigure(1, weight=1, minsize=120)
        ttk.Button(top, text="选择结果目录", command=self.choose_dir).grid(row=0, column=0)
        self.dir_label = ttk.Label(top, text=str(initial_dir or "未选择"),
                                   style="File.TLabel", wraplength=630)
        self.dir_label.grid(row=0, column=1, sticky="ew", padx=12)
        self.dir_label.bind("<Configure>", self._wrap_path)
        ttk.Button(top, text="加载", command=self.load, style="Accent.TButton").grid(row=0, column=2)
        self.compare_button = ttk.Button(top, text="对比", command=self.compare, state="disabled")
        self.compare_button.grid(row=0, column=3, padx=(8, 0))

        self.canvas = ResultCanvas(self)

        if initial_dir is not None and (initial_dir / "layout.csv").exists():
            self.load(initial_dir)

    def _wrap_path(self, event):
        width = max(120, event.width - 2)
        if int(self.dir_label.cget("wraplength")) != width:
            self.dir_label.configure(wraplength=width)


    def choose_dir(self):
        path = filedialog.askdirectory(title="选择结果目录")
        if path:
            self.load(Path(path))

    def load(self, path: Path | None = None):
        directory = Path(path) if path is not None else self.result_dir
        if directory is None:
            messagebox.showwarning("提示", "请先选择结果目录")
            return
        try:
            data = load_result(directory)
        except Exception as exc:
            messagebox.showerror("错误", f"加载失败：{exc}")
            return
        if not data["names"]:
            messagebox.showwarning("提示", "该目录下没有可展示的布局数据")
            return
        if self.comparison_window and self.comparison_window.winfo_exists():
            self.comparison_window.destroy()
        self.result_dir = directory
        self.dir_label.config(text=str(directory), foreground=INK)
        self.canvas.set_data(data)
        self.compare_button.state(["!disabled"])

    def compare(self):
        if self.result_dir is None or self.canvas.data is None:
            return
        if self.comparison_window and self.comparison_window.winfo_exists():
            self.comparison_window.lift()
            return
        try:
            self.comparison_window = show_comparison(self, self.result_dir, self.canvas.data, SMAT_FILE)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("无法对比布局", str(exc), parent=self)


# ------------------------------------------------------------------ 接口说明 Tab
class DocsTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=(16, 13))
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        for title, text in [("用户接口", docs_text.USER_DOC),
                            ("开发者接口", docs_text.DEVELOPER_DOC)]:
            page = ttk.Frame(nb, style="Card.TFrame", padding=(10, 8))
            nb.add(page, text=title)
            box = tk.Text(page, wrap="word", font=UI_FONT, background=PAPER, foreground=INK,
                          borderwidth=0, highlightthickness=0, padx=16, pady=10, spacing2=3)
            scroll = ttk.Scrollbar(page, command=box.yview)
            box.configure(yscrollcommand=scroll.set)
            scroll.pack(side="right", fill="y")
            box.pack(fill="both", expand=True)
            box.tag_configure("title", font=(UI_FONT[0], 15, "bold"), foreground=INK, spacing3=8)
            box.tag_configure("section", font=HEADING_FONT, foreground=ACCENT, spacing1=7, spacing3=5)
            box.tag_configure("field", font=LOG_FONT, foreground=MUTED)
            for line in text.splitlines():
                if line.startswith("═"):
                    continue
                if line.startswith(("一、", "二、")):
                    tag = "title"
                elif line.startswith(("【", "◆")):
                    tag = "section"
                elif line.startswith("    "):
                    tag = "field"
                else:
                    tag = ""
                box.insert("end", line + "\n", (tag,) if tag else ())
            box.configure(state="disabled")



# ------------------------------------------------------------------ 主窗口
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("半导体工厂布局优化 · 桌面应用")
        self.geometry("1080x720")

        self.minsize(740, 520)
        apply_theme(self)

        mast = tk.Frame(self, background=NAV)
        mast.pack(fill="x")
        mark = tk.Canvas(mast, width=29, height=25, background=NAV,
                         highlightthickness=0, borderwidth=0)
        mark.pack(side="left", padx=(17, 9), pady=10)
        mark.create_rectangle(2, 2, 26, 22, outline="#ACD6A2", width=1)
        for x, height in ((8, 10), (14, 15), (20, 9)):
            mark.create_line(x, 18, x, 18 - height, fill="#ACD6A2", width=2)
        tk.Label(mast, text="晶圆厂布局优化", background=NAV, foreground=PAPER,
                 font=(UI_FONT[0], 14, "bold")).pack(side="left", pady=9)


        # 优先使用 result/heuristic（完整展示示例），缺失时回退 sample
        initial = RESULT_ROOT / "heuristic"
        if not (initial / "layout.csv").exists():
            initial = prepare_sample_result()

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=9, pady=(5, 7))
        nb.add(InferenceTab(nb), text="① 推理（训练）")
        nb.add(ViewTab(nb, initial), text="② 加载（展示）")
        nb.add(DocsTab(nb), text="③ 接口说明")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
