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
from app.inference import InferenceWorker, RESULT_ROOT  # noqa: E402
from app.visualization import ResultCanvas, load_result  # noqa: E402

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
        super().__init__(master, padding=10)
        self.order_file: Path | None = None
        self.worker: InferenceWorker | None = None

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Button(top, text="上传订单文件", command=self.choose_order).pack(side="left")
        self.order_label = ttk.Label(top, text="未选择文件", foreground="gray")
        self.order_label.pack(side="left", padx=10)

        # 参数区
        params = ttk.LabelFrame(self, text="训练参数", padding=8)
        params.pack(fill="x", pady=8)
        self.max_epochs = tk.StringVar(value="100")
        self.iter_per_epoch = tk.StringVar(value="200")
        self.lp_every = tk.StringVar(value="10")
        self.warm_start = tk.BooleanVar(value=True)
        grid = ttk.Frame(params)
        grid.pack(fill="x")
        for i, (label, var) in enumerate([
            ("max_epochs", self.max_epochs),
            ("iter_per_epoch", self.iter_per_epoch),
            ("lp_every", self.lp_every),
        ]):
            ttk.Label(grid, text=label).grid(row=0, column=i * 2, sticky="e", padx=4)
            ttk.Entry(grid, textvariable=var, width=8).grid(row=0, column=i * 2 + 1, padx=4)
        ttk.Checkbutton(grid, text="warm-start（持久化 HiGHS 热求解）",
                        variable=self.warm_start).grid(row=0, column=6, padx=12)

        # 进度与日志
        ttk.Button(self, text="开始训练", command=self.start).pack(anchor="w", pady=4)
        self.progress = ttk.Progressbar(self, maximum=100, length=400)
        self.progress.pack(fill="x", pady=4)
        self.stage_label = ttk.Label(self, text="就绪", foreground="gray")
        self.stage_label.pack(anchor="w")

        log_frame = ttk.LabelFrame(self, text="训练日志", padding=4)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=14, state="disabled", wrap="none")
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
            self.order_label.config(text=str(self.order_file), foreground="black")

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
        self.stage_label.config(text="正在启动训练...", foreground="black")

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
                    self.stage_label.config(text="训练完成", foreground="green")
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
                    self.stage_label.config(text="训练失败", foreground="red")
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
        super().__init__(master, padding=10)
        self.result_dir: Path | None = initial_dir

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Button(top, text="选择结果目录", command=self.choose_dir).pack(side="left")
        self.dir_label = ttk.Label(top, text=str(initial_dir or "未选择"), foreground="gray")
        self.dir_label.pack(side="left", padx=10)
        ttk.Button(top, text="加载", command=self.load).pack(side="left", padx=4)

        self.canvas = ResultCanvas(self)

        if initial_dir is not None and (initial_dir / "layout.csv").exists():
            self.load(initial_dir)

    def choose_dir(self):
        path = filedialog.askdirectory(title="选择结果目录")
        if path:
            self.result_dir = Path(path)
            self.dir_label.config(text=str(self.result_dir), foreground="black")
            self.load()

    def load(self, path: Path | None = None):
        if path is not None:
            self.result_dir = Path(path)
        if self.result_dir is None:
            messagebox.showwarning("提示", "请先选择结果目录")
            return
        try:
            data = load_result(self.result_dir)
        except Exception as exc:
            messagebox.showerror("错误", f"加载失败：{exc}")
            return
        if not data["names"] and not data["util"][0].size:
            messagebox.showwarning("提示", "该目录下没有可展示的结果数据")
            return
        self.dir_label.config(text=str(self.result_dir), foreground="black")
        self.canvas.set_data(data)


# ------------------------------------------------------------------ 接口说明 Tab
class DocsTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        for title, text in [("用户接口", docs_text.USER_DOC),
                            ("开发者接口", docs_text.DEVELOPER_DOC)]:
            page = ttk.Frame(nb)
            nb.add(page, text=title)
            box = tk.Text(page, wrap="word")
            scroll = ttk.Scrollbar(page, command=box.yview)
            box.configure(yscrollcommand=scroll.set)
            scroll.pack(side="right", fill="y")
            box.pack(fill="both", expand=True)
            box.insert("1.0", text)
            box.configure(state="disabled")


# ------------------------------------------------------------------ 主窗口
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("半导体工厂布局优化 · 桌面应用")
        self.geometry("1080x720")

        # 优先使用 result/heuristic（完整展示示例），缺失时回退 sample
        initial = RESULT_ROOT / "heuristic"
        if not (initial / "layout.csv").exists():
            initial = prepare_sample_result()

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        nb.add(InferenceTab(nb), text="① 推理（训练）")
        nb.add(ViewTab(nb, initial), text="② 加载（展示）")
        nb.add(DocsTab(nb), text="③ 接口说明")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
