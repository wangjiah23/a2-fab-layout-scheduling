"""Serve the team's layout workbench at / and its local SSH console at /rl/."""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import argparse
import html
import json
import re
import secrets
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory

import gradio as gr
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from connection import Connection, inspect_key
from results import result_package

WORKBENCH = Path(__file__).resolve().parents[2] / "sample_app/周报演示应用/晶圆厂布局工作台.html"
SESSIONS = {}
DEMO_LABEL = "演示策略 · 1 轮训练 / 2 个完整布局"
KINDS = {"setup": "部署", "train": "训练", "infer": "推理"}
STATES = {
    "queued": "排队中",
    "running": "运行中",
    "cleanup": "等待回收",
    "completed": "已完成",
    "failed": "失败",
    "stopped": "已停止",
}
AXES = ["训练轮次", "采样平均成本降幅（%）"]


def disconnect(key):
    session = SESSIONS.pop(key, None)
    if session is not None:
        session.close()


def require_session(key):
    if key not in SESSIONS:
        raise ValueError("请先连接 SSH 服务器。")
    return SESSIONS[key]


def guarded(fn):
    @wraps(fn)
    def run(*args):
        try:
            return fn(*args)
        except Exception as error:
            raise gr.Error(str(error) or "SSH 会话已结束，请重新连接。") from None

    return run


def connect(key, host, port, user, password, pin, request: gr.Request):
    disconnect(key)
    try:
        session = Connection(host, port, user, password, pin)
    except Exception as error:
        return "", "", f"连接失败。{html.escape(str(error))}", gr.Timer(active=False)
    token = secrets.token_hex(24)
    session.exports = {}
    session.browser = request.session_hash
    SESSIONS[token] = session
    return token, "", f"已连接 {user}@{host}（SSH {int(port)}）", gr.Timer(value=4, active=True)


def start_setup(key, progress=gr.Progress()):
    session = require_session(key)
    progress(0, desc="正在准备演示策略")
    session.install_demo(lambda sent, total: progress(sent / total, desc="正在上传演示策略"))
    progress(1, desc="已发送部署请求")
    result = session.call("start", kind="setup")
    return gr.Dropdown(choices=[result["id"]], value=result["id"])


def start_model(key, kind, device, ram, vram, workers, cycles, batch, checkpoint):
    fields = dict(
        kind=kind,
        device=device,
        ram_gib=ram,
        vram_gib=vram,
        workers=workers,
        cycles=cycles,
        batch=batch,
        checkpoint=checkpoint,
    )
    result = require_session(key).call("start", **fields)
    return gr.Dropdown(choices=[result["id"]], value=result["id"])


def stop(key, job):
    require_session(key).call("stop", job=job)
    return "已请求停止选中任务，正在回收进程和显存。"


def bundle_link(session, key, directory):
    if key not in session.exports:
        package = result_package(session.call("result", job=key))
        token = secrets.token_hex(20)
        (directory / (token + ".json")).write_text(
            json.dumps(package, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        m = package["metrics"]
        pct = (1 - m["optimized_cost"] / m["reference_cost"]) * 100
        session.exports[key] = (
            f"**终局运输成本 {m['optimized_cost']:,.2f}　降幅 {pct:.2f}%**　"
            f"[在工作台对照布局](/?rl_result={token})　[下载结果包](/rl-results/{token})"
        )
    return session.exports[key]


def refresh(key, selected, device, checkpoint, directory):
    empty = (
        gr.Dropdown(choices=[], value=None),
        gr.Dropdown(choices=[], value=None),
        gr.Dropdown(choices=[], value=None),
        "尚未连接。",
        [],
        "",
        gr.LinePlot(visible=False),
        "",
    )
    if not key:
        return empty
    session = require_session(key)
    data = session.call("status", job=selected)
    host, state = data["host"], data["state"]
    gpus = host["gpus"]
    devices = [(f"GPU {g['index']} · {g['name']}", g["uuid"]) for g in gpus] + [("CPU", "cpu")]
    default_device = max(gpus, key=lambda g: g["free"])["uuid"] if gpus else "cpu"
    selected_device = device if device in dict(devices).values() else default_device
    policies = [(DEMO_LABEL if p == "demo" else p, p) for p in data["checkpoints"]]
    chosen = checkpoint if checkpoint in data["checkpoints"] else (policies[0][1] if policies else None)
    table = [
        [g["index"], g["name"], f"{g['free'] / 1024:.1f} / {g['total'] / 1024:.1f}", g["util"]] for g in gpus
    ]
    info = (
        "环境已就绪" if data["ready"] else "环境待部署"
    ) + f"　可用内存 {host['ram_free'] / 1024:.1f} GiB　可用磁盘 {host['disk_free'] / 1024:.1f} GiB"
    if host["warning"]:
        info += "　" + html.escape(host["warning"])
    link = ""
    if state:
        info += f"\n\n{KINDS[state['kind']]} · **{STATES[state['status']]}**　{html.escape(state['detail'])}"
        if state["status"] == "running" and "rss" in state:
            info += f"　任务 RSS {state['rss'] / 1024:.2f} GiB　显存 {state['vram'] / 1024:.2f} GiB"
        if (state["kind"], state["status"]) == ("infer", "completed"):
            link = bundle_link(session, state["id"], directory)
        if (state["kind"], state["status"]) == ("train", "completed"):
            info += f"　推理页可选检查点 `{state['id']}/final`。"
    curve = data["curve"]
    plot = (
        gr.LinePlot(value=pd.DataFrame(curve, columns=AXES), x=AXES[0], y=AXES[1], visible=True)
        if curve
        else gr.LinePlot(visible=False)
    )
    jobs = data["jobs"]
    job = state.get("id")
    if job and job not in jobs:
        jobs = [job, *jobs]
    return (
        gr.Dropdown(choices=jobs, value=job),
        gr.Dropdown(choices=devices, value=selected_device),
        gr.Dropdown(choices=policies, value=chosen),
        info,
        table,
        data["log"],
        plot,
        link,
    )


def build_panel(directory):
    with gr.Blocks(title="RL 远程运行 · 晶圆厂布局工作台", analytics_enabled=False) as panel:
        session = gr.State("", time_to_live=7200, delete_callback=disconnect)
        tick = gr.Timer(4, active=False)
        gr.Markdown("[← 晶圆厂布局工作台](/)\n# RL 远程运行\nSMAT2022 LVHM_E · 全厂基准 · 1,068 台设备")
        with gr.Accordion("SSH 连接", open=True):
            with gr.Row():
                host = gr.Textbox(label="服务器 IP 或主机名", placeholder="输入服务器地址", scale=2)
                port = gr.Number(label="SSH 端口", value=2202, precision=0)
                user = gr.Textbox(label="SSH 用户名")
            with gr.Row():
                pin = gr.Textbox(
                    label="主机 SHA256 指纹",
                    info="首次连接请与管理员核对。已在本机 known_hosts 中登记的主机可留空。",
                    scale=3,
                )
                verify = gr.Button("读取主机指纹")
            password = gr.Textbox(
                label="SSH 密码", type="password", placeholder="使用本机 SSH 密钥或 agent 时留空"
            )
            with gr.Row():
                login = gr.Button("连接服务器", variant="primary")
                logout = gr.Button("断开 SSH")
            connection_status = gr.Markdown()
        with gr.Row():
            deploy = gr.Button("部署 / 校验环境")
            refresh_btn = gr.Button("刷新状态")
        status = gr.Markdown("尚未连接。")
        resources = gr.Dataframe(
            headers=["GPU", "型号", "可用 / 总显存（GiB）", "利用率（%）"],
            interactive=False,
            label="服务器 GPU",
        )
        with gr.Row():
            device = gr.Dropdown(label="计算设备", choices=[], interactive=True)
            vram = gr.Number(label="显存预算（GiB，0 自动）", value=0, precision=0)
            ram = gr.Number(label="内存预算（GiB，0 自动）", value=0, precision=0)
            workers = gr.Number(label="并行 LP 数（0 自动）", value=0, precision=0)
        with gr.Tabs():
            with gr.Tab("推理"):
                checkpoint = gr.Dropdown(label="策略检查点", choices=[], interactive=True)
                infer = gr.Button("启动推理", variant="primary")
            with gr.Tab("训练"):
                with gr.Row():
                    cycles = gr.Number(label="训练轮数", value=10, precision=0)
                    batch = gr.Number(label="每轮完整布局数", value=8, precision=0)
                train = gr.Button("启动训练", variant="primary")
        with gr.Row():
            job = gr.Dropdown(label="远程任务", choices=[], interactive=True, scale=3)
            halt = gr.Button("停止选中任务", variant="stop")
        gr.Markdown("关闭页面或断开 SSH 后，远程任务继续运行。重新连接同一账号可查看进度。")
        results = gr.Markdown()
        plot = gr.LinePlot(x=AXES[0], y=AXES[1], label="训练采样", visible=False, height=200)
        log = gr.Textbox(label="任务日志", lines=9, max_lines=15, interactive=False, autoscroll=True)
        view = [job, device, checkpoint, status, resources, log, plot, results]

        def render(key, j, d, p):
            return refresh(key, j, d, p, directory)

        inputs = [session, job, device, checkpoint]
        event = dict(
            concurrency_id="rl-ssh", concurrency_limit=1, api_visibility="private", show_progress="hidden"
        )

        def after(trigger):
            trigger.success(guarded(render), inputs, view, **event)

        verify.click(guarded(inspect_key), [host, port], pin, **event)
        connected = login.click(
            connect,
            [session, host, port, user, password, pin],
            [session, password, connection_status, tick],
            **event,
        )

        def first(key):
            return refresh(key, None, None, None, directory)

        connected.then(guarded(first), session, view, **event)

        def leave(key):
            disconnect(key)
            return "", "", "已断开 SSH。", gr.Timer(active=False)

        logout.click(leave, session, [session, password, connection_status, tick], **event).then(
            guarded(first), session, view, **event
        )
        after(deploy.click(guarded(start_setup), session, job, **{**event, "show_progress": "full"}))
        for button, kind in ((infer, "infer"), (train, "train")):

            def launch(key, d, r, v, w, c, b, p, kind=kind):
                return start_model(key, kind, d, r, v, w, c, b, p)

            after(
                button.click(
                    guarded(launch),
                    [session, device, ram, vram, workers, cycles, batch, checkpoint],
                    job,
                    **event,
                )
            )
        halt.click(guarded(stop), [session, job], status, **event)
        job.input(guarded(render), inputs, view, **event)

        def poll(key, j, d, p):
            try:
                # A timer must not reactivate a connection that the user just closed.
                return (*render(key, j, d, p), gr.skip(), gr.skip())
            except Exception as error:
                return (
                    *([gr.skip()] * len(view)),
                    gr.Timer(active=False),
                    f"自动刷新已暂停。{html.escape(str(error))} 请重新连接或手动刷新。",
                )

        tick.tick(poll, inputs, [*view, tick, connection_status], **event)
        refresh_btn.click(guarded(render), inputs, view, **event)

        def close_tab(request: gr.Request):
            for key, ssh in list(SESSIONS.items()):
                if ssh.browser == request.session_hash:
                    disconnect(key)

        panel.unload(close_tab)
    return panel


def create_app():
    storage = TemporaryDirectory(prefix="a2-rl-results-")
    directory = Path(storage.name)

    @asynccontextmanager
    async def lifespan(app):
        yield
        for key in list(SESSIONS):
            disconnect(key)
        storage.cleanup()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.get("/")
    def workbench():
        return FileResponse(WORKBENCH, media_type="text/html")

    @app.get("/rl-results/{token}")
    def exported(token: str):
        if not re.fullmatch("[a-f0-9]{40}", token) or not (path := directory / (token + ".json")).is_file():
            raise HTTPException(404)
        return FileResponse(
            path,
            filename="E3-PPO-fab-result.json",
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )

    return gr.mount_gradio_app(
        app,
        build_panel(directory),
        path="/rl",
        server_name="127.0.0.1",
        ssr_mode=False,
        run_history=False,
        enable_monitoring=False,
        footer_links=[],
        css=".gradio-container{max-width:1100px!important;margin:auto} .gradio-container h1{font-weight:650}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    uvicorn.run(create_app(), host="127.0.0.1", port=parser.parse_args().port)
