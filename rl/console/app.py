"""Hosted A2 layout workbench. Only the operator selects an installed checkpoint."""

import argparse
import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

RL = Path(__file__).resolve().parents[1]
WORKBENCH = RL.parent / "sample_app/周报演示应用/晶圆厂布局工作台.html"
Identifier = Annotated[str, Field(min_length=1, max_length=180)]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Demand(Input):
    part: Identifier
    route: Identifier
    demand_lots: Annotated[Number, Field(ge=0, le=1e9)]
    pieces_per_lot: int = Field(default=25, strict=True, ge=1, le=100000)


class Step(Input):
    route: Identifier
    step: int = Field(strict=True, ge=1)
    step_type: Literal["process", "delay"]
    stnfam: str = Field(default="", max_length=180)
    time_min_per_lot: Annotated[Number, Field(ge=0, le=1e7)]


class Position(Input):
    equipment_name: Identifier
    x: Number
    y: Number
    width: Annotated[Number, Field(gt=0)] | None = None
    height: Annotated[Number, Field(gt=0)] | None = None
    bay: str | None = Field(default=None, max_length=180)
    tool_group: str | None = Field(default=None, max_length=180)
    original_slot: int | None = Field(default=None, ge=0)
    new_slot: int | None = Field(default=None, ge=0)


class Scenario(Input):
    factory_id: Literal["SMAT2022-LVHM_E"] = "SMAT2022-LVHM_E"
    scope: Literal["full"] = "full"
    horizon_min: Annotated[Number, Field(gt=0, le=1e7)] = 10080
    effective_time_fraction: Annotated[Number, Field(gt=0, le=1)] = 0.85
    demands: list[Demand] = Field(min_length=1, max_length=100)
    new_routes: list[Step] = Field(default_factory=list, max_length=10000)
    initial_layout: list[Position] | None = Field(default=None, min_length=1068, max_length=1068)


class Job(Input):
    format: Literal["fab-job-v1"] = Field(alias="schema")
    method: Literal["rl"]
    scenario: Scenario


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def select_gpu():
    """Pick the idle compute GPU with most headroom, otherwise run on CPU."""
    try:

        def query(args):
            return subprocess.check_output(
                ["nvidia-smi", args, "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).splitlines()

        occupied = {s.strip() for s in query("--query-compute-apps=gpu_uuid")}
        candidates = []
        for row in query("--query-gpu=uuid,memory.free,utilization.gpu"):
            uuid, free, load = [f.strip() for f in row.split(",")]
            if uuid not in occupied and int(free) >= 3072 and int(load) < 20:
                candidates.append((int(free), uuid))
        return max(candidates)[1] if candidates else ""
    except (OSError, ValueError, subprocess.SubprocessError):
        return ""


async def run_job(job, python, checkpoint):
    write(job / "state.json", {"status": "running"})
    device = await asyncio.to_thread(select_gpu)
    paths = sorted(python.parent.parent.glob("lib/python*/site-packages/nvidia/*/lib"))
    libs = [*map(str, paths), os.environ.get("LD_LIBRARY_PATH", "")]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": device,
        "LD_LIBRARY_PATH": os.pathsep.join(filter(None, libs)),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [
        str(python),
        "-m",
        "smat_rl.infer_scene",
        "--job",
        str(job),
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cuda" if device else "cpu",
    ]
    with (job / "solver.log").open("ab") as log:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=RL, env=env, stdout=log, stderr=log, start_new_session=True
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=1200)
        finally:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
    if proc.returncode or not (job / "result.json").is_file():
        detail = (
            json.loads((job / "error.json").read_bytes())["detail"]
            if (job / "error.json").is_file()
            else "本次求解未取得可验证的结果。"
        )
        write(job / "state.json", dict(status="failed", detail=detail))
    else:
        write(job / "state.json", dict(status="completed"))


def build_app(state, python, checkpoint):
    state = state.resolve()
    queue = asyncio.Queue(maxsize=8)

    async def work():
        while True:
            job = await queue.get()
            try:
                await run_job(job, python, checkpoint)
            except asyncio.CancelledError:
                write(job / "state.json", dict(status="failed", detail="服务已停止，可重新提交。"))
                raise
            except Exception as exc:
                logging.exception("Solver job %s failed", job.name)
                detail = "本次求解超过 20 分钟。" if isinstance(exc, TimeoutError) else "求解服务暂时不可用。"
                write(job / "state.json", dict(status="failed", detail=detail))
            finally:
                queue.task_done()

    @asynccontextmanager
    async def lifespan(_):
        import fcntl

        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (state / "service.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for p in state.glob("*/state.json"):
                if json.loads(p.read_bytes())["status"] in ("queued", "running"):
                    write(p, dict(status="failed", detail="服务已重启，可重新提交。"))
            runner = asyncio.create_task(work())
            try:
                yield
            finally:
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)

    app = FastAPI(
        title="A2 晶圆厂布局工作台", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )

    @app.get("/")
    @app.get("/rl/")
    def home():
        return FileResponse(WORKBENCH, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/input-schema")
    def input_schema():
        return Job.model_json_schema(by_alias=True)

    @app.get("/api/templates/order.txt")
    def order_template():
        return FileResponse(WORKBENCH.parent / "导入示例/05_订单.txt", filename="order.txt")

    @app.post("/api/jobs", status_code=202)
    async def submit(request: Request):
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            raise HTTPException(415, "请提交 fab-job-v1 JSON。")
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > 2 * 1024**2:
                raise HTTPException(413, "场景任务超过 2 MiB。")
            raw.extend(chunk)
        try:
            task = Job.model_validate_json(bytes(raw))
        except ValidationError as exc:
            first = exc.errors(include_input=False)[0]
            field = ".".join(map(str, first["loc"]))
            raise HTTPException(422, f"场景字段 {field or 'JSON'} 格式错误：{first['msg']}") from None
        if queue.full() or shutil.disk_usage(state).free < 512 * 1024**2:
            raise HTTPException(503, "求解队列或磁盘暂时已满，请稍后提交。")
        job = state / secrets.token_hex(20)
        job.mkdir(mode=0o700)
        write(job / "request.json", task.model_dump(by_alias=True, exclude_none=True))
        write(job / "state.json", {"status": "queued"})
        queue.put_nowait(job)
        return {"id": job.name}

    def job_for(key):
        if not re.fullmatch(r"[0-9a-f]{40}", key) or not (state / key / "state.json").is_file():
            raise HTTPException(404, "找不到这次任务。")
        return state / key

    @app.get("/api/jobs/{key}")
    def status(key: str):
        job = job_for(key)
        view = json.loads((job / "state.json").read_bytes())
        if view["status"] == "running" and (job / "progress.json").is_file():
            view.update(json.loads((job / "progress.json").read_bytes()))
        return view

    @app.get("/api/jobs/{key}/{artifact}")
    def result(key: str, artifact: str):
        job = job_for(key)
        if artifact == "request.json":
            return FileResponse(job / artifact, headers={"Cache-Control": "private, no-store"})
        if artifact not in ("result.json", "simulation-input.zip"):
            raise HTTPException(404)
        if json.loads((job / "state.json").read_bytes())["status"] != "completed":
            raise HTTPException(409, "任务尚未完成。")
        return FileResponse(job / artifact, filename=artifact, headers={"Cache-Control": "private, no-store"})

    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--python", type=Path, default=RL / ".venv/bin/python")
    p.add_argument("--state", type=Path, default=Path.home() / ".local/share/a2-workbench/jobs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    opt = p.parse_args()
    if not opt.checkpoint.is_file() or not os.access(opt.python, os.X_OK):
        p.error("请指定已部署的 PPO 权重和 RL Python 环境。")
    os.umask(0o077)
    uvicorn.run(
        build_app(opt.state, Path(os.path.abspath(opt.python)), opt.checkpoint.resolve()),
        host=opt.host,
        port=opt.port,
        access_log=False,
    )
