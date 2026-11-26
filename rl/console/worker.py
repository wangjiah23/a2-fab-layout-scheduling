"""SSH entry point on the Linux GPU host. Only the Python standard library is needed."""

import csv
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path.home() / ".local/share/a2-fab-rl"
REPO = "https://github.com/wangjiah23/a2-fab-layout-scheduling.git"
REV = "b0d852ac722c55ef0baa2bdc471d77a7abd6ba92"
CODE = ROOT / "releases" / REV / "rl"
READY = CODE.parent / ".console-ready"
JOBS = ROOT / "jobs"
DEMO = "demo"
MIB = 1024**2
FINAL = {"completed", "failed", "stopped"}
ID = re.compile(r"[a-f0-9]{16}")


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def capture(argv, timeout=8):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=True).stdout.strip()


def mem_available():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(values["MemAvailable"].split()[0]) // 1024
    # Account for a lower cgroup-v2 limit on shared servers.
    cg = next((s[3:] for s in Path("/proc/self/cgroup").read_text().splitlines() if s.startswith("0::")), "")
    base = Path("/sys/fs/cgroup") / cg.lstrip("/")
    for parent in (base, *base.parents):
        cap = parent / "memory.max"
        if cap.is_file() and (raw := cap.read_text().strip()) != "max":
            available = min(available, (int(raw) - int((parent / "memory.current").read_text())) // MIB)
    return max(0, available)


def gpu_query(kind, fields):
    if not shutil.which("nvidia-smi"):
        return []
    output = capture(["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"])
    return [[s.strip() for s in row] for row in csv.reader(output.splitlines())]


def resources():
    gpus, warning = [], ""
    try:
        for index, key, name, total, free, util in gpu_query(
            "gpu", "index,uuid,name,memory.total,memory.free,utilization.gpu"
        ):
            gpus.append(
                dict(index=int(index), uuid=key, name=name, total=int(total), free=int(free), util=util)
            )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        warning = f"nvidia-smi: {exc}"
    return dict(
        gpus=gpus,
        ram_free=mem_available(),
        disk_free=shutil.disk_usage(ROOT).free // MIB,
        cpus=len(os.sched_getaffinity(0)),
        warning=warning,
    )


def stamp(pid):
    # PID and start time within a boot identify the process even after PID reuse.
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] != "Z":
            return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + fields[19]
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def job_dir(key):
    if not isinstance(key, str) or not ID.fullmatch(key) or not (JOBS / key / "state.json").is_file():
        raise ValueError("任务不存在。")
    return JOBS / key


def active_identity(identity):
    return bool(identity and identity[1] and stamp(identity[0]) == identity[1])


def group_alive(pgid):
    rows = capture(["ps", "-eo", "pgid=,stat="]).splitlines()
    return any(
        int(group) == pgid and not status.startswith("Z") for group, status in (row.split() for row in rows)
    )


def terminate_group(pid, proc=None):
    try:
        if not group_alive(pid):
            return
        os.killpg(pid, signal.SIGTERM)
        for _ in range(30):
            if proc is not None:
                proc.poll()
            if not group_alive(pid):
                return
            time.sleep(0.2)
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def reconcile(job):
    state = read(job / "state.json")
    if state["status"] not in FINAL and not active_identity(read(job / "owner.json")):
        # The worker may have committed its terminal state while we checked the PID.
        state = read(job / "state.json")
        if state["status"] in FINAL:
            return state
        child = state.get("child")
        if active_identity(child):
            try:
                if os.getpgid(child[0]) == child[0]:
                    terminate_group(child[0])
            except ProcessLookupError:
                pass
            except (OSError, subprocess.SubprocessError) as exc:
                state.update(status="cleanup", detail=f"回收任务进程组失败：{exc}")
                save(job / "state.json", state)
                return state
        state.update(status="failed", detail="远程管理进程已退出，请查看日志。", ended=time.time())
        save(job / "state.json", state)
    return state


def integer(args, key, low, high):
    n = args[key]
    if type(n) is not int or not low <= n <= high:
        raise ValueError(f"{key} 必须在 {low} 到 {high} 之间。")
    return n


def checkpoints():
    saved = list(JOBS.glob("*/output/final.pt"))
    saved.extend(p for p in JOBS.glob("*/output/last_rank0.pt") if not (p.parent / "final.pt").exists())
    saved.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    items = [p.parents[1].name + "/" + p.stem for p in saved]
    return items + ([DEMO] if (ROOT / "models/demo.pt").is_file() else [])


def checkpoint_path(choice):
    if choice not in checkpoints():
        raise ValueError("请选择已安装的演示策略或本工作台产生的检查点。")
    if choice == DEMO:
        return ROOT / "models/demo.pt"
    key, name = choice.split("/")
    return job_dir(key) / "output" / (name + ".pt")


def validate(args):
    kind = args.get("kind")
    if kind == "setup":
        return {"kind": kind}
    if kind not in ("train", "infer") or not READY.is_file():
        raise ValueError("请先完成环境部署。")
    host = resources()
    cpu = args.get("device") == "cpu"
    gpu = next((g for g in host["gpus"] if g["uuid"] == args.get("device")), None)
    if not cpu and not gpu:
        raise ValueError("所选 GPU 不可用，请刷新资源。")
    ram = integer(args, "ram_gib", 0, 4096) * 1024
    available = host["ram_free"]
    if available <= 0 or ram > available:
        raise ValueError(f"当前只有 {available} MiB 可用内存。")
    gpu_limit = integer(args, "vram_gib", 0, 256) * 1024
    budget = dict(
        kind=kind,
        device=args["device"],
        ram_mib=ram or max(1, int(available * 0.8)),
        workers=integer(args, "workers", 0, min(host["cpus"], 64))
        or min(host["cpus"], 2 if kind == "train" else 1),
    )
    if gpu:
        capacity = min(gpu["free"] - 512, int(0.96 * gpu["total"]))
        if capacity <= 0 or gpu_limit > capacity:
            raise ValueError(f"GPU {gpu['index']} 可用 {gpu['free']} MiB，请降低显存预算或换卡。")
        limit = gpu_limit or max(1, int(capacity * 0.85))
        budget.update(
            vram_mib=limit,
            allocator_fraction=limit / gpu["total"],
            fit_fraction=min(0.99, (gpu["total"] - gpu["free"] + limit) / gpu["total"]),
        )
    if kind == "train":
        budget.update(cycles=integer(args, "cycles", 1, 10000), batch=integer(args, "batch", 2, 256))
    else:
        budget["checkpoint"] = str(checkpoint_path(args.get("checkpoint")))
    return budget


def setup_steps():
    checkout = CODE.parent
    checkout.mkdir(parents=True, exist_ok=True)
    uv = str(ROOT / "bin/uv")
    python = str(CODE / ".venv/bin/python")
    if not os.access(uv, os.X_OK):
        installer = str(ROOT / "install-uv.sh")
        yield (
            "下载 uv 0.10.2",
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "2",
                "--output",
                installer,
                "https://astral.sh/uv/0.10.2/install.sh",
            ],
        )
        yield "安装用户级 uv", ["env", f"UV_UNMANAGED_INSTALL={ROOT / 'bin'}", "sh", installer]
    if not (checkout / ".git").is_dir():
        yield "初始化代码目录", ["git", "init", "--quiet", str(checkout)]
    if (
        not (checkout / ".git/HEAD").is_file()
        or subprocess.run(
            ["git", "-C", str(checkout), "cat-file", "-e", REV + "^{commit}"], capture_output=True
        ).returncode
    ):
        yield (
            "从 GitHub 下载 RL 代码和全厂数据",
            ["git", "-C", str(checkout), "fetch", "--progress", "--depth=1", REPO, REV],
        )
    yield "检出 RL " + REV[:7], ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", REV]
    yield (
        "按 uv.lock 安装 PyTorch、CuPy 和 LP 依赖",
        [uv, "sync", "--project", str(CODE), "--locked", "--python", "3.11", "--extra", "cuda"],
    )
    yield (
        "验证模型、定价内核和 CUDA",
        [
            python,
            "-c",
            """
import shutil, subprocess, torch, cupy, highspy
from smat_rl.lp.paths import _cpu_kernel
_cpu_kernel()
print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, '可见 GPU', torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    print('GPU kernel:', cupy.arange(4).sum().item())
elif shutil.which('nvidia-smi') and subprocess.check_output(['nvidia-smi', '-L'], text=True).strip():
    raise RuntimeError('检测到 NVIDIA GPU，但 PyTorch 无法使用 CUDA，请检查驱动。')
else:
    print('CPU 可用')
""",
        ],
    )


def model_command(options, out):
    infer = options["kind"] == "infer"
    gpu = options["device"] != "cpu"
    python = str(CODE / ".venv/bin/python")
    cmd = [python, "-m", "smat_rl.train"]
    if gpu:
        # Apply the same allocator cap to inference and training, before model allocation.
        cmd = [
            python,
            "-c",
            "import runpy,torch; torch.cuda.set_per_process_memory_fraction("
            + repr(options["allocator_fraction"])
            + "); runpy.run_module('smat_rl.train',run_name='__main__')",
        ]
    cmd += [
        "--run",
        str(out),
        "--device",
        "cuda" if gpu else "cpu",
        "--lp-backend",
        "path-cuda" if gpu else "path-cpu",
        "--lp-workers",
        str(options["workers"]),
        "--cpu-threads",
        "1",
    ]
    if infer:
        return cmd + ["--evaluate-checkpoint", options["checkpoint"]]
    cmd += [
        "--slot-compatibility",
        "footprint",
        "--completion-prior",
        "--eval-every",
        "0",
        "--cycles",
        str(options["cycles"]),
        "--envs-per-rank",
        str(options["batch"]),
    ]
    if gpu:
        cmd += ["--gpu-memory-fraction", str(options["fit_fraction"]), "--gradient-batch", "4096"]
    return cmd


def group_usage(pgid, gpu):
    rows = [line.split() for line in capture(["ps", "-eo", "pid=,pgid=,rss="]).splitlines()]
    members = {int(pid): int(rss) for pid, group, rss in rows if int(group) == pgid}
    ram = sum(members.values()) // 1024
    vram = 0
    if gpu != "cpu":
        for pid, key, memory in gpu_query("compute-apps", "pid,gpu_uuid,used_gpu_memory"):
            if pid.isdigit() and int(pid) in members and key == gpu:
                if not memory.isdigit():
                    raise ValueError("无法读取任务 GPU 显存。")
                vram += int(memory)
    return ram, vram


class Stopped(Exception):
    pass


def execute(job):
    options = read(job / "request.json")
    state = read(job / "state.json")
    proc = None

    def request_stop(*_):
        (job / "stop").touch()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        setup = options["kind"] == "setup"
        if not setup:
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
                env.pop(key, None)
            env["CUDA_VISIBLE_DEVICES"] = "" if options["device"] == "cpu" else options["device"]
            if "vram_mib" in options:
                env["CUPY_GPU_MEMORY_LIMIT"] = str(options["vram_mib"] * MIB)
        steps = (
            setup_steps()
            if setup
            else [
                (
                    "训练 PPO" if options["kind"] == "train" else "推理并求解终局 LP",
                    model_command(options, job / "output"),
                )
            ]
        )
        if setup:
            READY.unlink(missing_ok=True)
        for label, command in steps:
            if (job / "stop").exists():
                raise Stopped()
            state.update(status="running", detail=label)
            save(job / "state.json", state)
            print(label, flush=True)
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                env=env,
                cwd=CODE if CODE.is_dir() else ROOT,
                start_new_session=True,
            )
            state["child"] = [proc.pid, stamp(proc.pid)]
            save(job / "state.json", state)
            while proc.poll() is None:
                if (job / "stop").exists():
                    raise Stopped()
                if not setup:
                    rss, vram = group_usage(proc.pid, options["device"])
                    state.update(rss=rss, vram=vram)
                    save(job / "state.json", state)
                    if rss > options["ram_mib"] or vram > options.get("vram_mib", math.inf):
                        raise RuntimeError(f"超过任务资源预算，RSS {rss} MiB，显存 {vram} MiB")
                time.sleep(1.5)
            if (job / "stop").exists():
                raise Stopped()
            if proc.returncode:
                raise RuntimeError(f"{label} 退出码 {proc.returncode}，请查看日志。")
            state.pop("child", None)
            proc = None
        if setup:
            READY.touch()
        state.update(status="completed", detail="已完成")
    except Stopped:
        state.update(status="stopped", detail="已停止")
    except Exception as exc:
        state.update(status="failed", detail=str(exc))
        print(str(exc), flush=True)
    finally:
        if proc is not None:
            try:
                terminate_group(proc.pid, proc)
                proc.wait()
            except (OSError, subprocess.SubprocessError) as exc:
                state.update(status="cleanup", detail=f"回收任务进程组失败：{exc}")
                print(state["detail"], flush=True)
                save(job / "state.json", state)
                return
        state.pop("child", None)
        state.update(ended=time.time())
        save(job / "state.json", state)


def tail(path, size=12000):
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, f.seek(0, 2) - size))
        return f.read(size).decode(errors="replace").replace("\r", "\n")


def handle(request):
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    JOBS.mkdir(mode=0o700, exist_ok=True)
    with (ROOT / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pointer = ROOT / "latest.json"
        latest = read(pointer) if pointer.is_file() else None
        state = reconcile(job_dir(latest)) if latest else {}
        action = request.get("action")
        if action == "start":
            if latest and state["status"] not in FINAL:
                raise ValueError(f"任务 {latest} 正在运行。")
            config = validate(request)
            key = uuid.uuid4().hex[:16]
            job = JOBS / key
            job.mkdir(mode=0o700)
            save(job / "request.json", config)
            save(
                job / "state.json",
                dict(id=key, kind=config["kind"], status="queued", detail="已排队", started=time.time()),
            )
            with (job / "job.log").open("ab", buffering=0) as log:
                proc = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--work", key],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            save(job / "owner.json", [proc.pid, stamp(proc.pid)])
            save(pointer, key)
            return {"id": key}
        selected = request.get("job") or latest
        if action == "stop":
            if not request.get("job") or selected != latest or state.get("status") in FINAL:
                raise ValueError("所选任务已结束或已被新任务替换。")
            (job_dir(selected) / "stop").touch()
            return {"id": selected}
        if action == "result":
            job = job_dir(selected)
            s = read(job / "state.json")
            if (s["status"], s["kind"]) != ("completed", "infer"):
                raise ValueError("请选择一次已完成的推理。")
            return dict(
                layout=read(job / "output/best_policy_layout.json"), elapsed=s["ended"] - s["started"]
            )
        if action != "status":
            raise ValueError("未知操作。")
        job = job_dir(selected) if selected else None
        history = sorted(JOBS.glob("*/state.json"), key=lambda p: p.parent.stat().st_ctime, reverse=True)[:50]
        curve = []
        if job and (metrics := job / "output/metrics.csv").is_file():
            with metrics.open() as f:
                for row in csv.DictReader(f):
                    if row.get("mean_return"):
                        curve.append([int(row["cycle"]), round(100 * float(row["mean_return"]), 4)])
        return dict(
            ready=READY.is_file(),
            host=resources(),
            latest=latest,
            jobs=[p.parent.name for p in history],
            state=read(job / "state.json") if job else {},
            checkpoints=checkpoints(),
            curve=curve,
            log=tail(job / "job.log") if job else "",
        )


if __name__ == "__main__":
    os.umask(0o077)
    if sys.argv[1:] and sys.argv[1] == "--work":
        execute(job_dir(sys.argv[2]))
    else:
        try:
            print(json.dumps(handle(json.load(sys.stdin)), ensure_ascii=False))
        except Exception as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False))
            sys.exit(1)
