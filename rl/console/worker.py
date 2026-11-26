"""SSH entry point on the Linux GPU host. Only the Python standard library is needed."""

import csv
import importlib.util
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

# SFTP installs this file under <selected root>/helpers/. Detached children
# execute the same file, so a reconnect and a background run use the same root.
ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_LOCK = Path.home() / ".local/state/a2-fab-rl/active.lock"
REPO = "https://github.com/wangjiah23/a2-fab-layout-scheduling.git"
REV = "b0d852ac722c55ef0baa2bdc471d77a7abd6ba92"
PYTHON_VERSION = "3.11"
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


def cgroup_dirs():
    root = Path("/sys/fs/cgroup")
    group = next(
        (s[3:] for s in Path("/proc/self/cgroup").read_text().splitlines() if s.startswith("0::")), ""
    )
    relative = Path(group.lstrip("/"))
    if ".." in relative.parts:
        return [root]
    return [root / parent for parent in (relative, *relative.parents)]


def mem_available(groups):
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(values["MemAvailable"].split()[0]) // 1024
    # Account for a lower cgroup-v2 limit on shared servers.
    for parent in groups:
        cap = parent / "memory.max"
        if cap.is_file() and (raw := cap.read_text().strip()) != "max":
            available = min(available, (int(raw) - int((parent / "memory.current").read_text())) // MIB)
    return max(0, available)


def cpu_ticks(affinity):
    idle = total = 0
    for row in Path("/proc/stat").read_text().splitlines():
        label, *fields = row.split()
        if label.startswith("cpu") and label[3:].isdigit() and int(label[3:]) in affinity:
            ticks = list(map(int, fields[:8]))  # Guest time is already included in user time.
            idle += ticks[3]
            total += sum(ticks)
    return idle, total


def cpu_capacity(groups):
    affinity = os.sched_getaffinity(0)
    capacity = len(affinity)
    quotas = {}
    for parent in groups:
        path = parent / "cpu.max"
        if path.is_file():
            quota, period = path.read_text().split()
            if quota != "max":
                cores = int(quota) / int(period)
                capacity = min(capacity, max(1, math.floor(cores)))
                quotas[parent / "cpu.stat"] = cores

    def usage(path):
        return int(dict(row.split() for row in path.read_text().splitlines())["usage_usec"])

    before_usage = {path: usage(path) for path in quotas}
    started = time.monotonic()
    before_idle, before_total = cpu_ticks(affinity)
    time.sleep(0.12)
    idle, total = cpu_ticks(affinity)
    free = len(affinity) * (idle - before_idle) / max(1, total - before_total)
    elapsed_us = (time.monotonic() - started) * 1e6
    for path, cores in quotas.items():
        free = min(free, cores - (usage(path) - before_usage[path]) / max(1, elapsed_us))
    return capacity, max(1, min(capacity, round(free)))


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
    groups = cgroup_dirs()
    cpus, idle_cpus = cpu_capacity(groups)
    return dict(
        gpus=gpus,
        ram_free=mem_available(groups),
        disk_free=shutil.disk_usage(ROOT).free // MIB,
        cpus=cpus,
        idle_cpus=idle_cpus,
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


def owns_child_group(job, child):
    if not child:
        return False
    try:
        if active_identity(child):
            return os.getpgid(child[0]) == child[0]
    except ProcessLookupError:
        pass
    # A torchrun launcher can disappear before its ranks. Only reclaim a surviving
    # group when a member still carries the job marker inherited at launch.
    marker = os.fsencode("A2_FAB_JOB=" + str(job))
    for row in capture(["ps", "-eo", "pid=,pgid="]).splitlines():
        pid, group = row.split()
        if int(group) != child[0]:
            continue
        try:
            if marker in Path("/proc", pid, "environ").read_bytes().split(b"\0"):
                return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return False


def reconcile(job):
    state = read(job / "state.json")
    if state["status"] in FINAL:
        return state
    try:
        owner = read(job / "owner.json")
    except FileNotFoundError:
        owner = None
    if not active_identity(owner):
        state = read(job / "state.json")
        if state["status"] in FINAL:
            return state
        child = state.get("child")
        try:
            if owns_child_group(job, child):
                terminate_group(child[0])
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
    models = sorted(
        "models/" + p.name
        for p in (ROOT / "models").glob("*.pt")
        if p.name != "demo.pt" and p.is_file() and not p.is_symlink()
    )
    return items + models + ([DEMO] if (ROOT / "models/demo.pt").is_file() else [])


def checkpoint_path(choice):
    if choice not in checkpoints():
        raise ValueError("请在当前部署目录的策略列表中选择检查点。")
    if choice == DEMO:
        return ROOT / "models/demo.pt"
    if choice.startswith("models/"):
        return ROOT / choice
    key, name = choice.split("/")
    return job_dir(key) / "output" / (name + ".pt")


def model_settings(args, host):
    kind = args.get("kind")
    devices = args.get("devices", [args.get("device")])
    if (
        not isinstance(devices, list)
        or not devices
        or not all(isinstance(d, str) for d in devices)
        or len(devices) != len(set(devices))
        or (len(devices) > 1 and (kind != "train" or "cpu" in devices))
    ):
        raise ValueError("推理请选择一个设备，训练可选 CPU 或多张不同的 GPU。")
    world = len(devices)
    gpu_by_id = {g["uuid"]: g for g in host["gpus"]}
    gpu = devices != ["cpu"]
    if gpu and any(d not in gpu_by_id for d in devices):
        raise ValueError("所选 GPU 不可用，请刷新资源。")
    ram = integer(args, "ram_gib", 0, 4096) * 1024
    available = host["ram_free"]
    if available <= 0 or ram > available:
        raise ValueError(f"当前可分配内存为 {available} MiB。")
    gpu_limit = integer(args, "vram_gib", 0, 256) * 1024
    workers = integer(args, "workers", 0, min(host["cpus"], 64))
    budget = dict(
        kind=kind,
        device=devices[0],
        devices=devices,
        ram_mib=ram or max(1, min(int(available * 0.8), 6144 if kind == "infer" else available)),
        workers=workers or 1,
    )
    if gpu:
        cards = [gpu_by_id[d] for d in devices]
        capacity = min(min(g["free"] - 512, int(0.96 * g["total"])) for g in cards)
        if capacity <= 0 or gpu_limit > capacity:
            raise ValueError("所选 GPU 的可用显存不足，请降低每卡显存预算或换卡。")
        limit = gpu_limit or max(1, int(capacity * 0.85))
        fractions = [limit / g["total"] for g in cards]
        fit = [min(0.99, (g["total"] - g["free"] + limit) / g["total"]) for g in cards]
        budget.update(
            vram_mib=limit,
            allocator_fraction=fractions[0],
            fit_fraction=fit[0],
            allocator_fractions=fractions,
            fit_fractions=fit,
        )
    if kind == "train":
        batch = integer(args, "batch", 0, 256 * world)
        if batch and (batch < 2 or batch % world):
            raise ValueError(f"每轮总布局数须至少为 2，且可被设备数 {world} 整除。")
        idle = host.get("idle_cpus", host["cpus"])
        lp_slots = min(
            64, max(1, idle - max(1, idle // 12)), max(1, (budget["ram_mib"] - 1792 * world) // 256)
        )
        if world > host["cpus"] or (workers and workers < world):
            raise ValueError("每个训练进程至少需要一个 CPU 和一个 LP 工作线程。")
        if world > 1 and budget["ram_mib"] < world * 2048:
            raise ValueError("多卡训练的主存预算不足，每卡至少预留 2 GiB。")
        if not batch:
            tiers = ((4, 2048), (8, 3584), (16, 5632), (32, 9728), (64, 18432))
            cap = max((n for n, mib in tiers if budget["vram_mib"] >= mib), default=2) if gpu else 8
            cap = min(cap, 4 * max(1, min(workers or lp_slots, lp_slots) // world))
            batch = (1 << (cap.bit_length() - 1)) * world
        budget.update(
            cycles=integer(args, "cycles", 1, 10000),
            batch=batch,
            workers=workers or max(world, min(batch, lp_slots)),
            auto_batch=args["batch"] == 0,
            auto_workers=workers == 0,
        )
    return budget


def validate(args, host=None):
    kind = args.get("kind")
    if kind == "setup":
        return {"kind": kind}
    if kind not in ("train", "infer") or not READY.is_file():
        raise ValueError("请先完成环境部署。")
    settings = model_settings(args, resources() if host is None else host)
    if kind == "infer":
        settings["checkpoint"] = str(checkpoint_path(args.get("checkpoint")))
    return settings


def setup_steps():
    checkout = CODE.parent
    checkout.mkdir(parents=True, exist_ok=True)
    uv = str(ROOT / "bin/uv")
    python = str(CODE / ".venv/bin/python")
    if not os.access(uv, os.X_OK):
        if importlib.util.find_spec("pip"):
            bootstrap = ROOT / "cache/uv-bootstrap"
            yield (
                "从 PyPI 下载 uv 0.10.2",
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--only-binary=:all:",
                    "--upgrade",
                    "--target",
                    str(bootstrap),
                    "uv==0.10.2",
                ],
            )
            yield "安装用户级 uv", ["install", "-D", "-m", "700", str(bootstrap / "bin/uv"), uv]
        else:
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
        bundle = ROOT / "cache" / (REV + ".bundle")
        fetch = ["git", "-C", str(checkout), "fetch", "--progress"]
        if bundle.is_file():
            yield "从已传入的 Git 提交包检出代码和全厂数据", [*fetch, str(bundle), "refs/heads/release"]
        else:
            yield "从 GitHub 下载 RL 代码和全厂数据", [*fetch, "--depth=1", REPO, REV]
    yield "检出 RL " + REV[:7], ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", REV]
    yield f"准备 Python {PYTHON_VERSION}", [uv, "python", "install", PYTHON_VERSION]
    yield (
        "按 uv.lock 安装 PyTorch、CuPy 和 LP 依赖",
        [uv, "sync", "--project", str(CODE), "--locked", "--python", PYTHON_VERSION, "--extra", "cuda"],
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
    devices = options.get("devices", [options["device"]])
    world = len(devices)
    cmd = [python, "-m", "smat_rl.train"]
    if gpu:
        # --run-path keeps spawned ranks in the supervised launcher's process group.
        cmd = [python]
        if world > 1:
            cmd += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={world}",
                "--start-method=spawn",
                "--run-path",
                "--",
            ]
        cmd += [str(Path(__file__).resolve()), "--rank", str(out.parent / "request.json")]
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
        str(options["batch"] // world),
    ]
    if gpu:
        cmd += ["--gradient-batch", "4096"]
    else:
        cmd += ["--gradient-batch", "8"]
    return cmd


def run_rank(request_path, argv):
    import runpy

    import torch

    options = read(Path(request_path))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    devices = options.get("devices", [options["device"]])
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(options["allocator_fractions"][rank], rank)
    workers, extra = divmod(options["workers"], len(devices))
    argv[argv.index("--lp-workers") + 1] = str(workers + int(rank < extra))
    if options["kind"] == "train":
        argv += ["--gpu-memory-fraction", str(options["fit_fractions"][rank])]
    # The uploaded rank script lives in helpers/, separate from the pinned RL package.
    sys.path.insert(0, str(CODE))
    sys.argv = ["smat_rl.train", *argv]
    runpy.run_module("smat_rl.train", run_name="__main__")


def group_usage(pgid, devices):
    rows = [line.split() for line in capture(["ps", "-eo", "pid=,pgid=,rss="]).splitlines()]
    members = {int(pid): int(rss) for pid, group, rss in rows if int(group) == pgid}
    ram = sum(members.values()) // 1024
    usage = {key: 0 for key in devices if key != "cpu"}
    if usage:
        for pid, key, memory in gpu_query("compute-apps", "pid,gpu_uuid,used_gpu_memory"):
            if pid.isdigit() and int(pid) in members and key in usage:
                if not memory.isdigit():
                    raise ValueError("无法读取任务 GPU 显存。")
                usage[key] += int(memory)
    return ram, usage


class Stopped(Exception):
    pass


def execute(job, account_fd=None):
    options = read(job / "request.json")
    state = read(job / "state.json")
    proc = None
    if account_fd is not None:
        import fcntl
        from threading import Thread

        try:
            fcntl.flock(account_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            # A disjoint inference may join a legacy job already holding the exclusive lock.
            Thread(target=fcntl.flock, args=(account_fd, fcntl.LOCK_SH), daemon=True).start()

    def request_stop(*_):
        (job / "stop").touch()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        # CuPy loads NVRTC by soname. The locked CUDA wheels provide it even
        # on driver-only hosts where the system linker has no CUDA search path.
        nvidia = CODE / f".venv/lib/python{PYTHON_VERSION}/site-packages/nvidia"
        cuda_libs = [str(nvidia / package / "lib") for package in ("cuda_nvrtc", "cuda_runtime")]
        if os.environ.get("LD_LIBRARY_PATH"):
            cuda_libs.append(os.environ["LD_LIBRARY_PATH"])
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "A2_FAB_JOB": str(job),
            "GIT_TERMINAL_PROMPT": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "UV_CACHE_DIR": str(ROOT / "cache/uv"),
            "UV_PYTHON_INSTALL_DIR": str(ROOT / "python"),
            "UV_PROJECT_ENVIRONMENT": str(CODE / ".venv"),
            "PIP_CACHE_DIR": str(ROOT / "cache/pip"),
            "UV_PYTHON_BIN_DIR": str(ROOT / "bin"),
            "CUPY_CACHE_DIR": str(ROOT / "cache/cupy"),
            "LD_LIBRARY_PATH": os.pathsep.join(cuda_libs),
        }
        setup = options["kind"] == "setup"
        if options["kind"] == "train":
            print(f"每轮 {options['batch']} 个完整布局，{options['workers']} 个并行 LP", flush=True)
        devices = options.get("devices", [options.get("device")])
        if setup:
            gpus = resources()["gpus"]
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = max(gpus, key=lambda g: g["free"])["uuid"]
        else:
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
                env.pop(key, None)
            env["CUDA_VISIBLE_DEVICES"] = "" if options["device"] == "cpu" else ",".join(devices)
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
                pass_fds=() if account_fd is None else (account_fd,),
            )
            state["child"] = [proc.pid, stamp(proc.pid)]
            save(job / "state.json", state)
            while proc.poll() is None:
                if (job / "stop").exists():
                    raise Stopped()
                if not setup:
                    rss, per_gpu = group_usage(proc.pid, devices)
                    vram = sum(per_gpu.values())
                    state["vram_by_gpu"] = per_gpu
                    state.update(rss=rss, vram=vram, peak_vram=max(state.get("peak_vram", 0), vram))
                    save(job / "state.json", state)
                    if rss > options["ram_mib"] or max(per_gpu.values(), default=0) > options.get(
                        "vram_mib", math.inf
                    ):
                        raise RuntimeError(f"超过任务资源预算，RSS {rss} MiB，显存 {vram} MiB")
                time.sleep(1.5)
            if (job / "stop").exists():
                raise Stopped()
            if proc.returncode:
                raise RuntimeError(f"{label} 退出码 {proc.returncode}，请查看日志。")
            # A successful launcher may exit before its subprocesses release the GPU.
            terminate_group(proc.pid, proc)
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
        if account_fd is not None:
            os.close(account_fd)
        state.update(ended=time.time())
        save(job / "state.json", state)


def tail(path, size=12000):
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, f.seek(0, 2) - size))
        return f.read(size).decode(errors="replace").replace("\r", "\n")


def active_jobs(root):
    for path in (root / "jobs").glob("*/state.json"):
        state = read(path)
        if state["status"] in FINAL:
            continue
        job = path.parent
        owner = job / "owner.json"
        if (owner.is_file() and active_identity(read(owner))) or owns_child_group(job, state.get("child")):
            yield job.name, read(job / "request.json"), state


def available_for_jobs(host, active):
    available = dict(host)
    assigned = {}
    for job_id, cfg, state in active:
        for device in cfg.get("devices", [cfg.get("device")]):
            assigned[device] = job_id
        # Resident pages are already excluded by the OS's MemAvailable.
        available["ram_free"] -= max(0, cfg.get("ram_mib", 0) - state.get("rss", 0))
        available["cpus"] -= cfg.get("workers", 1)
    available["idle_cpus"] = min(available["cpus"], host.get("idle_cpus", host["cpus"]))
    return available, assigned


def start_job(request):
    import fcntl

    # Serialize admission across deployment directories, including the interval before CUDA allocates.
    ACCOUNT_LOCK.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (ACCOUNT_LOCK.parent / "dispatch.lock").open("a") as admission, ACCOUNT_LOCK.open("a+") as slot:
        fcntl.flock(admission, fcntl.LOCK_EX)
        try:
            fcntl.flock(slot, fcntl.LOCK_SH | fcntl.LOCK_NB)
            legacy = False
        except BlockingIOError:
            legacy = True
        slot.seek(0)
        previous = slot.read().strip()
        roots = {ROOT, Path.home() / ".local/share/a2-fab-rl"}
        if previous and Path(previous).is_absolute():
            roots.add(Path(previous))
        active = []
        for root in roots:
            jobs = list(active_jobs(root))
            if root != ROOT and jobs:
                raise ValueError(f"该账号在 {root} 有运行中的任务，请在同一目录分配设备。")
            active.extend(jobs)
        if legacy and (previous != str(ROOT) or not any("devices" not in cfg for _, cfg, _ in active)):
            raise ValueError("旧版任务仍在登记资源，请稍后刷新。")
        devices = set(request.get("devices") or [request.get("device")])
        host, assigned = available_for_jobs(resources(), active)
        for job_id, cfg, _ in active:
            if (
                request["kind"] == "setup"
                or cfg["kind"] == "setup"
                or any(assigned.get(d) == job_id for d in devices)
            ):
                raise ValueError(f"任务 {job_id} 正占用所选资源，请选择其他设备。")
        if active and host["cpus"] < 1:
            raise ValueError("运行中的任务已分配全部 LP 工作线程。")
        config = validate(request, host)
        if previous != str(ROOT):
            slot.seek(0)
            slot.truncate()
            slot.write(str(ROOT))
            slot.flush()
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
                [sys.executable, str(Path(__file__).resolve()), "--work", key, str(slot.fileno())],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                pass_fds=(slot.fileno(),),
            )
        save(job / "owner.json", [proc.pid, stamp(proc.pid)])
        save(ROOT / "latest.json", key)
        return {"id": key}


def handle(request):
    import fcntl

    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    JOBS.mkdir(mode=0o700, exist_ok=True)
    with (ROOT / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pointer = ROOT / "latest.json"
        latest = read(pointer) if pointer.is_file() else None
        state = reconcile(job_dir(latest)) if latest else {}
        action = request.get("action")
        if action == "start":
            return start_job(request)
        selected = request.get("job") or latest
        if action == "stop":
            if not request.get("job") or reconcile(job_dir(selected)).get("status") in FINAL:
                raise ValueError("所选任务已结束或已被新任务替换。")
            (job_dir(selected) / "stop").touch()
            return {"id": selected}
        if action == "result":
            job = job_dir(selected)
            s = read(job / "state.json")
            if (s["status"], s["kind"]) != ("completed", "infer"):
                raise ValueError("请选择一次已完成的推理。")
            config = job / "output/config.json"
            return dict(
                layout=read(job / "output/best_policy_layout.json"),
                elapsed=s["ended"] - s["started"],
                checkpoint=Path(read(job / "request.json").get("checkpoint", "")).name,
                compatibility=read(config).get("slot_compatibility", "process-phase")
                if config.exists()
                else "footprint",
            )
        if action != "status":
            raise ValueError("未知操作。")
        # A refresh queued in the browser can still carry a job from the previous directory.
        try:
            job = job_dir(selected) if selected else None
        except ValueError:
            job = job_dir(latest) if latest else None
        history = sorted(JOBS.glob("*/state.json"), key=lambda p: p.parent.stat().st_ctime, reverse=True)[:50]
        curve = []
        last_rate = None
        if job and (metrics := job / "output/metrics.csv").is_file():
            with metrics.open() as f:
                for row in csv.DictReader(f):
                    if row.get("mean_return"):
                        curve.append([int(row["cycle"]), round(100 * float(row["mean_return"]), 4)])
                        timing = [row.get(k) for k in ("rollout_seconds", "lp_seconds", "update_seconds")]
                        if row.get("accepted") and all(timing):
                            seconds = sum(map(float, timing))
                            if seconds > 0:
                                last_rate = round(60 * int(row["accepted"]) / seconds, 1)
        view = reconcile(job) if job and job.name != latest else state if job else {}
        if view and view["kind"] != "setup":
            config = read(job / "request.json")
            view["devices"] = config.get("devices", [config["device"]])
        if view.get("kind") == "train":
            view.update(batch=config["batch"], workers=config["workers"], layouts_per_minute=last_rate)
        if view.get("kind") == "train" and view["status"] == "running":
            cycles = config["cycles"]
            done = curve[-1][0] if curve else 0
            if done >= cycles:
                phase = "正在保存检查点"
            elif (job / f"output/trajectories_rank0_cycle{done + 1}.json").is_file():
                phase = "本轮终局 LP 已返回，正在更新策略"
            else:
                phase = "正在采样布局和求解终局 LP"
            view["detail"] = f"训练 PPO · 已完成 {done}/{cycles} 轮，{phase}"
        host = resources()
        available, assigned = available_for_jobs(host, active_jobs(ROOT))
        return dict(
            ready=READY.is_file(),
            host=host,
            available=available,
            assigned=assigned,
            latest=latest,
            jobs=[p.parent.name for p in history],
            job_labels={p.parent.name: [s["kind"], s["status"]] for p in history if (s := read(p))},
            state=view,
            checkpoints=checkpoints(),
            curve=curve,
            log=tail(job / "job.log") if job else "",
        )


if __name__ == "__main__":
    os.umask(0o077)
    if sys.argv[1:2] == ["--rank"]:
        run_rank(sys.argv[2], sys.argv[3:])
    elif sys.argv[1:] and sys.argv[1] == "--work":
        execute(job_dir(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        try:
            print(json.dumps(handle(json.load(sys.stdin)), ensure_ascii=False))
        except Exception as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False))
            sys.exit(1)
