"""Command line and checkpoint-compatible PPO settings."""

import argparse
import math
from pathlib import Path

from .env import DEFAULT_INSTANCE


def configure_ppo_settings(args, saved):
    """Load the checkpoint architecture or select defaults for a fresh run."""
    for key, legacy, e3 in [
        ("actor_features", "local", "routing"),
        ("critic_features", "pooled", "transport"),
        ("minibatches_per_epoch", 1, 4),
    ]:
        supplied = getattr(args, key, None)
        if saved is not None:
            setting = saved.get(key, legacy)
            if args.resume and supplied is not None and supplied != setting:
                raise ValueError(f"Resume mismatch: {key} (checkpoint {setting}, requested {supplied})")
        elif supplied is not None:
            setting = supplied
        else:
            setting = e3 if args.slot_compatibility == "footprint" else legacy
        setattr(args, key, setting)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--instance", default=str(DEFAULT_INSTANCE))
    parser.add_argument(
        "--slot-compatibility",
        choices=["process-phase", "footprint"],
        help="Original four-part compatibility or E3 width/height-only slot classes",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--lp-backend", choices=["path-cuda", "path-cpu"], default="path-cuda")
    parser.add_argument("--lp-seed", type=Path, help="Sparse feasible flow used to initialize the LP solver")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--envs-per-rank", type=int, default=64)
    parser.add_argument("--lp-workers", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, choices=[1.0, 0.99], default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument(
        "--actor-features",
        choices=["local", "routing"],
        help="Use local candidate scores or include directed port distances (fresh E3 default)",
    )
    parser.add_argument(
        "--critic-features",
        choices=["pooled", "transport"],
        help="Use pooled encoders or add placed-pair transport features (the E3 default)",
    )
    parser.add_argument(
        "--minibatches-per-epoch",
        type=int,
        help="Adam updates per epoch (4 for fresh E3 runs, 1 for process-phase)",
    )
    parser.add_argument(
        "--decision-samples", type=int, default=256, help="Uniform positions/episode/epoch, 0 uses all"
    )
    parser.add_argument(
        "--gradient-batch",
        type=int,
        default=512,
        help="Per-rank backward microbatch limit within one optimizer step",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.0,
        help="Fit the CUDA backward microbatch under this fraction of device memory, up to --gradient-batch",
    )
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--completion-prior", action="store_true")
    parser.add_argument(
        "--eval-every", type=int, default=5, help="0 disables all in-training policy evaluation"
    )
    parser.add_argument(
        "--evaluate-checkpoint", help="Evaluate a saved PPO checkpoint in the separate --run directory"
    )
    parser.add_argument(
        "--diagnostic-samples", type=int, default=0, help="Separate fixed-seed pool at cycle0/final only"
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--policy-snapshot-every", type=int, default=0, help="Save a policy-only snapshot this often"
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.gamma != 1:
        parser.error("This controlled comparison fixes gamma=1")
    if (
        min(
            args.cycles,
            args.envs_per_rank,
            args.lp_workers,
            args.ppo_epochs,
            args.gradient_batch,
            args.checkpoint_every,
            args.cpu_threads,
        )
        < 1
    ):
        parser.error("Counts and intervals must be positive")
    if args.minibatches_per_epoch is not None and args.minibatches_per_epoch < 1:
        parser.error("--minibatches-per-epoch must be positive")
    if min(args.decision_samples, args.diagnostic_samples, args.eval_every, args.policy_snapshot_every) < 0:
        parser.error("Sample counts and optional intervals must be nonnegative")
    if (
        not 0 < args.clip_epsilon < 1
        or args.value_coef < 0
        or any(
            not math.isfinite(value) or value <= 0 for value in [args.lr, args.grad_clip, args.temperature]
        )
    ):
        parser.error("Invalid optimization parameters")
    if not 0 <= args.gpu_memory_fraction < 1 or (args.gpu_memory_fraction and args.device != "cuda"):
        parser.error("--gpu-memory-fraction requires CUDA and a value in [0, 1)")
    if args.lp_backend == "path-cuda" and args.device != "cuda":
        parser.error("path-cuda requires a CUDA policy device")
    args.lp_seed = str(
        (args.lp_seed or Path(args.instance).parent / "full_slots" / "seed_flow.npz").resolve()
    )
    return args
