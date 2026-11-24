"""Train or evaluate PPO on the full original-slot factory layout."""

import csv
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .cli import configure_ppo_settings, parse_args
from .env import OriginalSlotEnv, configure_slot_compatibility
from .lp import OriginalSlotLP, SlotPathWorkers
from .model import SlotActorCritic
from .placement import GroupDecisionTargets, GroupSlotPlan
from .ppo import fit_cuda_gradient_batch, ppo_update
from .rollout import collect_policy
from .runtime import append_csv, collective_sum, json_save, torch_save


def certify(pool, layouts, run, cycle, rank, phase):
    evaluations = [pool.submit(layout) for layout in layouts]
    results = [evaluation.result() for evaluation in evaluations]
    bad = [
        {"row": i, "device_at_slot": layouts[i].tolist(), "result": item}
        for i, item in enumerate(results)
        if "error" in item
    ]
    if bad:
        json_save(run / f"lp_rejections_{phase}_rank{rank}_cycle{cycle}.json", bad)
    return results, len(bad)


def main(args):
    rank, local_rank, world = (
        int(os.environ.get(key, default))
        for key, default in [("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)]
    )
    evaluation_state = None
    if args.evaluate_checkpoint:
        if world != 1 or args.resume:
            raise ValueError(
                "Checkpoint evaluation requires a single process and a separate output directory"
            )
        evaluation_state = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        saved = evaluation_state["config"]
        args.slot_compatibility = saved.get("slot_compatibility", "process-phase")
        args.temperature = saved["temperature"]
        args.completion_prior = saved["completion_prior"]
        args.seed = saved["seed"]
        args.ppo_epochs = saved["ppo_epochs"]
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=45))
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if not args.resume and (run / "config.json").exists():
        raise ValueError("Choose a fresh run directory or --resume")
    state = evaluation_state
    if args.resume:
        state = torch.load(run / f"last_rank{rank}.pt", map_location="cpu", weights_only=False)
    if args.slot_compatibility is None:
        args.slot_compatibility = (
            state["config"].get("slot_compatibility", "process-phase") if state else "footprint"
        )
    configure_ppo_settings(args, state["config"] if state is not None else None)
    torch.manual_seed(args.seed)
    lp = OriginalSlotLP(args.instance)
    configure_slot_compatibility(lp, args.slot_compatibility)
    env = OriginalSlotEnv(lp, evaluate_terminal=False)
    plan = GroupSlotPlan(env)
    local_samples = args.envs_per_rank * min(
        args.decision_samples or plan.decision_count, plan.decision_count
    )
    if evaluation_state is None and not 1 <= args.minibatches_per_epoch <= local_samples:
        raise ValueError("Each PPO optimizer minibatch needs at least one decision per rank")
    targets = GroupDecisionTargets(env, plan, device)
    log_counts = np.full_like(plan.completion_counts, -np.inf)
    np.log(plan.completion_counts, out=log_counts, where=plan.completion_counts > 0)
    targets.log_completion_counts = torch.as_tensor(log_counts, dtype=torch.float32, device=device)
    actor = (
        SlotActorCritic(
            env,
            transport_critic=args.critic_features == "transport",
            routing_actor=args.actor_features == "routing",
        )
        .to(device)
        .eval()
    )
    transport_history = None
    if actor.placed_transport is not None and evaluation_state is None:
        transport_history = actor.distance.new_empty(
            (args.envs_per_rank, plan.decision_count, actor.placed_transport.width)
        )
    if args.gpu_memory_fraction and evaluation_state is None:
        args.gradient_batch = fit_cuda_gradient_batch(actor, env, plan, targets, args, device, rank, world)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    data_generator = torch.Generator(device=device).manual_seed(args.seed + 1009 * rank)
    gradient_generator = torch.Generator(device=device).manual_seed(args.seed + 1000003 + 1009 * rank)
    config = {
        **vars(args),
        "world_size": world,
        "global_episodes_per_cycle": world * args.envs_per_rank,
        "instance_fingerprint": lp.fingerprint,
        "reference_cost": lp.reference_cost,
        "machines": env.M,
        "decision_steps": plan.decision_count,
        "group_slots": True,
        "algorithm": "masked on-policy PPO-Clip",
        "reward": "only terminal 1-C/Cref; gamma=1",
        "critic_target": "MC terminal return at every decision for both lambdas",
        "advantage_normalization": "global fixed pre-update mean/std over nonforced decision states",
        "objective": "static decision mean clipped actor + value_coef times MC critic MSE; no entropy bonus",
        "optimizer_steps_per_cycle": args.ppo_epochs * args.minibatches_per_epoch,
        "actor_routing_width": actor.candidate_transport.feature_dim
        if actor.candidate_transport is not None
        else 0,
        "critic_transport_edges": actor.placed_transport.width if actor.placed_transport is not None else 0,
        "precision": "FP32 actor and critic, TF32 disabled",
        "diagnostic_schedule": "cycle0 and final cycle only, fixed independent sample seed"
        if args.eval_every
        else "off",
        "initialization": "seeded random actor, zero-output state critic, no demonstrations",
        "torch_version": torch.__version__,
    }
    start_cycle = episodes = training_lp = evaluation_lp = updates = 0
    best = float("inf")
    if evaluation_state is not None:
        if evaluation_state["config"]["instance_fingerprint"] != lp.fingerprint:
            raise ValueError("Checkpoint instance mismatch")
        actor.load_state_dict(evaluation_state["model"])
        episodes = evaluation_state.get("episodes", 0)
        training_lp = evaluation_state.get("training_lp_attempts", 0)
    if args.resume:
        if state["config"].get("slot_compatibility", "process-phase") != args.slot_compatibility:
            raise ValueError("Resume mismatch: slot_compatibility")
        for key in [
            "instance_fingerprint",
            "world_size",
            "envs_per_rank",
            "completion_prior",
            "temperature",
            "gae_lambda",
            "ppo_epochs",
            "decision_samples",
            "lr",
            "value_coef",
            "clip_epsilon",
        ]:
            if state["config"][key] != config[key]:
                raise ValueError(f"Resume mismatch: {key}")
        actor.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        data_generator.set_state(state["data_rng"].cpu())
        gradient_generator.set_state(state["gradient_rng"].cpu())
        start_cycle, episodes, training_lp, evaluation_lp, updates, best = (
            state[key]
            for key in [
                "cycle",
                "episodes",
                "training_lp_attempts",
                "evaluation_lp_attempts",
                "updates",
                "best_cost",
            ]
        )
        if world > 1:
            checkpoints = [None] * world
            dist.all_gather_object(checkpoints, (start_cycle, updates))
            if len(set(checkpoints)) != 1:
                raise ValueError("Rank checkpoints have different cycle/update counts")
    metrics_path = run / "metrics.csv"
    existing_fields = None
    if args.resume and rank == 0 and metrics_path.exists():
        with metrics_path.open(newline="") as handle:
            existing_fields = next(csv.reader(handle), None)
    ddp = (
        DDP(
            actor,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        if world > 1
        else actor
    )
    pool = SlotPathWorkers(
        lp, args.lp_seed, args.lp_workers, local_rank if args.lp_backend == "path-cuda" else None
    )
    started = time.perf_counter()

    def evaluate(cycle):
        nonlocal evaluation_lp, best
        if rank == 0:
            begin = time.perf_counter()
            actions, layouts, _ = collect_policy(
                actor,
                env,
                plan,
                targets,
                1,
                data_generator,
                args.temperature,
                greedy=True,
                completion_prior=args.completion_prior,
            )
            diagnostic_count = (
                args.diagnostic_samples if evaluation_state is not None or cycle in (0, args.cycles) else 0
            )
            if diagnostic_count:
                generator = torch.Generator(device=device).manual_seed(args.seed + 123456)
                _, sampled, _ = collect_policy(
                    actor,
                    env,
                    plan,
                    targets,
                    diagnostic_count,
                    generator,
                    args.temperature,
                    completion_prior=args.completion_prior,
                )
                layouts = np.concatenate([layouts, sampled])
            results, failures = certify(pool, layouts, run, cycle, rank, "evaluation")
            evaluation_lp += len(layouts)
            if failures:
                raise RuntimeError("An evaluation LP failed certification")
            result = results[0]
            row = dict(
                cycle=cycle,
                episodes=episodes,
                cost=result["cost"],
                improvement=result["improvement"],
                training_lp_attempts=training_lp,
                evaluation_lp_attempts=evaluation_lp,
                total_lp_attempts=training_lp + evaluation_lp,
                diagnostic_samples=diagnostic_count,
                sampled_mean_cost=float(np.mean([item["cost"] for item in results[1:]]))
                if diagnostic_count
                else "",
                sampled_best_cost=min(item["cost"] for item in results[1:]) if diagnostic_count else "",
                seconds=time.perf_counter() - begin,
            )
            append_csv(run / "evaluations.csv", row)
            json_save(run / f"evaluation_cycle{cycle}.json", dict(summary=row, results=results))
            if result["cost"] < best:
                best = float(result["cost"])
                if evaluation_state is None:
                    torch_save(
                        run / "best.pt",
                        dict(
                            model=actor.state_dict(), config=config, cycle=cycle, cost=best, updates=updates
                        ),
                    )
                json_save(
                    run / "best_policy_layout.json",
                    {
                        **result,
                        "actions": actions[0].tolist(),
                        "device_at_slot": layouts[0].tolist(),
                        "cycle": cycle,
                        "source": "greedy_PPO_policy",
                    },
                )
            print(json.dumps({"event": "evaluation", **row}), flush=True)
        if world > 1:
            values = torch.tensor([best, evaluation_lp], device=device, dtype=torch.float64)
            dist.broadcast(values, 0)
            best, evaluation_lp = float(values[0]), int(values[1])

    try:
        if evaluation_state is not None:
            json_save(run / "config.json", config)
            evaluate(evaluation_state["cycle"])
            return
        if rank == 0:
            json_save(run / "config.json", config)
            json_save(run / "status.json", dict(status="starting", **config))
        if args.eval_every and not args.resume:
            evaluate(0)
        for cycle in range(start_cycle + 1, args.cycles + 1):
            cycle_started = time.perf_counter()
            actions, layouts, old_log_probs, old_values, active = collect_policy(
                actor,
                env,
                plan,
                targets,
                args.envs_per_rank,
                data_generator,
                args.temperature,
                completion_prior=args.completion_prior,
                return_values=True,
                return_active=True,
                transport_history=transport_history,
            )
            rollout_seconds = time.perf_counter() - cycle_started
            lp_started = time.perf_counter()
            results, rejected = certify(pool, layouts, run, cycle, rank, "training")
            lp_seconds = time.perf_counter() - lp_started
            counts = collective_sum([len(layouts), len(layouts) - rejected], device, world)
            training_lp += int(counts[0])
            records = [
                dict(
                    episode=index,
                    action_sha256=hashlib.sha256(actions[index].tobytes()).hexdigest(),
                    layout_sha256=hashlib.sha256(layout.tobytes()).hexdigest(),
                    result=result,
                )
                for index, (layout, result) in enumerate(zip(layouts, results))
            ]
            json_save(
                run / f"trajectories_rank{rank}_cycle{cycle}.json",
                dict(
                    cycle=cycle,
                    rank=rank,
                    actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                    records=records,
                ),
            )
            if counts[0] != counts[1]:
                raise RuntimeError("Training LP failed certification; PPO update skipped on all ranks")
            rewards = torch.tensor([1 - item["cost"] / lp.reference_cost for item in results], device=device)
            histories = torch.as_tensor(actions, device=device)
            update_started = time.perf_counter()
            if device.type == "cuda" and cycle == start_cycle + 1:
                torch.cuda.reset_peak_memory_stats(device)
            diagnostics = ppo_update(
                ddp,
                optimizer,
                histories,
                old_log_probs,
                old_values,
                active,
                rewards,
                targets,
                gradient_generator,
                args,
                world,
                transport_history,
            )
            if device.type == "cuda" and cycle == start_cycle + 1:
                print(
                    json.dumps(
                        dict(
                            event="cuda_update_memory",
                            rank=rank,
                            cycle=cycle,
                            allocated_mib=torch.cuda.max_memory_allocated(device) // 2**20,
                            reserved_mib=torch.cuda.max_memory_reserved(device) // 2**20,
                        )
                    ),
                    flush=True,
                )
            update_seconds = time.perf_counter() - update_started
            updates += diagnostics["optimizer_steps"]
            episodes += int(counts[1])
            totals = collective_sum(
                [
                    float(rewards.sum()),
                    sum(item["cost"] for item in results),
                    rollout_seconds,
                    lp_seconds,
                    update_seconds,
                ],
                device,
                world,
            )
            if rank == 0:
                row = dict(
                    cycle=cycle,
                    episodes=episodes,
                    updates=updates,
                    attempted=int(counts[0]),
                    accepted=int(counts[1]),
                    training_lp_attempts=training_lp,
                    evaluation_lp_attempts=evaluation_lp,
                    mean_return=totals[0] / counts[1],
                    mean_cost=totals[1] / counts[1],
                    rollout_seconds=totals[2] / world,
                    lp_seconds=totals[3] / world,
                    update_seconds=totals[4] / world,
                    **{key: value for key, value in diagnostics.items() if key != "epochs"},
                )
                # Keep the existing CSV header on resume. JSON contains the full diagnostics.
                append_csv(metrics_path, {k: row.get(k) for k in existing_fields} if existing_fields else row)
                json_save(run / f"update_cycle{cycle}.json", diagnostics)
                print(json.dumps({"event": "cycle", **row}), flush=True)
            if args.eval_every and (cycle % args.eval_every == 0 or cycle == args.cycles):
                evaluate(cycle)
            if cycle % args.checkpoint_every == 0 or cycle == args.cycles:
                torch_save(
                    run / f"last_rank{rank}.pt",
                    dict(
                        model=actor.state_dict(),
                        optimizer=optimizer.state_dict(),
                        data_rng=data_generator.get_state(),
                        gradient_rng=gradient_generator.get_state(),
                        cycle=cycle,
                        episodes=episodes,
                        updates=updates,
                        training_lp_attempts=training_lp,
                        evaluation_lp_attempts=evaluation_lp,
                        best_cost=best,
                        config=config,
                    ),
                )
            if rank == 0:
                if cycle == args.cycles or (
                    args.policy_snapshot_every and cycle % args.policy_snapshot_every == 0
                ):
                    name = "final.pt" if cycle == args.cycles else f"policy_cycle{cycle}.pt"
                    torch_save(
                        run / name,
                        dict(
                            model=actor.state_dict(),
                            config=config,
                            cycle=cycle,
                            episodes=episodes,
                            updates=updates,
                            training_lp_attempts=training_lp,
                        ),
                    )
                json_save(
                    run / "status.json",
                    dict(
                        status="running",
                        cycle=cycle,
                        episodes=episodes,
                        updates=updates,
                        training_lp_attempts=training_lp,
                        evaluation_lp_attempts=evaluation_lp,
                        best_cost=best,
                        wall_seconds=time.perf_counter() - started,
                        cycle_seconds=time.perf_counter() - cycle_started,
                    ),
                )
            if world > 1:
                dist.barrier()
        if rank == 0:
            json_save(
                run / "status.json",
                dict(
                    status="completed",
                    cycles=args.cycles,
                    episodes=episodes,
                    updates=updates,
                    training_lp_attempts=training_lp,
                    evaluation_lp_attempts=evaluation_lp,
                    best_cost=best,
                    wall_seconds=time.perf_counter() - started,
                ),
            )
    except BaseException as error:
        failure = dict(status="failed", rank=rank, error=repr(error))
        json_save(run / f"failure_rank{rank}.json", failure)
        if rank == 0:
            json_save(run / "status.json", failure)
        raise
    finally:
        pool.shutdown()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
