"""Masked sampling on the canonical physical placement sequence."""

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .placement import ordered_slot_mask


def completion_bias(obs, candidates, targets):
    """Log number of canonical completions after each possible smallest slot."""
    placed = (targets.slot_ids < obs["step"][:, None]) & (obs["placement"] >= 0)
    occupied = torch.zeros_like(obs["placement"], dtype=torch.int32)
    occupied.scatter_add_(1, obs["placement"].long().clamp_min(0), placed.int())
    free = (occupied == 0) & (targets.classes == targets.sequence[obs["step"]][:, None])
    higher = free.flip(1).cumsum(1).flip(1) - free.long()
    count = higher.gather(1, candidates.clamp_min(0))
    remaining = targets.after[obs["step"]][:, None]
    return targets.log_completion_counts[remaining, count]


def policy_distribution(
    model, obs, candidates, temperature, targets=None, completion_prior=False, return_value=False
):
    legal = (candidates >= 0) & obs["mask"].gather(1, candidates.clamp_min(0)).bool()
    if return_value:
        scores, values = model(obs, candidate_slots=candidates.clamp_min(0), return_value=True)
    else:
        scores = model(obs, candidate_slots=candidates.clamp_min(0))
    logits = scores.float() / temperature
    if completion_prior:
        logits = logits + completion_bias(obs, candidates, targets)
    logits = logits.masked_fill(~legal, -torch.inf)
    distribution = torch.distributions.Categorical(logits=logits, validate_args=False)
    return distribution, legal, values if return_value else logits


@torch.inference_mode()
def collect_policy(
    model,
    env,
    plan,
    targets,
    batch_size,
    generator,
    temperature,
    greedy=False,
    completion_prior=False,
    return_values=False,
    return_active=False,
    transport_history=None,
):
    """Place every machine, automatically filling slots with only one possible group."""
    device = targets.times.device
    network = model.module if isinstance(model, DDP) else model
    transport = getattr(network, "placed_transport", None) if return_values else None
    if transport_history is not None:
        if (
            transport is None
            or transport_history.shape != (batch_size, plan.decision_count, transport.width)
            or transport_history.device != device
            or transport_history.dtype != network.distance.dtype
        ):
            raise ValueError("Transport history must match the rollout and the transport critic")
    if transport is not None:
        totals = network.distance.new_zeros((batch_size, transport.width))
    placement = torch.full((batch_size, env.M), -1, dtype=torch.int16, device=device)
    occupied = torch.zeros((batch_size, env.C), dtype=torch.bool, device=device)
    compatibility = targets.sequence[:, None] == targets.classes[None]
    log_probs = torch.zeros((batch_size, plan.decision_count), device=device)
    if return_values:
        state_values = torch.zeros_like(log_probs)
    if return_active:
        active_decisions = torch.empty_like(log_probs, dtype=torch.bool)
    decision = 0
    for step in range(env.M):
        legal = compatibility[step] & ~occupied
        if not plan.is_decision[step]:
            chosen = legal.to(torch.int8).argmax(1, keepdim=True)
        else:
            previous = int(plan.same_type_prev[step])
            previous_slot = placement[:, previous : previous + 1] if previous >= 0 else -1
            legal, _ = ordered_slot_mask(
                legal, targets.slot_ids, previous_slot, int(plan.same_type_after[step])
            )
            obs = {
                "placement": placement,
                "step": torch.full((batch_size,), step, device=device),
                "mask": legal,
            }
            if transport is not None:
                obs["placed_transport"] = totals
                if transport_history is not None:
                    transport_history[:, decision].copy_(totals)
            candidates = targets.candidates[env.machine_class[step]].expand(batch_size, -1)
            distribution, candidate_legal, auxiliary = policy_distribution(
                model, obs, candidates, temperature, targets, completion_prior, return_value=return_values
            )
            if return_values:
                state_values[:, decision] = auxiliary
            if return_active:
                active_decisions[:, decision] = candidate_legal.sum(dim=1) > 1
            greedy_logits = distribution.logits if return_values else auxiliary
            choice = (
                greedy_logits.argmax(1, keepdim=True)
                if greedy
                else torch.multinomial(distribution.probs, 1, generator=generator)
            )
            log_probs[:, decision] = distribution.log_prob(choice.squeeze(1))
            chosen = candidates.gather(1, choice)
            decision += 1
        placement[:, step : step + 1].copy_(chosen)
        occupied.scatter_(1, chosen, True)
        if transport is not None and decision < plan.decision_count:
            transport.add_placement(totals, placement, step, network.distance)

    actions = placement.cpu().numpy()
    if not np.array_equal(
        np.sort(actions, axis=1), np.broadcast_to(np.arange(env.C), actions.shape)
    ) or not np.all(env.slot_class[actions] == env.machine_class[None]):
        raise RuntimeError("Policy rollout produced an incomplete or incompatible layout")
    layouts = np.empty_like(actions)
    np.put_along_axis(layouts, actions.astype(np.int64), env.machine_order[None], axis=1)
    if any(
        not np.array_equal(history, plan.canonical_actions(env, layout))
        for history, layout in zip(actions, layouts)
    ):
        raise RuntimeError("Policy rollout violated the canonical group action mask")
    result = actions, layouts, log_probs
    if return_values:
        result += (state_values,)
    return (*result, active_decisions) if return_active else result
