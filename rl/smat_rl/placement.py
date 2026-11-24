"""Ordered placement decisions for interchangeable machines in a size class."""

from collections import Counter, deque
from math import comb

import numpy as np
import torch


def ordered_slot_mask(free, slot_ids, previous_slot, same_type_after):
    """Keep increasing same-type slots with enough higher slots to finish the block."""
    higher_free = free.flip(1).cumsum(1, dtype=torch.int32).flip(1) - free.to(torch.int32)
    legal = free & (slot_ids > previous_slot) & (higher_free >= same_type_after)
    return legal, higher_free


class GroupSlotPlan:
    """Map the fixed machine sequence to the decision times that require a policy."""

    def __init__(self, env):
        groups = env.d["group_ids"][env.machine_order]
        machine_types = list(zip(env.machine_class, groups))
        remaining = Counter(machine_types)
        remaining_groups = Counter(cls for cls, _ in remaining)
        previous = {}

        self.slot_class = env.slot_class
        self.machine_class = env.machine_class
        self.is_decision = np.empty(len(machine_types), dtype=bool)
        self.same_type_prev = np.full(len(machine_types), -1, dtype=np.int64)
        self.same_type_after = np.empty(len(machine_types), dtype=np.int64)
        for step, kind in enumerate(machine_types):
            cls, _ = kind
            if kind in previous and previous[kind] != step - 1:
                raise ValueError("Every class/group type must occupy one contiguous block")
            self.same_type_prev[step] = previous.get(kind, -1)
            previous[kind] = step
            self.is_decision[step] = remaining_groups[cls] > 1
            remaining[kind] -= 1
            self.same_type_after[step] = remaining[kind]
            if remaining[kind] == 0:
                remaining_groups[cls] -= 1

        # The sentinel is the physical terminal state after all auto-fills.
        self.decision_times = np.append(np.flatnonzero(self.is_decision), len(machine_types))
        self.decision_count = len(self.decision_times) - 1

        class_sizes = np.bincount(self.slot_class)
        largest_class = int(class_sizes.max())
        self.candidates = np.full((len(class_sizes), largest_class), -1, dtype=np.int64)
        for cls, size in enumerate(class_sizes):
            self.candidates[cls, :size] = np.flatnonzero(self.slot_class == cls)
        self.completion_counts = np.array(
            [
                [comb(higher, after) for higher in range(largest_class + 1)]
                for after in range(int(self.same_type_after.max()) + 1)
            ],
            dtype=np.float64,
        )

    def legal_numpy(self, placement, physical_steps):
        """Legal ordered slots for a batch of physical prefixes or full histories."""
        placement = np.asarray(placement)
        steps = np.asarray(physical_steps)
        machines, slots = len(self.machine_class), len(self.slot_class)
        if placement.shape != (steps.size, machines) or steps.ndim != 1:
            raise ValueError("Expected one physical step per row of full-length placements")
        occupied = np.zeros((len(steps), slots), dtype=bool)
        rows, columns = np.nonzero((np.arange(machines) < steps[:, None]) & (placement >= 0))
        occupied[rows, placement[rows, columns]] = True
        current = np.minimum(steps, machines - 1)
        free = (self.slot_class == self.machine_class[current, None]) & ~occupied
        higher_free = np.cumsum(free[:, ::-1], axis=1)[:, ::-1] - free
        previous = self.same_type_prev[current]
        previous_slot = np.where(previous >= 0, placement[np.arange(len(steps)), previous], -1)
        return (
            free
            & (steps[:, None] < machines)
            & (np.arange(slots) > previous_slot[:, None])
            & (higher_free >= self.same_type_after[current, None])
        )

    def canonical_actions(self, env, device_at_slot):
        """Realize a compatible layout's slot groups in the fixed physical order."""
        slots = np.asarray(device_at_slot)
        if (
            slots.shape != (env.C,)
            or not np.issubdtype(slots.dtype, np.integer)
            or not np.array_equal(np.sort(slots), np.arange(env.C))
        ):
            raise ValueError("Expected a complete device-at-slot permutation")
        if not np.array_equal(env.slot_class[slots], env.slot_class):
            raise ValueError("Expected an original-slot compatible layout")

        groups = env.d["group_ids"]
        available = {}
        for slot, kind in enumerate(zip(env.slot_class, groups[slots])):
            available.setdefault(kind, deque()).append(slot)
        return np.array(
            [available[kind].popleft() for kind in zip(env.machine_class, groups[env.machine_order])],
            dtype=np.int16,
        )


class GroupDecisionTargets:
    """Reconstruct a batch of legal observations at policy decision times."""

    def __init__(self, env, plan, device):
        self.decisions = plan.decision_count
        self.machines = env.M
        self.times = torch.as_tensor(plan.decision_times, device=device)
        self.classes = torch.as_tensor(env.slot_class, device=device)
        self.sequence = torch.as_tensor(env.machine_class, device=device)
        self.previous = torch.as_tensor(plan.same_type_prev, device=device)
        self.after = torch.as_tensor(plan.same_type_after, device=device)
        self.slot_ids = torch.arange(env.C, device=device)
        self.candidates = torch.as_tensor(plan.candidates, device=device)

    def observations(self, histories, decisions):
        physical = self.times[decisions]
        placed = self.slot_ids < physical[:, None]
        occupied = torch.zeros_like(histories, dtype=torch.int32)
        occupied.scatter_add_(1, histories.long(), placed.int())
        current = physical.clamp_max(self.machines - 1)
        free = (occupied == 0) & (self.classes == self.sequence[current, None])
        previous = self.previous[current]
        previous_slot = histories.gather(1, previous.clamp_min(0)[:, None])
        previous_slot = torch.where(previous[:, None] >= 0, previous_slot, -1)
        legal, _ = ordered_slot_mask(free, self.slot_ids, previous_slot, self.after[current, None])
        legal &= physical[:, None] < self.machines
        return {"placement": torch.where(placed, histories, -1), "step": physical, "mask": legal}
