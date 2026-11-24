"""Directed port distances visible to the actor before choosing a physical slot."""

import numpy as np
import torch
from torch import nn


class CandidateTransportFeatures(nn.Module):
    """Encode a candidate using only occupied slots and known remaining inventories.

    The first four channels are demand-weighted nearest distances (placed out/in,
    future out/in). The next four give the corresponding fractions of demand
    with an available endpoint. The last two vectors retain the placed out/in
    distances to every group, including groups with no direct flow. A missing
    group is encoded as 2, outside the normalized distance range [0, 1].
    """

    missing = 2.0

    def __init__(self, env):
        super().__init__()
        groups = env.d["group_ids"][env.machine_order].astype(np.int64)
        classes = env.slot_class.astype(np.int64)
        self.slot_count = len(classes)
        self.machine_count = len(groups)
        self.group_count = env.G
        self.class_count = int(classes.max()) + 1
        self.feature_dim = 8 + 2 * self.group_count

        flow = np.asarray(env.d["flow_weights"], dtype=np.float64)
        out = np.divide(
            flow,
            flow.sum(1, keepdims=True),
            out=np.zeros_like(flow),
            where=flow.sum(1, keepdims=True) > 0,
        )
        inc = np.divide(flow.T, flow.sum(0)[:, None], out=np.zeros_like(flow), where=flow.sum(0)[:, None] > 0)
        distance = (env.d["distances"] / max(1.0, float(env.d["distances"].max()))).astype(np.float32)
        # Groups with the same remaining compatible classes share one nearest
        # future distance. Combine their flow weights before processing batches.
        remaining = np.zeros((self.group_count, self.class_count), dtype=np.int64)
        same_type_after = np.zeros(self.machine_count + 1, dtype=np.int64)
        class_slots = np.full((self.class_count, np.bincount(classes).max()), -1, dtype=np.int64)
        for cls in range(self.class_count):
            members = np.flatnonzero(classes == cls)
            class_slots[cls, : len(members)] = members
        patterns = []
        for t in reversed(range(self.machine_count)):
            group = groups[t]
            cls = env.machine_class[t]
            same_type_after[t] = remaining[group, cls]
            patterns_at_step = {}
            for other in np.flatnonzero((out[group] + inc[group]) > 0):
                classes_left = np.flatnonzero(remaining[other])
                if same_type_after[t]:
                    # Two extra bins distinguish slots reachable by the active
                    # ordered block from slots reachable by any later group.
                    classes_left[classes_left == cls] = self.class_count + int(other != group)
                signature = tuple(classes_left)
                if signature:
                    weights = patterns_at_step.setdefault(signature, np.zeros(2))
                    weights += (out[group, other], inc[group, other])
            patterns.append(patterns_at_step)
            # The candidate consumes machine t, so future endpoints use t+1:.
            remaining[group, cls] += 1
        pattern_count = max(1, max(map(len, patterns)))
        pattern_width = max((len(signature) for row in patterns for signature in row), default=1)
        future_classes = np.full(
            (self.machine_count + 1, pattern_count, pattern_width),
            self.class_count + 2,
            dtype=np.int64,
        )
        future_weights = np.zeros((self.machine_count + 1, pattern_count, 2), dtype=np.float32)
        for t, row in enumerate(reversed(patterns)):
            for k, (signature, weights) in enumerate(row.items()):
                future_classes[t, k, : len(signature)] = signature
                future_weights[t, k] = weights

        for name, array in [
            ("distance_out", distance),
            ("distance_in", distance.T.copy()),
            ("slot_class", classes),
            ("sequence_groups", groups),
            ("physical_times", np.arange(self.machine_count)),
            ("machine_class", env.machine_class.astype(np.int64)),
            ("class_slots", class_slots),
            ("same_type_after", same_type_after),
            ("out_weights", out.astype(np.float32)),
            ("in_weights", inc.astype(np.float32)),
            ("future_classes", future_classes),
            ("future_weights", future_weights),
        ]:
            self.register_buffer(name, torch.as_tensor(array), persistent=False)

    def _endpoint_buckets(self, placement, steps):
        # Free endpoints are grouped by compatibility class, occupied endpoints
        # by their actual group. Unseen history writes only to the extra column.
        valid = (self.physical_times < steps[:, None]) & (placement >= 0)
        slot_ids = torch.where(valid, placement.long(), self.slot_count)
        buckets = torch.empty((len(steps), self.slot_count + 1), device=steps.device, dtype=torch.long)
        buckets[:, :-1] = self.slot_class + self.group_count
        buckets.scatter_(1, slot_ids, torch.where(valid, self.sequence_groups, 0))
        return buckets[:, :-1]

    def _direction(
        self,
        distance,
        candidates,
        buckets,
        weights,
        signatures,
        future_weights,
        current_ports,
        higher,
        lower,
        all_higher_reserved,
    ):
        rows, width = candidates.shape
        ports = distance[candidates]
        # A vacant candidate is consumed by the current move. Its own port may
        # still be meaningful when querying an occupied slot for diagnostics.
        diagonal = distance[candidates, candidates]
        diagonal = diagonal.masked_fill(buckets.gather(1, candidates) >= self.group_count, self.missing)
        ports.scatter_(2, candidates.unsqueeze(-1), diagonal.unsqueeze(-1))
        nearest = ports.new_full((rows, width, self.group_count + self.class_count + 3), self.missing)
        nearest.scatter_reduce_(2, buckets[:, None, :].expand_as(ports), ports, reduce="amin")
        placed = nearest[:, :, : self.group_count]
        placed_present = placed < self.missing
        placed_cost = (placed.masked_fill(~placed_present, 0.0) * weights[:, None, :]).sum(-1)
        placed_mass = (placed_present * weights[:, None, :]).sum(-1)

        by_class = nearest[:, :, self.group_count :]
        current_costs = ports.gather(2, current_ports[:, None].expand(-1, width, -1))
        higher_min = current_costs.masked_fill(~higher, self.missing).amin(-1)
        lower_min = current_costs.masked_fill(~lower, self.missing).amin(-1)
        by_class[..., self.class_count] = higher_min
        by_class[..., self.class_count + 1] = torch.where(
            all_higher_reserved, lower_min, torch.minimum(lower_min, higher_min)
        )
        signature_ids = signatures.flatten(1)[:, None].expand(-1, width, -1)
        future = by_class.gather(2, signature_ids).reshape(rows, width, *signatures.shape[1:]).amin(-1)
        future_present = future < self.missing
        future_cost = (future.masked_fill(~future_present, 0.0) * future_weights[:, None, :]).sum(-1)
        future_mass = (future_present * future_weights[:, None, :]).sum(-1)
        return placed_cost, future_cost, placed_mass, future_mass, placed

    def iter_features(self, placement, step, candidate_slots):
        """Yield (row slice, candidate slice, features) with bounded distance scratch."""
        if placement.shape != (len(step), self.machine_count) or step.ndim != 1:
            raise ValueError("Expected full-length placements and one physical step per row")
        if candidate_slots.ndim != 2 or candidate_slots.shape[0] != len(step):
            raise ValueError("Expected a candidate slot matrix with one row per state")
        if not len(step) or not candidate_slots.shape[1]:
            return
        # Each direction materializes at most 16 MiB of FP32 port distances,
        # even in mixed-step PPO minibatches and all-slot diagnostic queries.
        pair_limit = max(1, (1 << 22) // self.slot_count)
        candidate_chunk = min(candidate_slots.shape[1], pair_limit)
        row_chunk = max(1, pair_limit // candidate_chunk)
        for first in range(0, len(step), row_chunk):
            rows = slice(first, min(first + row_chunk, len(step)))
            times = step[rows].long()
            physical = times.clamp_max(self.machine_count - 1)
            current = self.sequence_groups[physical]
            buckets = self._endpoint_buckets(placement[rows], times)
            signatures = self.future_classes[times]
            future_weights = self.future_weights[times]
            members = self.class_slots[self.machine_class[physical]]
            current_ports = members.clamp_min(0)
            free = (members >= 0) & (buckets.gather(1, current_ports) >= self.group_count)
            for start in range(0, candidate_slots.shape[1], candidate_chunk):
                columns = slice(start, min(start + candidate_chunk, candidate_slots.shape[1]))
                ids = candidate_slots[rows, columns].long()
                candidates = ids.clamp_min(0)
                # The current block must consume k higher ports. When exactly k
                # remain, every one belongs to the block in every completion.
                higher = free[:, None] & (members[:, None] > candidates[:, :, None])
                lower = free[:, None] & (members[:, None] < candidates[:, :, None])
                reserved = higher.sum(-1) == self.same_type_after[times, None]
                out = self._direction(
                    self.distance_out,
                    candidates,
                    buckets,
                    self.out_weights[current],
                    signatures,
                    future_weights[..., 0],
                    current_ports,
                    higher,
                    lower,
                    reserved,
                )
                inc = self._direction(
                    self.distance_in,
                    candidates,
                    buckets,
                    self.in_weights[current],
                    signatures,
                    future_weights[..., 1],
                    current_ports,
                    higher,
                    lower,
                    reserved,
                )
                totals = torch.stack([out[0], inc[0], out[1], inc[1], out[2], inc[2], out[3], inc[3]], dim=-1)
                result = torch.cat([totals, out[4], inc[4]], dim=-1)
                result.masked_fill_(((ids < 0) | (times[:, None] == self.machine_count)).unsqueeze(-1), 0.0)
                yield rows, columns, result

    def forward(self, placement, step, candidate_slots):
        output = self.distance_out.new_empty((*candidate_slots.shape, self.feature_dim))
        for rows, columns, features in self.iter_features(placement, step, candidate_slots):
            output[rows, columns] = features
        return output
