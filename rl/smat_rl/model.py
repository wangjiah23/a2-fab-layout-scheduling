"""CNN, directed GCN and shared slot-scoring MLP for the actor and critic."""

import numpy as np
import torch
from torch import nn

from .candidate_transport import CandidateTransportFeatures


class PlacedTransportFeatures(nn.Module):
    """One weighted, directed distance sum per nonzero process-flow edge.

    For g -> h, sum normalized slot distances over pairs already placed, times
    (flow[g,h] / mean_positive_flow) / (inventory[g] * inventory[h]).
    """

    def __init__(self, groups, flow, inventory):
        super().__init__()
        origin, destination = np.nonzero(flow)
        self.width = len(origin)
        edge_ids = np.full(flow.shape, -1, dtype=np.int64)
        edge_ids[origin, destination] = np.arange(self.width)
        machine_edges = edge_ids[groups[:, None], groups[None, :]]
        source, target = np.nonzero(machine_edges >= 0)
        # A directed pair becomes observable immediately after placing its later endpoint.
        ready = np.maximum(source, target)
        order = np.argsort(ready, kind="stable")
        self.offsets = np.searchsorted(ready[order], np.arange(len(groups) + 1)).tolist()
        source, target = source[order], target[order]
        edge = machine_edges[source, target]
        demand = flow[origin, destination].astype(np.float64)
        scale = demand / demand.mean() if self.width else demand
        scale = scale / inventory[origin].astype(np.float64) / inventory[destination]
        for name, array in [
            ("source", source),
            ("target", target),
            ("edge", edge),
            ("pair_weight", scale[edge].astype(np.float32)),
        ]:
            self.register_buffer(name, torch.as_tensor(array), persistent=False)

    def _add_pairs(self, totals, slots, distance, pairs, valid=None):
        source, target = self.source[pairs], self.target[pairs]
        costs = distance[slots[:, source].long(), slots[:, target].long()] * self.pair_weight[pairs]
        if valid is not None:
            costs = costs * (valid[:, source] & valid[:, target])
        totals.scatter_add_(1, self.edge[pairs].expand_as(costs), costs)

    def add_placement(self, totals, placement, step, distance):
        start, end = self.offsets[step : step + 2]
        if start < end:
            self._add_pairs(totals, placement, distance, slice(start, end))

    def forward(self, slots, valid, distance):
        # Standalone value queries can reconstruct the features directly. During
        # PPO, rollout supplies the exact same cached vector to every update.
        totals = distance.new_zeros((len(slots), self.width))
        for start in range(0, len(slots), 16):
            rows = slice(start, start + 16)
            self._add_pairs(totals[rows], slots[rows], distance, slice(None), valid[rows])
        return totals


class SlotActorCritic(nn.Module):
    """Score each candidate slot and optionally estimate the state's terminal return."""

    def __init__(self, env, *, transport_critic=False, routing_actor=False):
        super().__init__()
        data = env.d
        self.M = env.M
        self.C = env.C
        self.K = env.K
        self.G = env.G
        self.shape = tuple(env.meta["image_shape"])
        self.R = int(np.prod(self.shape))
        sequence_groups = data["group_ids"][env.machine_order].astype(np.int64)
        remaining = np.empty((env.M + 1, env.K), np.float32)
        remaining[0] = data["counts"]
        group_placed = np.zeros((env.M + 1, env.G), np.float32)
        for t, k in enumerate(env.sequence):
            remaining[t + 1] = remaining[t]
            remaining[t + 1, k] -= 1
            group_placed[t + 1] = group_placed[t]
            group_placed[t + 1, sequence_groups[t]] += 1
        group_flow = data["flow_weights"].astype(np.float32)
        out = group_flow / (group_flow.sum(1, keepdims=True) + 1e-8)
        inc = group_flow.T / (group_flow.T.sum(1, keepdims=True) + 1e-8)
        incident_flow = np.maximum(group_flow.sum(1) + group_flow.sum(0), 1.0)
        log_out = group_flow / incident_flow[:, None] / data["inventory"][None]
        log_in = group_flow.T / incident_flow[:, None] / data["inventory"][None]
        geometry = np.zeros((3, self.R), np.float32)
        geometry[0, data["raster_index"]] = 1
        geometry[1:, data["raster_index"]] = data["slot_features"][:, :2].T
        for name, array in [
            ("sequence", env.sequence),
            ("sequence_groups", sequence_groups),
            ("raster_index", data["raster_index"]),
            ("remaining_table", remaining / data["counts"][None]),
            ("group_fraction_table", group_placed / data["inventory"][None]),
            ("static_graph", data["static_graph"]),
            ("geometry", geometry),
            ("slot_features", data["slot_features"]),
            ("outgoing", out),
            ("incoming", inc),
            ("log_out", log_out),
            ("log_in", log_in),
            (
                "distance",
                (data["distances"] / max(1.0, float(data["distances"].max()))).astype(np.float32),
            ),
        ]:
            self.register_buffer(name, torch.as_tensor(np.array(array)))
        self.cnn = nn.Sequential(
            nn.Conv2d(self.G + 4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.gcn1 = nn.Linear(24, 64)
        self.gcn2 = nn.Linear(192, 64)
        self.global_encoder = nn.Sequential(nn.Linear(2 * self.K + 1, 32), nn.ReLU())
        # Existing PPO checkpoints store the shared logit head under "advantage".
        self.advantage = nn.Sequential(nn.Linear(64 + 64 + 64 + 32 + 1 + 6, 64), nn.ReLU(), nn.Linear(64, 1))
        self.placed_transport = (
            PlacedTransportFeatures(sequence_groups, group_flow, data["inventory"])
            if transport_critic
            else None
        )
        # Adding the critic preserves the actor's initialization and RNG stream.
        with torch.random.fork_rng(devices=[]):
            critic_width = 64 + 64 + 32 + (self.placed_transport.width if transport_critic else 0)
            self.critic = nn.Sequential(nn.Linear(critic_width, 64), nn.ReLU(), nn.Linear(64, 1))
            nn.init.zeros_(self.critic[-1].weight)
            nn.init.zeros_(self.critic[-1].bias)
        self.candidate_transport = CandidateTransportFeatures(env) if routing_actor else None
        if routing_actor:
            # Both ablations start with identical scores, sampling RNG and critic.
            with torch.random.fork_rng(devices=[]):
                self.route_projection = nn.Linear(self.candidate_transport.feature_dim, 64, bias=False)
                nn.init.zeros_(self.route_projection.weight)

    def message(self, x, layer):
        return torch.relu(layer(torch.cat([x, self.outgoing @ x, self.incoming @ x], -1)))

    def forward(self, obs, candidate_slots, return_value=False):
        """Return one logit per candidate in each batch row, plus values on request."""
        placement = obs["placement"].long()
        steps = obs["step"].long()
        batch_size = len(placement)
        rows = torch.arange(batch_size, device=placement.device)
        valid = (placement >= 0) & (torch.arange(self.M, device=placement.device)[None] < steps[:, None])
        slots = placement.clamp_min(0)
        raster = self.raster_index[slots]
        index = self.sequence_groups[None] * self.R + raster
        counts = torch.zeros((batch_size, self.G * self.R), device=placement.device, dtype=torch.float32)
        counts.scatter_add_(1, index, valid.float())
        counts = counts.reshape(batch_size, self.G, self.R)
        occupancy = counts.sum(1, keepdim=True)
        image = torch.cat([counts, occupancy, self.geometry[None].expand(batch_size, -1, -1)], 1).reshape(
            batch_size, self.G + 4, *self.shape
        )
        spatial = self.cnn(image).flatten(2)
        fraction = self.group_fraction_table[steps]
        group_features = torch.cat(
            [
                self.static_graph[None].expand(batch_size, -1, -1),
                (1 - fraction)[:, :, None],
                fraction[:, :, None],
            ],
            -1,
        )
        graph = self.message(self.message(group_features, self.gcn1), self.gcn2)
        pool = graph.mean(1)
        current_type = self.sequence[steps.clamp_max(self.M - 1)]
        current_group = self.sequence_groups[steps.clamp_max(self.M - 1)]
        selected = graph[rows, current_group]
        onehot = torch.nn.functional.one_hot(current_type, self.K).float() * (steps < self.M)[:, None]
        global_features = self.global_encoder(
            torch.cat([self.remaining_table[steps], onehot, steps[:, None].float() / self.M], 1)
        )
        if return_value:
            parts = [spatial.mean(2), pool, global_features]
            if self.placed_transport is not None:
                transport = obs.get("placed_transport")
                if transport is None:
                    transport = self.placed_transport(slots, valid, self.distance)
                parts.append(transport)
            state_features = torch.cat(parts, dim=-1)
        outgoing = torch.zeros((batch_size, self.C), device=placement.device)
        incoming = torch.zeros_like(outgoing)
        outbound_weights = self.log_out[current_group[:, None], self.sequence_groups[None]] * valid
        inbound_weights = self.log_in[current_group[:, None], self.sequence_groups[None]] * valid
        outgoing.scatter_add_(1, slots, outbound_weights)
        incoming.scatter_add_(1, slots, inbound_weights)
        flow = (outgoing @ self.distance.T + incoming @ self.distance)[:, :, None]
        spatial = spatial.transpose(1, 2)[rows[:, None], self.raster_index[candidate_slots]]
        width = candidate_slots.shape[1]
        fused = torch.cat(
            [
                spatial,
                selected[:, None].expand(-1, width, -1),
                pool[:, None].expand(-1, width, -1),
                global_features[:, None].expand(-1, width, -1),
                flow[rows[:, None], candidate_slots],
                self.slot_features[candidate_slots],
            ],
            -1,
        )
        with torch.autocast(device_type=fused.device.type, enabled=False):
            actor_input = fused.float()
            if self.candidate_transport is None:
                logits = self.advantage(actor_input).squeeze(-1)
            else:
                routing = self.candidate_transport(placement, steps, candidate_slots)
                hidden = self.advantage[0](actor_input) + self.route_projection(routing)
                logits = self.advantage[2](self.advantage[1](hidden)).squeeze(-1)
            if return_value:
                return logits, self.critic(state_features.float()).squeeze(-1)
        return logits
