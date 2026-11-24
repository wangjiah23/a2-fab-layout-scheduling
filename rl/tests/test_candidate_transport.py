"""CPU regressions for the routing actor on the full SMAT E3 instance."""

import unittest
from itertools import permutations, product
from types import SimpleNamespace

import numpy as np
import torch

from smat_rl.candidate_transport import CandidateTransportFeatures
from smat_rl.env import OriginalSlotEnv, configure_slot_compatibility, load_instance
from smat_rl.model import SlotActorCritic
from smat_rl.placement import GroupDecisionTargets, GroupSlotPlan


def numpy_candidate_features(env, history, step, candidates):
    """Reference implementation over individual machines, slots, and group OD edges."""
    groups = env.d["group_ids"][env.machine_order]
    current = groups[step]
    flow = env.d["flow_weights"].astype(np.float64)
    distance = env.d["distances"] / env.d["distances"].max()
    weights = np.stack([flow[current, :], flow[:, current]])
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    free = np.ones(env.C, bool)
    free[history[:step]] = False
    current_class = env.machine_class[step]
    future_groups = groups[step + 1 :]
    future_classes = env.machine_class[step + 1 :]
    same_type_left = np.count_nonzero((future_classes == current_class) & (future_groups == current))
    expected = np.zeros((len(candidates), 8 + 2 * env.G))
    expected[:, 8:] = 2.0
    for pos, slot in enumerate(candidates):
        for group in range(env.G):
            placed = history[:step][groups[:step] == group]
            classes = future_classes[future_groups == group]
            possible = np.flatnonzero(free & np.isin(env.slot_class, classes))
            in_current = env.slot_class[possible] == current_class
            higher = np.count_nonzero(free[slot + 1 :] & (env.slot_class[slot + 1 :] == current_class))
            if group == current:
                possible = possible[~in_current | (possible > slot)]
            elif higher == same_type_left:
                possible = possible[~in_current | (possible < slot)]
            for offset, endpoints in [(0, placed), (2, possible[possible != slot])]:
                if not len(endpoints):
                    continue
                nearest = np.array([distance[slot, endpoints].min(), distance[endpoints, slot].min()])
                expected[pos, offset : offset + 2] += nearest * weights[:, group]
                expected[pos, 4 + offset : 6 + offset] += weights[:, group]
                if offset == 0:
                    expected[pos, [8 + group, 8 + env.G + group]] = nearest
    return expected


class CandidateTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        meta, data = load_instance()
        factory = SimpleNamespace(
            meta=meta, data=data, reference_cost=meta["reference_cost"], d={}, fingerprint="00" * 32
        )
        configure_slot_compatibility(factory, "footprint")
        cls.env = OriginalSlotEnv(factory, evaluate_terminal=False)
        cls.plan = GroupSlotPlan(cls.env)
        cls.targets = GroupDecisionTargets(cls.env, cls.plan, torch.device("cpu"))
        cls.encoder = CandidateTransportFeatures(cls.env)
        cls.identity = np.arange(cls.env.C)
        cls.shuffled = cls.identity.copy()
        rng = np.random.default_rng(20261124)
        for footprint in np.unique(cls.env.slot_class):
            slots = np.flatnonzero(cls.env.slot_class == footprint)
            cls.shuffled[slots] = rng.permutation(slots)

    def history(self, layout):
        return torch.from_numpy(self.plan.canonical_actions(self.env, layout))

    def prefix_pair(self, layout, swapped, step):
        other = layout.copy()
        other[list(swapped)] = other[list(reversed(swapped))]
        histories = torch.stack([self.history(layout), self.history(other)])
        decision = int(np.flatnonzero(self.plan.decision_times == step).item())
        return self.targets.observations(histories, torch.full((2,), decision))

    def test_directed_distance_and_coverage_agree_with_slot_enumeration(self):
        history = self.history(self.shuffled)
        steps = self.targets.times[torch.tensor([0, 88, 405, self.plan.decision_count - 1])]
        # Select legal slots at different physical times and include a repeat to test padded gathers.
        masks = self.plan.legal_numpy(np.tile(history.numpy(), (len(steps), 1)), steps.numpy())
        candidates = torch.tensor(np.stack([np.flatnonzero(mask)[[0, -1, 0]] for mask in masks]))
        actual = self.encoder(history.expand(len(steps), -1), steps, candidates)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.shape, (len(steps), 3, 8 + 2 * self.env.G))
        for i, step in enumerate(steps.tolist()):
            expected = numpy_candidate_features(self.env, history.numpy(), step, candidates[i].numpy())
            np.testing.assert_allclose(actual[i].numpy(), expected, rtol=2e-5, atol=2e-7)

    def test_two_real_prefix_pairs_have_distinguishable_candidate_features(self):
        torch.manual_seed(42)
        old_actor = SlotActorCritic(self.env).eval()
        for layout, swapped, step, slots in [
            (self.identity, (923, 737), 3, [690, 162]),
            (self.shuffled, (189, 1064), 97, [1036, 283]),
        ]:
            with self.subTest(physical_placement=step + 1), torch.no_grad():
                obs = self.prefix_pair(layout, swapped, step)
                candidates = torch.tensor([slots, slots])
                self.assertTrue(obs["mask"].gather(1, candidates).all())
                self.assertTrue(torch.equal(obs["mask"][0], obs["mask"][1]))
                local = old_actor(obs, candidate_slots=candidates)
                torch.testing.assert_close(local[0], local[1], rtol=0.0, atol=1e-7)
                features = self.encoder(obs["placement"], obs["step"], candidates)
                # Differences between candidate vectors remove any common prefix-only shift.
                contrast = features[:, 0] - features[:, 1]
                if step == 3:
                    # Changed vacant endpoints affect the nearest future incoming distance.
                    self.assertGreater(abs((contrast[0, 3] - contrast[1, 3]).item()), 1e-6)
                else:
                    # Neither exchanged group is a flow neighbor. Its location is visible
                    # through the per-group geometry even when every OD summary agrees.
                    torch.testing.assert_close(features[0, :, :8], features[1, :, :8], rtol=0.0, atol=1e-7)
                    self.assertGreater((contrast[0, 8:] - contrast[1, 8:]).abs().max().item(), 1e-6)
                    torch.manual_seed(42)
                    routed = SlotActorCritic(self.env, routing_actor=True).eval()
                    torch.testing.assert_close(routed(obs, candidate_slots=candidates), local)
                    # A realizable head reads one geometry channel. This checks model
                    # capacity on the counterexample independently of any LP returns.
                    channel = int((contrast[0] - contrast[1]).abs().argmax())
                    for parameter in routed.advantage.parameters():
                        parameter.zero_()
                    routed.route_projection.weight[0, channel] = 1.0
                    routed.advantage[2].weight[0, 0] = 1.0
                    logits = routed(obs, candidate_slots=candidates)
                    torch.testing.assert_close(logits, features[:, :, channel])
                    log_odds = logits[:, 0] - logits[:, 1]
                    self.assertGreater(abs((log_odds[0] - log_odds[1]).item()), 1e-6)

    def test_future_endpoints_match_every_canonical_completion(self):
        classes = np.array([0, 0, 0, 0, 1, 1, 1])
        groups = np.array([0, 0, 1, 1, 0, 2, 2])
        distances = np.arange(1, 50, dtype=float).reshape(7, 7)
        distances[2, [0, 1, 3, 4, 5, 6]] = [1, 15, 6, 5, 7, 8]
        distances[[0, 1, 3, 4, 5, 6], 2] = [12, 15, 1, 7, 5, 8]
        flow = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        env = SimpleNamespace(
            C=7,
            G=3,
            M=7,
            machine_class=classes,
            slot_class=classes,
            machine_order=np.arange(7),
            d={"group_ids": groups, "distances": distances, "flow_weights": flow},
        )
        plan = GroupSlotPlan(env)
        completions = [
            plan.canonical_actions(env, np.array(a + b))
            for a, b in product(permutations(range(4)), permutations(range(4, 7)))
        ]
        candidates = torch.tensor([[1, 2]])
        features = CandidateTransportFeatures(env)(torch.full((1, 7), -1), torch.tensor([0]), candidates)
        for i, slot in enumerate(candidates[0].tolist()):
            suffixes = np.stack([history[1:] for history in completions if history[0] == slot])
            possible = [np.unique(suffixes[:, groups[1:] == group]) for group in range(3)]
            # Slot 3 is reserved for the second machine of group 0 after choosing 2.
            # The same group can also use any slot of the other footprint.
            if slot == 2:
                self.assertEqual([s.tolist() for s in possible], [[3, 4, 5, 6], [0, 1], [4, 5, 6]])
            else:
                self.assertEqual(possible[1].tolist(), [0, 2, 3])
            nearest_out = [distances[slot, endpoints].min() for endpoints in possible]
            nearest_in = [distances[endpoints, slot].min() for endpoints in possible]
            expected = [
                np.average(nearest_out, weights=flow[0]),
                np.average(nearest_in, weights=flow[:, 0]),
            ]
            np.testing.assert_allclose(
                features[0, i, 2:4].numpy(), np.asarray(expected) / distances.max(), rtol=1e-6
            )

    def test_online_and_replay_match_and_cannot_see_suffix(self):
        env = OriginalSlotEnv(self.env.evaluator, evaluate_terminal=False)
        history = self.history(self.shuffled)
        decision = 88
        physical = self.plan.decision_times[decision]
        for slot in history[:physical].tolist():
            env.step(slot)
        online = {k: torch.as_tensor(v).reshape(1, -1) for k, v in env.observe().items()}
        online["step"] = online["step"].flatten()
        online["mask"] = torch.from_numpy(
            self.plan.legal_numpy(online["placement"].numpy(), np.array([physical]))
        )
        replay = self.targets.observations(history[None], torch.tensor([decision]))
        # A complete action history may reach the model. Even a nonnegative,
        # contradictory suffix must have no effect before the action.
        changed_suffix = history[None].clone()
        changed_suffix[:, physical:] = history[physical:].flip([0])
        candidates = torch.where(replay["mask"][0])[0][None]
        torch.manual_seed(53)
        model = SlotActorCritic(env, transport_critic=True, routing_actor=True)
        with torch.no_grad():
            model.route_projection.weight.normal_(std=0.1)
            baseline = self.encoder(online["placement"], online["step"], candidates)
            online_logits = model(online, candidate_slots=candidates)
            for placement in [replay["placement"], history[None], changed_suffix]:
                obs = {**replay, "placement": placement}
                torch.testing.assert_close(baseline, self.encoder(placement, replay["step"], candidates))
                torch.testing.assert_close(online_logits, model(obs, candidate_slots=candidates))


if __name__ == "__main__":
    unittest.main()
