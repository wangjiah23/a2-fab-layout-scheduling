"""CPU regressions for E3 transport features, PPO minibatches and checkpoint defaults."""

import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from smat_rl.cli import configure_ppo_settings
from smat_rl.env import OriginalSlotEnv, configure_slot_compatibility, load_instance
from smat_rl.model import SlotActorCritic
from smat_rl.placement import GroupDecisionTargets, GroupSlotPlan
from smat_rl.ppo import critic_explained_variance, ppo_update
from smat_rl.rollout import collect_policy


class GroupPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        meta, data = load_instance()
        factory = SimpleNamespace(
            data=data, meta=meta, d={}, fingerprint="00" * 32, reference_cost=meta["reference_cost"]
        )
        configure_slot_compatibility(factory, "footprint")
        cls.env = OriginalSlotEnv(factory, evaluate_terminal=False)
        cls.plan = GroupSlotPlan(cls.env)
        cls.targets = GroupDecisionTargets(cls.env, cls.plan, torch.device("cpu"))
        counts = cls.plan.completion_counts
        logs = np.full_like(counts, -np.inf)
        np.log(counts, where=counts > 0, out=logs)
        cls.targets.log_completion_counts = torch.tensor(logs, dtype=torch.float32)

    def test_new_defaults_and_legacy_checkpoint(self):
        args = SimpleNamespace(slot_compatibility="footprint", resume=False)
        configure_ppo_settings(args, None)
        self.assertEqual((args.critic_features, args.minibatches_per_epoch), ("transport", 4))
        args = SimpleNamespace(slot_compatibility="footprint", resume=True)
        configure_ppo_settings(args, {"ppo_epochs": 4})
        self.assertEqual((args.critic_features, args.minibatches_per_epoch), ("pooled", 1))
        args.critic_features = "transport"
        with self.assertRaisesRegex(ValueError, "critic_features"):
            configure_ppo_settings(args, {})
        saved = {"critic_features": "transport", "minibatches_per_epoch": 3}
        args = SimpleNamespace(slot_compatibility="footprint", resume=True)
        configure_ppo_settings(args, saved)
        self.assertEqual((args.critic_features, args.minibatches_per_epoch), ("transport", 3))
        self.assertEqual(args.actor_features, "local")
        args.actor_features = "routing"
        with self.assertRaisesRegex(ValueError, "actor_features"):
            configure_ppo_settings(args, saved)
        for arm in ("routing", "local"):
            with self.subTest(arm=arm):
                fresh = SimpleNamespace(slot_compatibility="footprint", resume=False, actor_features=arm)
                configure_ppo_settings(fresh, None)
                self.assertEqual(
                    (fresh.actor_features, fresh.critic_features, fresh.minibatches_per_epoch),
                    (arm, "transport", 4),
                )
                restored = SimpleNamespace(slot_compatibility="footprint", resume=True)
                configure_ppo_settings(restored, vars(fresh))
                self.assertEqual(restored.actor_features, arm)

    def test_explained_variance_compares_layouts_at_equal_progress(self):
        rewards = torch.tensor([0.25, 0.5, 0.75])
        active = torch.ones((3, 40), dtype=torch.bool)
        active[1, 0] = False
        offset = torch.arange(40) / 64.0
        values = offset.expand(3, -1).clone()
        values[:, -32:] += rewards[:, None]
        result = critic_explained_variance(rewards, values, active, 1)
        self.assertTrue(0.0 < result["critic_within_step_ev"] < 1.0)
        self.assertAlmostEqual(result["critic_last32_within_step_ev"], 1.0)
        self.assertIsNone(
            critic_explained_variance(rewards * 0.0, values, active, 1)["critic_within_step_ev"]
        )

    def test_real_rollout_transport_and_optimizer_steps(self):
        env, plan, targets = self.env, self.plan, self.targets
        torch.manual_seed(42)
        pooled = SlotActorCritic(env)
        torch.manual_seed(42)
        model = SlotActorCritic(env, transport_critic=True)
        self.assertEqual((env.M, plan.decision_count, model.placed_transport.width), (1068, 907, 314))
        # Give the new value head a nonconstant output so causality and replay checks are meaningful.
        with torch.no_grad():
            model.critic[-1].weight.fill_(0.02)
        cache = torch.empty((2, plan.decision_count, model.placed_transport.width))
        actions, _, log_probs, values, active = collect_policy(
            model,
            env,
            plan,
            targets,
            2,
            torch.Generator().manual_seed(919),
            1.0,
            completion_prior=True,
            return_values=True,
            return_active=True,
            transport_history=cache,
        )
        histories = torch.from_numpy(actions)
        sample = torch.tensor([0, plan.decision_count // 2, plan.decision_count - 1])
        obs = targets.observations(histories[:1].expand(len(sample), -1), sample)
        w, inventory = env.d["flow_weights"], env.d["inventory"]
        edges = list(zip(*np.nonzero(w)))
        mean_flow = w[w > 0].mean(dtype=np.float64)
        distance = env.d["distances"] / env.d["distances"].max()
        for step in sample.tolist():
            physical = plan.decision_times[step]
            groups = env.d["group_ids"][env.machine_order[:physical]]
            slots = actions[0, :physical]
            expected = [
                np.sum(distance[np.ix_(slots[groups == g], slots[groups == h])])
                * (w[g, h] / mean_flow)
                / (int(inventory[g]) * int(inventory[h]))
                for g, h in edges
            ]
            np.testing.assert_allclose(cache[0, step].numpy(), expected, rtol=4e-5, atol=5e-7)
        candidates = targets.candidates[targets.sequence[obs["step"]].long()].clamp_min(0)
        with torch.no_grad():
            logits, reconstructed = model(obs, candidate_slots=candidates, return_value=True)
            torch.testing.assert_close(logits, pooled(obs, candidate_slots=candidates))
            torch.testing.assert_close(reconstructed, values[0, sample])
            # Replace every unseen placement by a different valid slot ID.
            obs["placement"] = torch.where(obs["placement"] >= 0, obs["placement"], env.C - 1)
            _, changed_suffix = model(obs, candidate_slots=candidates, return_value=True)
            torch.testing.assert_close(changed_suffix, reconstructed)

        args = SimpleNamespace(
            gae_lambda=1.0,
            ppo_epochs=4,
            minibatches_per_epoch=4,
            decision_samples=7,
            gradient_batch=2,
            temperature=1.0,
            completion_prior=True,
            clip_epsilon=0.2,
            value_coef=0.5,
            grad_clip=1.0,
        )
        # Fixed return markers exercise update plumbing without launching an LP experiment.
        rewards = torch.tensor([0.45, 0.50])
        models = [model, deepcopy(model)]
        first_gradients = []
        for network, microbatch in zip(models, (2, 8)):
            args.gradient_batch = microbatch
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

            def capture_first_gradient(opt, _args, _kwargs):
                if not opt.state:
                    first_gradients.append(
                        torch.cat([p.grad.flatten() for p in opt.param_groups[0]["params"]])
                    )

            optimizer.register_step_pre_hook(capture_first_gradient)
            diagnostics = ppo_update(
                network,
                optimizer,
                histories,
                log_probs,
                values,
                active,
                rewards,
                targets,
                torch.Generator().manual_seed(123),
                args,
                1,
                cache,
            )
            self.assertEqual(diagnostics["optimizer_steps"], 16)
            self.assertEqual(int(optimizer.state[network.critic[0].weight]["step"]), 16)
            self.assertLess(diagnostics["initial_ratio_max_error"], 1e-4)
            self.assertGreater(network.critic[0].weight.grad[:, -cache.shape[2] :].abs().sum().item(), 0.0)
        # Compare gradients at identical weights, before Adam amplifies FP32 rounding.
        self.assertEqual(len(first_gradients), 2)
        torch.testing.assert_close(*first_gradients, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
