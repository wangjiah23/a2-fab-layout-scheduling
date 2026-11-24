"""Place the fixed machine sequence on the original factory slots."""

import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_INSTANCE = Path(__file__).resolve().parents[1] / "data" / "original_slot_instance"


def load_instance(instance=DEFAULT_INSTANCE):
    path = Path(instance)
    metadata = json.loads((path / "instance.json").read_text(encoding="utf-8"))
    with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
        return metadata, dict(arrays)


class OriginalSlotEnv:
    """Execute physical placements and evaluate completed layouts.

    The physical mask admits each unused slot of the current machine's class.
    GroupSlotPlan adds the ordering constraints used by the PPO sampler.
    """

    def __init__(self, evaluator, evaluate_terminal=True, defer_evaluation=None):
        self.evaluator = evaluator
        self.meta = evaluator.meta
        self.d = evaluator.data
        self.evaluate_terminal = evaluate_terminal if defer_evaluation is None else not defer_evaluation
        self.reference_cost = float(evaluator.reference_cost)
        self.reward_scale = 1.0
        self.shaping = False
        self.M = self.C = len(self.d["classes"])
        self.K = len(self.d["counts"])
        self.G = len(self.d["inventory"])
        self.machine_order = self.d["order"].astype(np.int64)
        self.sequence = self.d["type_ids"][self.machine_order].astype(np.int64)
        self.type_graph = self.d["type_group"].astype(np.int64)
        self.slot_class = self.d["classes"]
        self.machine_class = self.slot_class[self.machine_order]
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.t = 0
        self.placement = np.full(self.M, -1, dtype=np.int16)
        self.device_at_slot = np.full(self.C, -1, dtype=np.int16)
        self.actions = []
        self.x = np.zeros((self.K, self.C), dtype=np.int8)
        return self.observe(), {}

    def legal_action_mask(self):
        if self.t == self.M:
            return np.zeros(self.C, dtype=np.int8)
        # Each class is a complete bipartite matching with equal machine/slot
        # inventories, so any unoccupied same-class slot preserves completion.
        return ((self.device_at_slot < 0) & (self.slot_class == self.machine_class[self.t])).astype(np.int8)

    def observe(self):
        return {
            "placement": self.placement.copy(),
            "step": np.array(self.t, dtype=np.int32),
            "mask": self.legal_action_mask(),
        }

    def step(self, action):
        if self.t == self.M:
            raise RuntimeError("Episode ended")
        if (
            not isinstance(action, (int, np.integer))
            or action < 0
            or action >= self.C
            or not self.legal_action_mask()[action]
        ):
            raise ValueError("Illegal original slot")
        machine = int(self.machine_order[self.t])
        kind = int(self.sequence[self.t])
        self.device_at_slot[action] = machine
        self.placement[self.t] = action
        self.x[kind, action] = 1
        self.actions.append(int(action))
        self.t += 1
        done = self.t == self.M
        reward = 0.0
        info = {}
        if done:
            if self.evaluate_terminal:
                result = self.evaluator.evaluate(self.device_at_slot)
                reward = 1 - result["cost"] / self.reference_cost
                info = {
                    **result,
                    "base_reward": reward,
                    "return": reward,
                    "actions": self.actions.copy(),
                }
            else:
                reward = None
                info = {
                    "evaluation_pending": True,
                    "device_at_slot": self.device_at_slot.tolist(),
                    "actions": self.actions.copy(),
                }
        return self.observe(), reward, done, False, info

    def reference_action(self):
        if self.t == self.M:
            raise RuntimeError("Episode ended")
        return int(self.machine_order[self.t])


def configure_slot_compatibility(lp, mode):
    """Derive E3 footprint classes in memory for every placement and LP caller."""
    if mode == "process-phase":
        return
    equipment = lp.meta["equipment"]
    footprints = [(round(float(item["width"]), 1), round(float(item["height"]), 1)) for item in equipment]
    keys = sorted(set(footprints))
    lookup = {key: index for index, key in enumerate(keys)}
    classes = np.asarray([lookup[key] for key in footprints], dtype=np.int16)
    type_class = np.asarray(
        [classes[np.flatnonzero(lp.data["type_ids"] == kind)[0]] for kind in range(len(lp.data["counts"]))],
        dtype=np.int16,
    )
    lp.data = {**lp.data, "classes": classes, "slot_class": classes, "type_class": type_class}
    lp.meta = {
        **lp.meta,
        "class_keys": keys,
        "compatibility": "E3 WIDTH/HEIGHT rounded to one decimal",
        "types": [{**kind, "class": int(type_class[index])} for index, kind in enumerate(lp.meta["types"])],
    }
    lp.fingerprint = hashlib.sha256(
        bytes.fromhex(lp.fingerprint) + b"E3-footprint" + classes.tobytes()
    ).hexdigest()
