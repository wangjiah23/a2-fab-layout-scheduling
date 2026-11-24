"""Instance inputs, reference cost and the complete machine-arc LP fallback."""

import hashlib
import time
from pathlib import Path

import highspy
import numpy as np
from scipy import sparse

from ..env import DEFAULT_INSTANCE, load_instance
from .paths import SCALE
from .workers import SlotPathWorkers

# Existing checkpoints identify this published set of four input files. Preserve
# that identity only if the packed route data matches the original E1 routes.
ORIGINAL_INPUTS = "32589f65dee2373fd3eb4c455f55a8c66653da4cef9e8d8da34c4d4083da1b83"
ORIGINAL_OPERATIONS = "2cc04a8b3bcd8edefccadf217f76009c64747d7b7e43dd419716ec6bce66f401"


class OriginalSlotLP:
    def __init__(self, instance=DEFAULT_INSTANCE):
        self.instance = Path(instance)
        self.meta, self.data = load_instance(self.instance)
        base = self.instance.parent / "full_slots"
        self.seed_flow = base / "seed_flow.npz"
        self.arc_matrix = sparse.load_npz(base / "flow_matrix.npz").tocsc()
        with np.load(base / "flow_vectors.npz", allow_pickle=False) as source:
            self.vectors = {
                k: source[k] for k in ("source", "target", "ny", "lower", "upper", "capacity_start")
            }
        capacity_start = int(self.vectors["capacity_start"])
        self.mass = 2 * self.vectors["upper"][:capacity_start].sum() - sum(self.meta["demands"].values())
        self.reference_cost = float(self.meta["reference_cost"])
        flow = sparse.load_npz(self.instance / "reference_flow.npz")
        if abs(flow.multiply(self.data["distances"]).sum() - self.reference_cost) > 0.1:
            raise ValueError("Original-layout reference flow and reference cost disagree")

        with np.load(self.instance / "operations.npz", allow_pickle=False) as source:
            self.operations = dict(source)
        ids = self.operations["group_ids"]
        offsets = self.operations["product_offsets"]
        tau = self.operations["tau"]
        demand = self.operations["demands"]
        if (
            ids.ndim != 1
            or len(ids) != self.meta["physical_steps"]
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= len(self.meta["groups"]))
            or tau.shape != ids.shape
            or not np.isfinite(tau).all()
            or np.any(tau < 0)
            or offsets.shape != (len(demand) + 1,)
            or offsets.dtype.kind not in "iu"
            or offsets[0] != 0
            or offsets[-1] != len(ids)
            or np.any(np.diff(offsets) <= 0)
            or demand.ndim != 1
            or not np.isfinite(demand).all()
            or np.any(demand < 0)
        ):
            raise ValueError("Packed product routes disagree with the slot instance")
        digest = hashlib.sha256()
        for path in (
            self.instance / "instance.json",
            self.instance / "arrays.npz",
            base / "flow_matrix.npz",
            base / "flow_vectors.npz",
        ):
            digest.update(path.read_bytes())
        route_digest = hashlib.sha256()
        for array, dtype in ((ids, "<i2"), (tau, "<f8"), (offsets, "<i4"), (demand, "<f8")):
            route_digest.update(np.asarray(array, dtype=dtype).tobytes())
        if digest.hexdigest() == ORIGINAL_INPUTS:
            if route_digest.hexdigest() != ORIGINAL_OPERATIONS:
                raise ValueError("Packed routes differ from the original factory. Regenerate operations.npz.")
        else:
            digest.update(route_digest.digest())
        self.fingerprint = digest.hexdigest()

    def validate_layout(self, permutation):
        layout = np.asarray(permutation)
        classes = self.data["classes"]
        if (
            layout.shape != classes.shape
            or not np.issubdtype(layout.dtype, np.integer)
            or not np.array_equal(np.sort(layout), np.arange(len(classes)))
            or not np.array_equal(classes[layout], classes)
        ):
            raise ValueError("Expected a compatible permutation of all machines")
        return layout.astype(np.int32, copy=False)

    def evaluate(self, device_at_slot, verify=False):
        """Solve one layout on CPU. Every result carries a global LP certificate."""
        pool = SlotPathWorkers(self, self.seed_flow, workers=1)
        try:
            result = pool.submit(device_at_slot).result()
        finally:
            pool.shutdown()
        if "error" in result:
            raise RuntimeError(result["error"])
        if verify:
            objective, lower = result["cost"], result["dual_lower_bound"]
            if objective - lower > max(1.0, abs(objective)) * 1e-7 or result["primal_infeasibility"] > 1e-5:
                raise RuntimeError("Terminal LP certificate failed verification")
        return result

    def solve_arcs(self, layout):
        """Solve the complete arc formulation after both path-master retries fail."""
        inverse = np.argsort(layout)
        vectors = self.vectors
        src, dst = vectors["source"], vectors["target"]
        cost = np.r_[np.zeros(int(vectors["ny"])), self.data["distances"][inverse[src], inverse[dst]] / SCALE]
        matrix = self.arc_matrix
        model = highspy.HighsLp()
        model.num_col_, model.num_row_ = matrix.shape[1], matrix.shape[0]
        model.col_cost_ = cost
        model.col_lower_ = np.zeros(len(cost))
        model.col_upper_ = np.full(len(cost), highspy.kHighsInf)
        model.row_lower_, model.row_upper_ = vectors["lower"], vectors["upper"]
        model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        model.a_matrix_.start_, model.a_matrix_.index_, model.a_matrix_.value_ = (
            matrix.indptr,
            matrix.indices,
            matrix.data,
        )
        failures = []
        started = time.perf_counter()
        for algorithm, limit in [("simplex", 240.0), ("ipm", 300.0)]:
            solver = highspy.Highs()
            options = {
                "output_flag": False,
                "threads": 1,
                "solver": algorithm,
                "time_limit": limit,
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-8,
                "ipm_optimality_tolerance": 1e-11,
                "run_crossover": "off",
            }
            for name, value in options.items():
                solver.setOptionValue(name, value)
            solver.passModel(model)
            solver.run()
            status = solver.getModelStatus()
            if status != highspy.HighsModelStatus.kOptimal:
                failures.append(solver.modelStatusToString(status))
                continue
            solution = solver.getSolution()
            primal = np.asarray(solution.col_value)
            dual = np.array(solution.row_dual)
            capacity = slice(int(vectors["capacity_start"]), None)
            dual[capacity] = np.minimum(dual[capacity], 0)
            activity = matrix @ primal
            residual = float(
                max(
                    0, np.max(vectors["lower"] - activity), np.max(activity - vectors["upper"]), -primal.min()
                )
            )
            lower = float(
                vectors["upper"] @ dual + self.mass * min(0.0, float((cost - matrix.T @ dual).min()))
            )
            objective = float(cost @ primal)
            gap = max(0.0, objective - lower) / max(1.0, abs(objective))
            if residual <= 1e-5 and gap <= 1e-7:
                return {
                    "cost": objective * SCALE,
                    "status": "optimal_full_LP_fallback",
                    "solver": algorithm,
                    "seconds": time.perf_counter() - started,
                    "primal_infeasibility": residual,
                    "global_relative_gap_bound": gap,
                    "dual_lower_bound": lower * SCALE,
                }
            failures.append(f"primal residual {residual}, relative gap {gap}")
        raise RuntimeError(f"Complete arc LP failed certification: {failures}")
