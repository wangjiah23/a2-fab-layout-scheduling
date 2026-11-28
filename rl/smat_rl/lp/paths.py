"""Exact path-column LP for the full layered manufacturing flow problem.

A column is one lot's route through eligible machines. Pricing searches every
path by dynamic programming and certifies the bound over the complete master.
Distances in the solver are scaled by 1e6, as in the original arc LP.
"""

import ctypes
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import highspy
import numpy as np

SCALE = 1e6


@lru_cache(maxsize=1)
def _path_kernels():
    source = Path(__file__).with_name("path_pricing.cpp")
    with tempfile.TemporaryDirectory(prefix="smat-path-") as directory:
        binary = Path(directory) / "pricing.so"
        subprocess.run(
            ["c++", "-O3", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(binary)], check=True
        )
        library = ctypes.CDLL(str(binary))
    int_array = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    double_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integer = ctypes.c_int
    library.price_paths.argtypes = [integer] * 3 + [int_array] * 3 + [double_array] * 4 + [int_array]
    library.alternate_path_edits.argtypes = (
        [integer] * 3 + [int_array] * 3 + [double_array] * 4 + [int_array] * 3
    )
    library.path_column_costs.argtypes = [integer] * 2 + [int_array] * 2 + [double_array] * 2
    library.path_column_coefficients.argtypes = (
        [integer] * 3 + [int_array] * 4 + [double_array] + [int_array] * 2 + [double_array]
    )
    for function in (
        library.price_paths,
        library.alternate_path_edits,
        library.path_column_costs,
        library.path_column_coefficients,
    ):
        function.restype = None
    return library


def _cpu_kernel():
    return _path_kernels().price_paths


class PathLP:
    def __init__(
        self,
        slots,
        seed_flow=None,
        max_iterations=1200,
        columns_per_product=4,
        tolerance=1e-7,
        retain_columns=4096,
    ):
        self.distance = np.asarray(slots.data["distances"], dtype=np.float64) / SCALE
        self.machines = len(self.distance)
        if (
            self.distance.shape != (self.machines, self.machines)
            or not np.isfinite(self.distance).all()
            or np.any(self.distance < 0)
        ):
            raise ValueError("Expected a finite nonnegative square slot distance matrix")
        self.capacity = np.full(
            self.machines, slots.meta["horizon_min"] * slots.meta["eta"], dtype=np.float64
        )
        if not np.isfinite(self.capacity).all() or np.any(self.capacity <= 0):
            raise ValueError("Machine-time capacities must be finite and positive")
        operations = slots.operations
        self.tau = operations["tau"]
        by_group = [
            np.flatnonzero(slots.data["group_ids"] == g).astype(np.int32)
            for g in range(len(slots.meta["groups"]))
        ]
        self.eligible = [by_group[g] for g in operations["group_ids"]]
        if not self.eligible or any(
            not len(ids) or ids.min() < 0 or ids.max() >= self.machines for ids in self.eligible
        ):
            raise ValueError("Every operation must have valid eligible machine IDs")
        if not np.isfinite(self.tau).all() or np.any(self.tau < 0):
            raise ValueError("Processing times must be finite and nonnegative")
        self.eligible_machines = np.concatenate(self.eligible)
        sizes = np.asarray([len(row) for row in self.eligible])
        self.operation_offsets = np.r_[0, np.cumsum(sizes)].astype(np.int32)
        self.product_offsets = operations["product_offsets"]
        self.route_lengths = np.diff(self.product_offsets)
        self.price_rounding = 32 * np.finfo(np.float64).eps * self.route_lengths
        self.demands = operations["demands"]
        self.products = len(self.demands)
        if min(max_iterations, columns_per_product, retain_columns) < 1:
            raise ValueError("Iteration and column limits must be positive")
        self.max_iterations = max_iterations
        self.columns_per_product = columns_per_product
        self.tolerance = tolerance
        self.retain_columns = retain_columns
        self.paths = []
        self.path_keys = set()
        self.master = None
        self._slices = [slice(int(a), int(b)) for a, b in zip(self.product_offsets, self.product_offsets[1:])]
        self._seed(seed_flow)
        self.seed_count = len(self.paths)

    def _seed(self, flow):
        if flow is not None:
            with np.load(flow) as saved:
                columns, values = saved["columns"], saved["values"]
                total = int(self.operation_offsets[-1]) + sum(
                    len(self.eligible[i]) * len(self.eligible[i + 1])
                    for route in self._slices
                    for i in range(route.start, route.stop - 1)
                )
                primal = np.zeros(total)
                primal[columns] = values
            self._decompose(primal)
            return
        # A balanced quantile coupling yields feasible initial paths whenever
        # identical eligibility groups fit their total machine-time capacity.
        for product, route in enumerate(self._slices):
            cdfs = []
            for ids in self.eligible[route]:
                cumulative = np.cumsum(self.capacity[ids])
                cdfs.append(cumulative / cumulative[-1])
            cuts = np.unique(np.concatenate([np.array([0.0]), *cdfs]))
            for midpoint in (cuts[:-1] + cuts[1:]) / 2:
                path = [
                    ids[np.searchsorted(cdf, midpoint, side="right")]
                    for ids, cdf in zip(self.eligible[route], cdfs)
                ]
                self._remember(product, path)

    def _decompose(self, primal):
        column = int(self.operation_offsets[-1])
        for product, route in enumerate(self._slices):
            eligible = self.eligible[route]
            if len(eligible) == 1:
                start, stop = self.operation_offsets[route.start : route.start + 2]
                for machine in eligible[0][primal[start:stop] > 1e-10]:
                    self._remember(product, [machine])
                continue
            transitions = []
            for source, target in zip(eligible, eligible[1:]):
                count = len(source) * len(target)
                transitions.append(primal[column : column + count].reshape(len(source), len(target)))
                column += count
            while True:
                source = int(transitions[0].sum(axis=1).argmax())
                if transitions[0][source].sum() <= 1e-9:
                    break
                positions = [source]
                amount = np.inf
                for matrix in transitions:
                    target = int(matrix[source].argmax())
                    amount = min(amount, matrix[source, target])
                    positions.append(target)
                    source = target
                if amount <= 1e-10:
                    # Ignore a numerically unmatched stub only below the
                    # master feasibility tolerance. Its own rows are verified.
                    if transitions[0].sum() > 1e-7:
                        raise ValueError("Arc solution cannot seed a complete path decomposition")
                    break
                for i, matrix in enumerate(transitions):
                    matrix[positions[i], positions[i + 1]] -= amount
                self._remember(product, [ids[i] for ids, i in zip(eligible, positions)])

    def _remember(self, product, path):
        path = np.array(path, dtype=np.int32, order="C", copy=True)
        key = (product, path.tobytes())
        if key in self.path_keys:
            return None
        self.path_keys.add(key)
        column = (product, path)
        self.paths.append(column)
        return column

    def price(self, distance, machine_dual):
        distance = np.ascontiguousarray(distance, dtype=np.float64)
        machine_dual = np.ascontiguousarray(machine_dual, dtype=np.float64)
        if distance.shape != (self.machines, self.machines) or machine_dual.shape != (self.machines,):
            raise ValueError("Pricing distance or dual shape mismatch")
        costs = np.empty(self.products)
        paths = np.empty(len(self.tau), dtype=np.int32)
        _cpu_kernel()(
            self.products,
            len(self.tau),
            self.machines,
            self.product_offsets,
            self.operation_offsets,
            self.eligible_machines,
            self.tau,
            distance,
            machine_dual,
            costs,
            paths,
        )
        return costs, paths

    def _new_master(self):
        h = highspy.Highs()
        for option, value in [
            ("output_flag", False),
            ("threads", 1),
            ("solver", "simplex"),
            ("simplex_strategy", 4),
            ("presolve", "off"),
            ("dual_feasibility_tolerance", 1e-9),
            ("primal_feasibility_tolerance", 1e-8),
            ("time_limit", 120.0),
        ]:
            if h.setOptionValue(option, value) != highspy.HighsStatus.kOk:
                raise RuntimeError(f"HiGHS rejected option {option}")
        lower = np.r_[self.demands, np.full(self.machines, -highspy.kHighsInf)]
        upper = np.r_[self.demands, self.capacity]
        h.addRows(
            len(lower),
            lower,
            upper,
            0,
            np.zeros(len(lower) + 1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0),
        )
        return h

    def _column_costs(self, distance, columns):
        count = len(columns)
        products = np.fromiter((product for product, _ in columns), dtype=np.int32, count=count)
        offsets = np.empty(count + 1, dtype=np.int32)
        offsets[0] = 0
        np.cumsum(self.route_lengths[products], out=offsets[1:])
        paths = np.concatenate([path for _, path in columns])
        costs = np.empty(count)
        _path_kernels().path_column_costs(count, self.machines, offsets, paths, distance, costs)
        return products, offsets, paths, costs

    def _append_columns(self, distance, columns):
        products, offsets, paths, costs = self._column_costs(distance, columns)
        count = len(columns)
        starts = np.empty(count + 1, dtype=np.int32)
        indices = np.empty(len(paths) + count, dtype=np.int32)
        data = np.empty(len(indices))
        _path_kernels().path_column_coefficients(
            count,
            self.products,
            self.machines,
            self.product_offsets,
            products,
            offsets,
            paths,
            self.tau,
            starts,
            indices,
            data,
        )
        used = int(starts[-1])
        status = self.master.addCols(
            count,
            costs,
            np.zeros(count),
            np.full(count, highspy.kHighsInf),
            used,
            starts,
            indices[:used],
            data[:used],
        )
        if status != highspy.HighsStatus.kOk:
            self.master = None
            raise RuntimeError("Failed to add path columns")

    def _priced_columns(self, distance, dual, allowances, paths):
        added = []
        keep = self.columns_per_product - 1
        positions = np.full((self.products, keep), -1, dtype=np.int32)
        machines = np.empty_like(positions)
        if keep:
            _path_kernels().alternate_path_edits(
                self.products,
                self.machines,
                keep,
                self.product_offsets,
                self.operation_offsets,
                self.eligible_machines,
                self.tau,
                distance,
                dual,
                allowances,
                paths,
                positions,
                machines,
            )
        for product in np.flatnonzero(allowances > 0):
            path = paths[self._slices[product]]
            column = self._remember(product, path)
            if column is not None:
                added.append(column)
            for i, position in enumerate(positions[product]):
                if position < 0:
                    break
                alternate = path.copy()
                alternate[position] = machines[product, i]
                column = self._remember(product, alternate)
                if column is not None:
                    added.append(column)
        return added

    def _prune(self, solution):
        if len(self.paths) <= 2 * self.retain_columns:
            return
        keep = np.asarray(solution.col_value) > 0
        keep[: self.seed_count] = True
        keep[-self.retain_columns :] = True
        remove = np.flatnonzero(~keep).astype(np.int32)
        if not len(remove):
            return
        status = self.master.deleteCols(len(remove), remove)
        if status != highspy.HighsStatus.kOk:
            self.master = None
        self.paths = [column for column, retained in zip(self.paths, keep) if retained]
        self.path_keys = {(product, path.tobytes()) for product, path in self.paths}

    def _solve_master(self):
        master = self.master
        optimal = highspy.HighsModelStatus.kOptimal
        numerical = (highspy.HighsModelStatus.kUnknown, highspy.HighsModelStatus.kSolveError)
        master.setOptionValue("time_limit", master.getRunTime() + 120.0)
        master.run()
        status = master.getModelStatus()
        if status == optimal:
            return 0
        if status in numerical:
            # Refit the same restricted LP from a fresh dual simplex basis.
            master.clearSolver()
            master.setOptionValue("simplex_strategy", 1)
            try:
                master.setOptionValue("time_limit", master.getRunTime() + 120.0)
                master.run()
                status = master.getModelStatus()
            finally:
                master.setOptionValue("simplex_strategy", 4)
            if status == optimal:
                return 1
        raise RuntimeError(f"Path master: {master.modelStatusToString(status)}")

    def solve(self, device_at_slot, pricing=None, *, include_allocation=False):
        started = time.perf_counter()
        master_seconds = pricing_seconds = column_seconds = 0.0
        master_recoveries = 0
        layout = np.asarray(device_at_slot)
        if (
            layout.shape != (self.machines,)
            or not np.issubdtype(layout.dtype, np.integer)
            or not np.array_equal(np.sort(layout), np.arange(self.machines))
        ):
            raise ValueError("Expected a complete machine-to-slot permutation")
        inverse = np.argsort(layout)
        distance = self.distance[np.ix_(inverse, inverse)]
        price = self.price if pricing is None else pricing
        tick = time.perf_counter()
        if self.master is None:
            self.master = self._new_master()
            self._append_columns(distance, self.paths)
        else:
            _, _, _, costs = self._column_costs(distance, self.paths)
            if (
                self.master.changeColsCost(len(costs), np.arange(len(costs), dtype=np.int32), costs)
                != highspy.HighsStatus.kOk
            ):
                raise RuntimeError("Failed to update path costs")
        column_seconds += time.perf_counter() - tick
        for iteration in range(self.max_iterations):
            tick = time.perf_counter()
            master_recoveries += self._solve_master()
            master_seconds += time.perf_counter() - tick
            solution = self.master.getSolution()
            row_dual = np.asarray(solution.row_dual)
            product_dual = row_dual[: self.products]
            machine_dual = np.minimum(0.0, row_dual[self.products :])
            value = float(self.master.getObjectiveValue())
            tick = time.perf_counter()
            minimum, paths = price(distance, machine_dual)
            pricing_seconds += time.perf_counter() - tick
            if not np.isfinite(minimum).all():
                raise RuntimeError("Nonfinite shortest path cost")
            rounding = self.price_rounding * np.maximum(1.0, np.abs(minimum))
            lower = float(
                self.demands @ np.minimum(product_dual, minimum - rounding) + self.capacity @ machine_dual
            )
            gap = max(0.0, value - lower) / max(1.0, abs(value))
            if pricing is not None and gap <= self.tolerance:
                # Certify the result with the independent CPU recurrence even
                # when an accelerated backend generates the working columns.
                tick = time.perf_counter()
                minimum, paths = self.price(distance, machine_dual)
                pricing_seconds += time.perf_counter() - tick
                rounding = self.price_rounding * np.maximum(1.0, np.abs(minimum))
                lower = float(
                    self.demands @ np.minimum(product_dual, minimum - rounding) + self.capacity @ machine_dual
                )
                gap = max(0.0, value - lower) / max(1.0, abs(value))
            residual = None
            if gap <= self.tolerance:
                values = np.asarray(solution.row_value)
                residual = max(
                    float(np.max(np.abs(values[: self.products] - self.demands))),
                    float(np.max(values[self.products :] - self.capacity)),
                    -min(solution.col_value),
                    0.0,
                )
                if residual <= 1e-5:
                    columns = len(self.paths)
                    allocation = {}
                    if include_allocation:
                        # Read before pruning, while column numbers match the HiGHS solution.
                        allocation["paths"] = [
                            (p, path, lots)
                            for (p, path), lots in zip(self.paths, solution.col_value)
                            if lots > 0
                        ]
                        allocation["loads"] = values[self.products :].copy()
                    self._prune(solution)
                    return {
                        **allocation,
                        "cost": value * SCALE,
                        "dual_lower_bound": lower * SCALE,
                        "global_relative_gap_bound": gap,
                        "primal_infeasibility": residual,
                        "solver_seconds": time.perf_counter() - started,
                        "pricing_iterations": iteration + 1,
                        "master_seconds": master_seconds,
                        "master_recoveries": master_recoveries,
                        "pricing_seconds": pricing_seconds,
                        "column_seconds": column_seconds,
                        "columns": columns,
                        "retained_columns": len(self.paths),
                        "status": "optimal_complete_path_LP",
                        "represented_machines": self.machines,
                        "all_operations": len(self.tau),
                    }
            tick = time.perf_counter()
            added = self._priced_columns(distance, machine_dual, product_dual - minimum - 1e-9, paths)
            if not added:
                raise RuntimeError(
                    f"Complete path pricing stalled at relative gap {gap:.3g}, residual {residual:.3g}"
                )
            self._append_columns(distance, added)
            column_seconds += time.perf_counter() - tick
        raise RuntimeError(f"Path pricing iteration limit at relative gap {gap:.3g}")
