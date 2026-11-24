"""Fused CUDA pricing for the complete layered process routes.

Costs use the restricted master's objective scale. The CUDA kernel returns
an eligible path and its FP64 price for each product. Conservative prices
for dual certificates are available separately via ``lower_bounds``.
"""

from pathlib import Path

import numpy as np


class CudaPathPricer:
    """Keep routes and original-slot distances resident on one CUDA device.

    ``inverse_slot[b, i]`` maps machine i to its slot in layout b. Omit it
    when distance is already indexed by machine. A one-dimensional dual
    vector returns one cost vector and one concatenated path. Batched duals
    return arrays with the batch axis retained. Each instance belongs to one
    solver worker and reuses its output buffers between calls.
    """

    def __init__(self, distance, product_offsets, operation_offsets, eligible_machines, tau, device=0):
        import cupy as cp

        distance = np.ascontiguousarray(distance, dtype=np.float64)
        product_offsets = np.ascontiguousarray(product_offsets, dtype=np.int32)
        operation_offsets = np.ascontiguousarray(operation_offsets, dtype=np.int32)
        eligible_machines = np.ascontiguousarray(eligible_machines, dtype=np.int32)
        tau = np.ascontiguousarray(tau, dtype=np.float64)
        if distance.ndim != 2 or distance.shape[0] != distance.shape[1] or not len(distance):
            raise ValueError("Distance matrix must be nonempty and square")
        if not np.isfinite(distance).all() or np.any(distance < 0):
            raise ValueError("Distances must be finite and nonnegative")
        if tau.ndim != 1 or not len(tau) or not np.isfinite(tau).all() or np.any(tau < 0):
            raise ValueError("Processing times must be finite and nonnegative")
        self._check_offsets(product_offsets, len(tau), "product")
        self._check_offsets(operation_offsets, len(eligible_machines), "operation")
        if len(operation_offsets) != len(tau) + 1:
            raise ValueError("Each operation needs one eligibility list")
        if (
            eligible_machines.ndim != 1
            or np.any(eligible_machines < 0)
            or np.any(eligible_machines >= len(distance))
        ):
            raise ValueError("Eligibility indices must reference existing machines")

        self.cp = cp
        self.device = cp.cuda.Device(device)
        self.machines = len(distance)
        self.products = len(product_offsets) - 1
        self.operations = len(tau)
        self.eligible_count = len(eligible_machines)
        self.route_lengths = np.diff(product_offsets).astype(np.float64)
        self.max_tau = np.maximum.reduceat(tau, product_offsets[:-1])
        self.max_distance = float(distance.max())
        self.max_width = int(np.diff(operation_offsets).max())
        self.threads = min(128, 32 * ((self.max_width + 31) // 32))
        self.shared_bytes = 24 * self.max_width
        self.capacity = 0
        source = Path(__file__).with_name("gpu_path_pricing.cu").read_text()
        with self.device:
            self._routes = tuple(
                cp.asarray(x) for x in (product_offsets, operation_offsets, eligible_machines, tau, distance)
            )
            self._identity = cp.arange(self.machines, dtype=cp.int32)
            # Keep the FP64 recurrence identical to separate CPU multiply/add.
            self._kernel = cp.RawKernel(source, "price_paths", options=("--fmad=false",))

    @staticmethod
    def _check_offsets(offsets, end, name):
        if (
            offsets.ndim != 1
            or len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != end
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError(f"{name.capitalize()} offsets must describe nonempty contiguous ranges")

    def _duals(self, machine_dual):
        dual = np.asarray(machine_dual, dtype=np.float64)
        single = dual.ndim == 1
        dual = np.ascontiguousarray(np.atleast_2d(dual))
        if dual.ndim != 2 or dual.shape[1] != self.machines or not len(dual) or not np.isfinite(dual).all():
            raise ValueError("Machine duals must be a finite vector or batch of vectors")
        return dual, single

    def price(self, machine_dual, inverse_slot=None):
        """Return FP64 product prices and the selected machine at each operation."""
        dual, single = self._duals(machine_dual)
        batch = len(dual)
        inverse = None
        if inverse_slot is not None:
            inverse = np.ascontiguousarray(np.atleast_2d(inverse_slot), dtype=np.int32)
            if inverse.shape != dual.shape or np.any(inverse < 0) or np.any(inverse >= self.machines):
                raise ValueError("Inverse slot indices must match the batch of machine duals")
        cp = self.cp
        with self.device:
            if batch > self.capacity:
                self._predecessor = cp.empty((batch, self.eligible_count), dtype=cp.int32)
                self._costs = cp.empty((batch, self.products), dtype=cp.float64)
                self._paths = cp.empty((batch, self.operations), dtype=cp.int32)
                self.capacity = batch
            device_inverse = self._identity if inverse is None else cp.asarray(inverse)
            sizes = (
                0 if inverse is None else self.machines,
                self.products,
                self.operations,
                self.machines,
                self.eligible_count,
                self.max_width,
            )
            self._kernel(
                (batch * self.products,),
                (self.threads,),
                (
                    *self._routes,
                    cp.asarray(dual),
                    device_inverse,
                    *(np.int32(x) for x in sizes),
                    self._predecessor,
                    self._costs,
                    self._paths,
                ),
                shared_mem=self.shared_bytes,
            )
            costs = self._costs[:batch].get()
            paths = self._paths[:batch].get()
        if not np.isfinite(costs).all():
            raise RuntimeError("CUDA path pricing returned nonfinite costs")
        return (costs[0], paths[0]) if single else (costs, paths)

    def lower_bounds(self, costs, machine_dual):
        """Round FP64 prices down by a uniform bound over all eligible paths.

        Each route uses at most 3L multiply/add operations. The absolute
        sum of its terms is bounded by L(max(distance)+max(tau)*max(abs(dual))).
        The minimum of approximate path costs inherits the same bound.
        """
        dual, single = self._duals(machine_dual)
        prices = np.atleast_2d(np.asarray(costs, dtype=np.float64))
        if prices.shape != (len(dual), self.products) or not np.isfinite(prices).all():
            raise ValueError("Product prices must match the batch of machine duals")

        def up(value):
            return np.nextafter(value, np.inf)

        max_penalty = up(np.abs(dual).max(1)[:, None] * self.max_tau[None])
        scale = up(self.route_lengths[None] * up(self.max_distance + max_penalty))
        count = 3 * self.route_lengths
        eps = np.finfo(np.float64).eps
        gamma = up(count * eps / (1.0 - count * eps))
        error = up(up(scale * gamma[None]) + 2 * count * np.finfo(np.float64).tiny)
        if not np.isfinite(error).all():
            raise RuntimeError("FP64 path price error bound overflowed")
        lower = np.nextafter(prices - error, -np.inf)
        return lower[0] if single else lower
