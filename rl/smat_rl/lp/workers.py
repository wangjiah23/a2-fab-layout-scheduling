"""Parallel path masters with one batched CUDA pricer per training rank."""

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from queue import Empty, Queue
from threading import Thread, local

import numpy as np

from .paths import PathLP, _cpu_kernel


class PricingBatch:
    def __init__(self, solver, device, workers):
        self.requests = Queue()
        self.thread = Thread(target=self.run, args=(solver, device, workers), daemon=True)
        self.thread.start()

    def price(self, dual, inverse):
        answer = Future()
        self.requests.put((dual, inverse, answer))
        return answer.result()

    def run(self, solver, device, workers):
        try:
            from .gpu_path_pricing import CudaPathPricer

            pricer = CudaPathPricer(
                solver.distance,
                solver.product_offsets,
                solver.operation_offsets,
                solver.eligible_machines,
                solver.tau,
                device=device,
            )
            with pricer.device:
                stream = pricer.cp.cuda.Stream(non_blocking=True)
            error = None
        except Exception as exc:
            error = exc
            self.report_fallback(exc, device)
        while (first := self.requests.get()) is not None:
            batch = [first]
            if error is None:
                until = time.monotonic() + 0.0005
                while len(batch) < workers:
                    try:
                        batch.append(self.requests.get(timeout=max(0.0, until - time.monotonic())))
                    except Empty:
                        break
                duals, inverses, answers = zip(*batch)
                try:
                    with pricer.device, stream:
                        costs, paths = pricer.price(np.stack(duals), np.stack(inverses))
                    for answer, cost, path in zip(answers, costs, paths):
                        answer.set_result((cost, path))
                    continue
                except Exception as exc:
                    error = exc
                    self.report_fallback(exc, device)
            for _, _, answer in batch:
                answer.set_exception(RuntimeError(f"CUDA pricing unavailable: {error}"))

    @staticmethod
    def report_fallback(error, device):
        print(
            json.dumps({"event": "cuda_pricing_fallback", "device": device, "error": str(error)}), flush=True
        )

    def close(self):
        self.requests.put(None)
        self.thread.join()


class SlotPathWorkers:
    def __init__(self, slots, seed, workers, cuda_device=None):
        _cpu_kernel()
        self.prototype = PathLP(slots, seed_flow=seed, columns_per_product=3, retain_columns=512)
        self.slots = slots
        self.groups = slots.data["group_ids"]
        self.reference = slots.reference_cost
        self.worker = local()
        self.pricing = None if cuda_device is None else PricingBatch(self.prototype, cuda_device, workers)
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, permutation):
        return self.pool.submit(self.evaluate, permutation)

    def fresh_solver(self):
        solver = copy(self.prototype)
        solver.paths = self.prototype.paths.copy()
        solver.path_keys = self.prototype.path_keys.copy()
        return solver

    def evaluate(self, permutation):
        layout = self.slots.validate_layout(permutation)
        if not hasattr(self.worker, "solver"):
            self.worker.solver = self.fresh_solver()
        solver = self.worker.solver
        cpu_calls = gpu_calls = 0

        def price(distance, dual):
            nonlocal cpu_calls, gpu_calls
            try:
                answer = self.pricing.price(dual, inverse)
                gpu_calls += 1
                return answer
            except RuntimeError:
                cpu_calls += 1
                return solver.price(distance, dual)

        inverse = np.argsort(layout).astype(np.int32) if self.pricing else None
        first_error = None
        arc_fallback = False
        started = time.perf_counter()
        try:
            result = solver.solve(layout, pricing=price if self.pricing else None)
        except RuntimeError as exc:
            first_error = str(exc)
            print(json.dumps({"event": "lp_cold_retry", "error": first_error}), flush=True)
            # Drop the entire previous basis and column bank. Use independent
            # CPU pricing and different column additions on the second solve.
            solver = self.worker.solver = self.fresh_solver()
            solver.columns_per_product = 4
            try:
                result = solver.solve(layout)
            except RuntimeError as retry_error:
                self.worker.solver = self.fresh_solver()
                arc_fallback = True
                print(json.dumps({"event": "lp_arc_fallback", "error": str(retry_error)}), flush=True)
                try:
                    result = self.slots.solve_arcs(layout)
                except RuntimeError as arc_error:
                    return {
                        "status": "uncertified",
                        "error": f"{first_error}; fresh path LP: {retry_error}; complete arc LP: {arc_error}",
                        "solver_seconds": time.perf_counter() - started,
                        "cold_retries": 1,
                        "gpu_pricing_calls": gpu_calls,
                        "cpu_pricing_fallback_calls": cpu_calls,
                    }
            solver.columns_per_product = self.prototype.columns_per_product
        return {
            **result,
            "improvement": 1 - result["cost"] / self.reference,
            "device_at_slot": layout.tolist(),
            "changed_slots": int(np.count_nonzero(self.groups[layout] != self.groups)),
            "pricing_backend": "arc-cpu" if arc_fallback else ("cuda" if gpu_calls else "cpu"),
            "gpu_pricing_calls": gpu_calls,
            "cpu_pricing_fallback_calls": cpu_calls,
            "cold_retries": int(first_error is not None),
            "solver_seconds": time.perf_counter() - started,
        }

    def shutdown(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.pricing is not None:
            self.pricing.close()
