// One block solves one complete product route for one layout. Each thread
// prices a destination machine against every eligible predecessor. The two
// value vectors and slot indices stay in shared memory between layers.
extern "C" __global__ void price_paths(
    const int* __restrict__ product_offsets,
    const int* __restrict__ operation_offsets,
    const int* __restrict__ eligible_machines,
    const double* __restrict__ tau,
    const double* __restrict__ distance,
    const double* __restrict__ machine_dual,
    const int* __restrict__ inverse_slot,
    int inverse_stride,
    int products,
    int operations,
    int machines,
    int eligible_count,
    int max_width,
    int* __restrict__ predecessor,
    double* __restrict__ costs,
    int* __restrict__ paths) {
    const int product = blockIdx.x % products;
    const long long layout = blockIdx.x / products;
    const int first = product_offsets[product];
    const int end = product_offsets[product + 1];
    const double* dual = machine_dual + layout * machines;
    const int* slot = inverse_slot + layout * inverse_stride;
    int* parent = predecessor + layout * eligible_count;

    extern __shared__ double workspace[];
    double* previous = workspace;
    double* next = previous + max_width;
    int* previous_slots = reinterpret_cast<int*>(next + max_width);
    int* next_slots = previous_slots + max_width;
    const double infinity = __longlong_as_double(0x7ff0000000000000LL);

    int begin = operation_offsets[first];
    int width = operation_offsets[first + 1] - begin;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        const int machine = eligible_machines[begin + j];
        previous[j] = -tau[first] * dual[machine];
        previous_slots[j] = slot[machine];
    }
    __syncthreads();

    for (int layer = first + 1; layer < end; ++layer) {
        begin = operation_offsets[layer];
        const int next_width = operation_offsets[layer + 1] - begin;
        for (int j = threadIdx.x; j < next_width; j += blockDim.x) {
            const int machine = eligible_machines[begin + j];
            const int destination = slot[machine];
            double best = infinity;
            int best_source = 0;
            for (int i = 0; i < width; ++i) {
                const long long edge = static_cast<long long>(previous_slots[i]) * machines + destination;
                const double candidate = previous[i] + distance[edge];
                if (candidate < best) {
                    best = candidate;
                    best_source = i;
                }
            }
            next[j] = best - tau[layer] * dual[machine];
            next_slots[j] = destination;
            parent[begin + j] = best_source;
        }
        __syncthreads();
        double* value_buffer = previous;
        previous = next;
        next = value_buffer;
        int* slot_buffer = previous_slots;
        previous_slots = next_slots;
        next_slots = slot_buffer;
        width = next_width;
    }

    if (threadIdx.x == 0) {
        int best = 0;
        for (int j = 1; j < width; ++j) {
            if (previous[j] < previous[best]) {
                best = j;
            }
        }
        costs[layout * products + product] = previous[best];
        for (int layer = end - 1; layer >= first; --layer) {
            const int index = operation_offsets[layer] + best;
            paths[layout * operations + layer] = eligible_machines[index];
            if (layer > first) {
                best = parent[index];
            }
        }
    }
}
