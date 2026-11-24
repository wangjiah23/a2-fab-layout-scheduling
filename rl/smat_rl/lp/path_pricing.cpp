#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// Minimize transport cost minus machine-time duals on every layered route.
extern "C" void price_paths(
    int products, int operations, int machines,
    const int32_t* product_offsets, const int32_t* operation_offsets,
    const int32_t* eligible, const double* tau, const double* distance,
    const double* dual, double* costs, int32_t* paths
) {
    std::vector<double> previous(machines), current(machines);
    std::vector<int32_t> predecessor(operation_offsets[operations]);
    for (int product = 0; product < products; ++product) {
        int first = product_offsets[product];
        int last = product_offsets[product + 1] - 1;
        int start = operation_offsets[first];
        int count = operation_offsets[first + 1] - start;
        for (int i = 0; i < count; ++i) {
            previous[i] = -tau[first] * dual[eligible[start + i]];
        }
        for (int op = first + 1; op <= last; ++op) {
            int next_start = operation_offsets[op];
            int next_count = operation_offsets[op + 1] - next_start;
            std::fill_n(current.begin(), next_count, std::numeric_limits<double>::infinity());
            for (int i = 0; i < count; ++i) {
                const double* row = distance + static_cast<std::size_t>(eligible[start + i]) * machines;
                double prior = previous[i];
                for (int j = 0; j < next_count; ++j) {
                    double candidate = prior + row[eligible[next_start + j]];
                    if (candidate < current[j]) {
                        current[j] = candidate;
                        predecessor[next_start + j] = i;
                    }
                }
            }
            for (int j = 0; j < next_count; ++j) {
                current[j] -= tau[op] * dual[eligible[next_start + j]];
            }
            previous.swap(current);
            start = next_start;
            count = next_count;
        }
        int index = int(std::min_element(previous.begin(), previous.begin() + count) - previous.begin());
        costs[product] = previous[index];
        for (int op = last; op >= first; --op) {
            int position = operation_offsets[op] + index;
            paths[op] = eligible[position];
            index = predecessor[position];
        }
    }
}

extern "C" void alternate_path_edits(
    int products, int machines, int keep,
    const int32_t* product_offsets, const int32_t* operation_offsets,
    const int32_t* eligible, const double* tau, const double* distance,
    const double* dual, const double* allowances, const int32_t* paths,
    int32_t* edit_positions, int32_t* edit_machines
) {
    struct Edit {
        double change;
        int position;
        int machine;
    };
    std::vector<Edit> best;
    best.reserve(keep + 1);
    for (int product = 0; product < products; ++product) {
        const double allowance = allowances[product];
        if (allowance <= 0.) {
            continue;
        }
        const int first = product_offsets[product];
        const int end = product_offsets[product + 1];
        best.clear();
        for (int op = first; op < end; ++op) {
            const int old = paths[op];
            const double* previous = op > first
                ? distance + static_cast<std::size_t>(paths[op - 1]) * machines : nullptr;
            const int next = op + 1 < end ? paths[op + 1] : -1;
            for (int index = operation_offsets[op]; index < operation_offsets[op + 1]; ++index) {
                const int machine = eligible[index];
                if (machine == old) {
                    continue;
                }
                double change = -tau[op] * (dual[machine] - dual[old]);
                if (previous) {
                    change += previous[machine] - previous[old];
                }
                if (next >= 0) {
                    change += distance[static_cast<std::size_t>(machine) * machines + next]
                        - distance[static_cast<std::size_t>(old) * machines + next];
                }
                if (change >= allowance || (int(best.size()) == keep && change >= best.back().change)) {
                    continue;
                }
                const auto insert = std::upper_bound(best.begin(), best.end(), change,
                    [](double value, const Edit& edit) { return value < edit.change; });
                best.insert(insert, {change, op - first, machine});
                if (int(best.size()) > keep) {
                    best.pop_back();
                }
            }
        }
        for (std::size_t i = 0; i < best.size(); ++i) {
            edit_positions[static_cast<std::size_t>(product) * keep + i] = best[i].position;
            edit_machines[static_cast<std::size_t>(product) * keep + i] = best[i].machine;
        }
    }
}

static double transport_cost(int machines, const double* distance, const int32_t* path, int length) {
    double cost = 0.;
    for (int j = 1; j < length; ++j) {
        cost += distance[static_cast<std::size_t>(path[j - 1]) * machines + path[j]];
    }
    return cost;
}

extern "C" void path_column_costs(
    int count, int machines, const int32_t* offsets, const int32_t* paths,
    const double* distance, double* costs
) {
    for (int i = 0; i < count; ++i) {
        costs[i] = transport_cost(machines, distance, paths + offsets[i], offsets[i + 1] - offsets[i]);
    }
}

extern "C" void path_column_coefficients(
    int count, int products, int machines, const int32_t* product_offsets,
    const int32_t* column_products, const int32_t* offsets, const int32_t* paths, const double* tau,
    int32_t* starts, int32_t* rows, double* coefficients
) {
    std::vector<double> usage(machines, 0.);
    int entries = 0;
    for (int i = 0; i < count; ++i) {
        const int product = column_products[i];
        const double* times = tau + product_offsets[product];
        starts[i] = entries;
        rows[entries] = product;
        coefficients[entries++] = 1.;
        for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
            usage[paths[j]] += times[j - offsets[i]];
        }
        for (int machine = 0; machine < machines; ++machine) {
            if (usage[machine] > 0.) {
                rows[entries] = products + machine;
                coefficients[entries++] = usage[machine];
                usage[machine] = 0.;
            }
        }
    }
    starts[count] = entries;
}
