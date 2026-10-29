#pragma once

#include "config.hpp"
#include "validation.hpp"
#include <cmath>
#include <vector>

namespace reference {

inline std::vector<double> apply_depth(const water::Config& config, const std::vector<double>& divergence) {
    constexpr double pi = 3.14159265358979323846;
    const auto& g = config.grid;
    std::vector<double> result(g.nx * g.ny);
    for (int n = 0; n < g.ny; ++n) {
        for (int m = 0; m < g.nx; ++m) {
            if (m == 0 && n == 0)
                continue;
            double coefficient = 0;
            for (int y = 0; y < g.ny; ++y)
                for (int x = 0; x < g.nx; ++x)
                    coefficient += divergence[y * g.nx + x] * std::cos(pi * m * (x + 0.5) / g.nx) *
                                   std::cos(pi * n * (y + 0.5) / g.ny);
            coefficient *= (m ? 2.0 : 1.0) * (n ? 2.0 : 1.0) / (g.nx * g.ny);
            const double k = std::hypot(pi * m / g.lx, pi * n / g.lz);
            const double sx = 2 * g.nx / double(g.lx) * std::sin(pi * m / (2 * g.nx));
            const double sz = 2 * g.ny / double(g.lz) * std::sin(pi * n / (2 * g.ny));
            coefficient *= k * std::tanh(k * config.depth) / (sx * sx + sz * sz);
            for (int y = 0; y < g.ny; ++y)
                for (int x = 0; x < g.nx; ++x)
                    result[y * g.nx + x] += coefficient * std::cos(pi * m * (x + 0.5) / g.nx) *
                                            std::cos(pi * n * (y + 0.5) / g.ny);
        }
    }
    return result;
}

inline water::Snapshot step(const water::Config& config, const water::Snapshot& old, float dt, float damping,
                            const std::vector<double>& pressure_head = {}) {
    auto state = old;
    const auto& g = config.grid;
    const double dx = double(g.lx) / g.nx, dz = double(g.lz) / g.ny;
    const auto head = [&](int i) { return old.eta[i] + (pressure_head.empty() ? 0 : pressure_head[i]); };
    const double r = 1 / (1 + double(damping) * dt);
    for (int y = 0; y < g.ny; ++y)
        for (int x = 0; x <= g.nx; ++x) {
            const int i = y * (g.nx + 1) + x;
            state.u[i] = x == 0 || x == g.nx
                             ? 0
                             : float(r * (old.u[i] - config.gravity * double(dt) *
                                                         (head(y * g.nx + x) - head(y * g.nx + x - 1)) / dx));
        }
    for (int y = 0; y <= g.ny; ++y)
        for (int x = 0; x < g.nx; ++x) {
            const int i = y * g.nx + x;
            state.v[i] =
                y == 0 || y == g.ny
                    ? 0
                    : float(r * (old.v[i] - config.gravity * double(dt) * (head(i) - head(i - g.nx)) / dz));
        }
    std::vector<double> divergence(g.nx * g.ny);
    for (int y = 0; y < g.ny; ++y)
        for (int x = 0; x < g.nx; ++x)
            divergence[y * g.nx + x] =
                (state.u[y * (g.nx + 1) + x + 1] - double(state.u[y * (g.nx + 1) + x])) / dx +
                (state.v[(y + 1) * g.nx + x] - double(state.v[y * g.nx + x])) / dz;
    const auto response = apply_depth(config, divergence);
    for (std::size_t i = 0; i < response.size(); ++i)
        state.eta[i] -= float(dt * response[i]);
    return state;
}

} // namespace reference
