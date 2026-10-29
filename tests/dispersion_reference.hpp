#pragma once

#include "config.hpp"
#include "validation.hpp"
#include <cmath>
#include <vector>

namespace reference {

inline std::vector<double> apply_depth(const water::Config& config, const std::vector<double>& divergence) {
    constexpr double pi = 3.14159265358979323846;
    const auto& g = config.grid;
    std::vector<double> cx(g.nx * g.nx), cy(g.ny * g.ny);
    for (int m = 0; m < g.nx; ++m)
        for (int x = 0; x < g.nx; ++x) cx[m * g.nx + x] = std::cos(pi * m * (x + .5) / g.nx);
    for (int n = 0; n < g.ny; ++n)
        for (int y = 0; y < g.ny; ++y) cy[n * g.ny + y] = std::cos(pi * n * (y + .5) / g.ny);
    std::vector<double> horizontal(g.nx * g.ny), coefficients(g.nx * g.ny), vertical(g.nx * g.ny), result(g.nx * g.ny);
    for (int y = 0; y < g.ny; ++y)
        for (int m = 0; m < g.nx; ++m)
            for (int x = 0; x < g.nx; ++x)
                horizontal[y * g.nx + m] += divergence[y * g.nx + x] * cx[m * g.nx + x];
    for (int n = 0; n < g.ny; ++n)
        for (int m = 0; m < g.nx; ++m) {
            if (m == 0 && n == 0) continue;
            double coefficient = 0;
            for (int y = 0; y < g.ny; ++y) coefficient += horizontal[y * g.nx + m] * cy[n * g.ny + y];
            const double k = std::hypot(pi * m / g.lx, pi * n / g.lz);
            const double sx = 2.0 * g.nx / g.lx * std::sin(pi * m / (2 * g.nx));
            const double sz = 2.0 * g.ny / g.lz * std::sin(pi * n / (2 * g.ny));
            coefficients[n * g.nx + m] = coefficient * (m ? 2 : 1) * (n ? 2 : 1) / (g.nx * g.ny) *
                k * std::tanh(k * config.depth) / (sx * sx + sz * sz);
        }
    for (int y = 0; y < g.ny; ++y)
        for (int m = 0; m < g.nx; ++m)
            for (int n = 0; n < g.ny; ++n)
                vertical[y * g.nx + m] += coefficients[n * g.nx + m] * cy[n * g.ny + y];
    for (int y = 0; y < g.ny; ++y)
        for (int x = 0; x < g.nx; ++x)
            for (int m = 0; m < g.nx; ++m)
                result[y * g.nx + x] += vertical[y * g.nx + m] * cx[m * g.nx + x];
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
