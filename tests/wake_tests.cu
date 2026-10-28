#include "validation.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace {
water::Config fixture() {
    water::Config c;
    c.grid = {80, 64, 4, 3};
    c.impact_radius = .15f;
    return c;
}
double ramp(double x) {
    x = std::clamp(x, 0.0, 1.0);
    return x * x * (3 - 2 * x);
}
double head(double x, double z, const water::BoatState& b, const water::Config& c) {
    const double vx = b.vx, vz = b.vz, yaw = b.yaw;
    const double speed = std::hypot(vx, vz);
    if (speed <= 1e-4)
        return 0;
    const double cell = std::max(double(c.grid.lx) / c.grid.nx, double(c.grid.lz) / c.grid.ny);
    const double length = std::max(double(water::boat_geometry.length), 4 * cell);
    const double width = std::max(double(water::boat_geometry.beam), 2 * cell);
    const double longitudinal = (x - b.x) * std::cos(yaw) + (z - b.z) * std::sin(yaw);
    const double transverse = -(x - b.x) * std::sin(yaw) + (z - b.z) * std::cos(yaw);
    auto bump = [](double q) { return std::abs(q) < 1 ? std::pow(1 - q * q, 3) : 0.0; };
    auto pattern = [&](double l) {
        return bump((l - .25 * length) / (.5 * length)) * bump(transverse / width) -
               .5 * bump((l + .35 * length) / (.75 * length)) * bump(transverse / (1.5 * width));
    };
    const double blend = ramp((vx * std::cos(yaw) + vz * std::sin(yaw) + .01) / .02);
    return .002 * (speed * speed / 2.25) * ramp((speed - .0001) / .0099) *
           std::min(1.0, double(c.depth) / 1.0) *
           (blend * pattern(longitudinal) + (1 - blend) * pattern(-longitudinal));
}
water::Snapshot forced(water::Simulation& s, const water::Config& c, water::BoatState b, float dt,
                       float damping) {
    s.reset();
    water::upload_boat(s, b);
    s.step(dt, damping);
    b = water::download_boat(s);
    const auto got = water::download(s);
    const auto& g = c.grid;
    const double dx = double(g.lx) / g.nx, dz = double(g.lz) / g.ny;
    std::vector<double> p(g.nx * g.ny), u(got.u.size()), v(got.v.size());
    for (int z = 0; z < g.ny; ++z)
        for (int x = 0; x < g.nx; ++x)
            p[z * g.nx + x] = head((x + .5) * dx, (z + .5) * dz, b, c);
    for (int z = 0; z < g.ny; ++z)
        for (int x = 1; x < g.nx; ++x)
            u[z * (g.nx + 1) + x] =
                -c.gravity * dt * (p[z * g.nx + x] - p[z * g.nx + x - 1]) / dx / (1 + damping * dt);
    for (int z = 1; z < g.ny; ++z)
        for (int x = 0; x < g.nx; ++x)
            v[z * g.nx + x] = -c.gravity * dt * (p[z * g.nx + x] - p[(z - 1) * g.nx + x]) / dz /
                              (1 + damping * dt);
    auto close = [](double actual, double expected) {
        CHECK(std::abs(actual - expected) <= 2e-6 + 2e-5 * std::abs(expected));
    };
    for (size_t i = 0; i < u.size(); ++i)
        close(got.u[i], u[i]);
    for (size_t i = 0; i < v.size(); ++i)
        close(got.v[i], v[i]);
    for (int z = 0; z < g.ny; ++z)
        for (int x = 0; x < g.nx; ++x) {
            double eta = -c.depth * dt *
                         ((u[z * (g.nx + 1) + x + 1] - u[z * (g.nx + 1) + x]) / dx +
                          (v[(z + 1) * g.nx + x] - v[z * g.nx + x]) / dz);
            close(got.eta[z * g.nx + x], eta);
        }
    return got;
}
double intensity(const water::Snapshot& s) {
    double sum = 0;
    for (const auto* a : {&s.u, &s.v})
        for (float x : *a)
            sum += double(x) * x;
    return sum;
}
} // namespace
TEST_CASE("wake", independent_pressure_gradient_and_continuity) {
    for (bool transpose : {false, true}) {
        auto c = fixture();
        if (transpose) {
            std::swap(c.grid.nx, c.grid.ny);
            std::swap(c.grid.lx, c.grid.lz);
        }
        water::Simulation s(c);
        for (float damping : {0.f, 2.5f})
            for (auto b : {water::BoatState{1.5f, 1.5f, .4f, .9f, .3f, 0},
                           {1.5f, 1.5f, 0, -.75f, 0, 0},
                           {.15f, .06f, 0, .4f, .3f, 0},
                           {1.5f, 1.5f, 0, 0, .3f, 0}})
                CHECK(intensity(forced(s, c, b, .001f, damping)) > 1e-10);
    }
}
TEST_CASE("wake", rest_coasting_speed_and_lateral_continuity) {
    auto c = fixture();
    water::Simulation s(c);
    auto rest = forced(s, c, {2, 1.5f, 0, 0, 0, 0}, .001f, 2.5f);
    CHECK(intensity(rest) == 0);
    for (float e : rest.eta)
        CHECK(e == 0);
    auto insignificant = forced(s, c, {2, 1.5f, 0, .0001f, 0, 0}, .001f, 0);
    CHECK(intensity(insignificant) == 0);
    for (float e : insignificant.eta)
        CHECK(e == 0);
    auto slow = forced(s, c, {2, 1.5f, 0, .375f, 0, 0}, .001f, 0);
    auto fast = forced(s, c, {2, 1.5f, 0, .75f, 0, 0}, .001f, 0);
    CHECK(intensity(fast) > 15 * intensity(slow));
    auto forward = forced(s, c, {2, 1.5f, 0, 1.5f, 0, 0}, .001f, 0);
    CHECK(intensity(forward) > 15 * intensity(fast));
    auto middle = forced(s, c, {2, 1.5f, 0, 0, .3f, 0}, .001f, 0);
    for (float vx : {-1e-6f, 1e-6f}) {
        auto side = forced(s, c, {2, 1.5f, 0, vx, .3f, 0}, .001f, 0);
        CHECK(intensity(side) > 0);
        for (size_t i = 0; i < side.u.size(); ++i)
            CHECK(std::abs(side.u[i] - middle.u[i]) < 1e-7);
    }
    auto reverse = forced(s, c, {2, 1.5f, 0, -.75f, 0, 0}, .001f, 0);
    CHECK(std::abs(intensity(reverse) / intensity(fast) - 1) < .02);
    for (int z = 0; z < c.grid.ny; ++z) {
        for (int x = 0; x < c.grid.nx; ++x)
            CHECK_NEAR(reverse.eta[z * c.grid.nx + x], fast.eta[z * c.grid.nx + c.grid.nx - 1 - x],
                       2e-6);
        for (int x = 0; x <= c.grid.nx; ++x)
            CHECK_NEAR(reverse.u[z * (c.grid.nx + 1) + x],
                       -fast.u[z * (c.grid.nx + 1) + c.grid.nx - x], 2e-6);
    }
}

namespace {
void boat_soak(water::Config c) {
    water::Simulation s(c);
    const int subdivisions = water::substeps_per_quantum(c);
    const float dt = float((1.0 / 120) / subdivisions);
    double maximum_mean = 0;
    float peak = 0;
    for (int second = 1; second <= 120; ++second) {
        // Each leg starts with the real moving boat near a different wall/corner.
        // Water is never reset, so every forcing episode accumulates in the same field.
        if (second % 15 == 1) {
            const int leg = (second - 1) / 15;
            const float yaw = float(leg) * .7853981634f;
            const float ex = .5f * (water::boat_geometry.length * std::abs(std::cos(yaw)) +
                                    water::boat_geometry.beam * std::abs(std::sin(yaw)));
            const float ez = .5f * (water::boat_geometry.length * std::abs(std::sin(yaw)) +
                                    water::boat_geometry.beam * std::abs(std::cos(yaw)));
            const float x = (leg & 1) ? ex + .001f : c.grid.lx - ex - .001f;
            const float z = (leg & 2) ? ez + .001f : c.grid.lz - ez - .001f;
            water::upload_boat(s, {x, z, yaw, 1.5f * std::cos(yaw), 1.5f * std::sin(yaw), 0});
        }
        for (int tick = 0; tick < 120; ++tick) {
            water::BoatControl control{1, 0};
            if (second > 40)
                control.rudder = 1;
            if (second > 80)
                control = {-1, -1};
            for (int sub = 0; sub < subdivisions; ++sub)
                s.step(dt, c.damping, control);
        }
        const auto state = water::download(s);
        for (const auto* values : {&state.eta, &state.u, &state.v})
            for (float value : *values)
                CHECK(std::isfinite(value));
        for (int z = 0; z < c.grid.ny; ++z) {
            CHECK(state.u[z * (c.grid.nx + 1)] == 0);
            CHECK(state.u[z * (c.grid.nx + 1) + c.grid.nx] == 0);
        }
        for (int x = 0; x < c.grid.nx; ++x) {
            CHECK(state.v[x] == 0);
            CHECK(state.v[c.grid.ny * c.grid.nx + x] == 0);
        }
        const double mean =
            std::accumulate(state.eta.begin(), state.eta.end(), 0.0) / state.eta.size();
        maximum_mean = std::max(maximum_mean, std::abs(mean));
        CHECK(std::abs(mean) <= 1e-6 * std::max(c.depth, .25f));
        for (float eta : state.eta)
            peak = std::max(peak, std::abs(eta));
        CHECK(peak < .02f * c.depth);
    }
    std::cout << "BOAT SOAK " << c.grid.nx << 'x' << c.grid.ny << " H = " << c.depth
              << " damping = " << c.damping << " seconds = 120 max_mean = " << maximum_mean
              << " peak = " << peak << '\n';
}
} // namespace
TEST_CASE("wake", prolonged_boat_conserves_volume_and_stays_small) {
    for (float depth : {.25f, 3.f, 6.f})
        for (float damping : {0.f, 2.5f})
            for (auto grid : {water::Grid{64, 48, 20, 12}, water::Grid{48, 64, 12, 20},
                              water::Grid{512, 512, 20, 12}}) {
                auto c = fixture();
                c.grid = grid;
                c.depth = depth;
                c.damping = damping;
                c.impact_radius = .9f;
                c.impact_strength = .001f;
                boat_soak(c);
            }
}
