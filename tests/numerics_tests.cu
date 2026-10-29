#include "dispersion_reference.hpp"
#include "simulation.hpp"
#include "validation.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

namespace {

constexpr double pi = 3.14159265358979323846;

water::Config fixture() {
    water::Config config;
    config.grid = {96, 64, 20.0f, 12.0f};
    config.impact_radius = 0.75f;
    config.damping = 0.0f;
    config.depth = 1;
    return config;
}

water::Snapshot zero_state(const water::Config& config) {
    const auto& g = config.grid;
    return {std::vector<float>(g.nx * g.ny),
            std::vector<float>((g.nx + 1) * g.ny),
            std::vector<float>(g.nx * (g.ny + 1))};
}

float time_step(const water::Config& config) {
    return static_cast<float>((1.0 / 120.0) / water::substeps_per_quantum(config));
}

water::Snapshot mode(const water::Config& config, bool along_x) {
    auto state = zero_state(config);
    const auto& g = config.grid;
    for (int y = 0; y < g.ny; ++y) {
        for (int x = 0; x < g.nx; ++x) {
            const double coordinate = along_x ? (x + 0.5) / g.nx : (y + 0.5) / g.ny;
            state.eta[y * g.nx + x] = static_cast<float>(0.001 * std::cos(pi * coordinate));
        }
    }
    return state;
}

double amplitude(const water::Snapshot& state, const water::Config& config, bool along_x) {
    const auto& g = config.grid;
    double projection = 0.0;
    for (int y = 0; y < g.ny; ++y) {
        for (int x = 0; x < g.nx; ++x) {
            const double coordinate = along_x ? (x + 0.5) / g.nx : (y + 0.5) / g.ny;
            projection += state.eta[y * g.nx + x] * std::cos(pi * coordinate);
        }
    }
    return 2.0 * projection / (g.nx * g.ny);
}

void check_walls(const water::Snapshot& state, const water::Config& config) {
    const auto& g = config.grid;
    for (int y = 0; y < g.ny; ++y) {
        CHECK(state.u[y * (g.nx + 1)] == 0.0f);
        CHECK(state.u[y * (g.nx + 1) + g.nx] == 0.0f);
    }
    for (int x = 0; x < g.nx; ++x) {
        CHECK(state.v[x] == 0.0f);
        CHECK(state.v[g.ny * g.nx + x] == 0.0f);
    }
}

double energy(const water::Snapshot& state, const water::Config& config) {
    const auto& g = config.grid;
    const double k = pi / g.lx;
    const double keff = 2 * g.nx / double(g.lx) * std::sin(pi / (2 * g.nx));
    const double modal_depth = k * std::tanh(k * config.depth) / (keff * keff);
    double total = 0.0;
    for (float height : state.eta) {
        CHECK(std::isfinite(height));
        total += config.gravity * static_cast<double>(height) * height;
    }
    for (int y = 0; y < g.ny; ++y) {
        for (int x = 0; x <= g.nx; ++x) {
            const double velocity = state.u[y * (g.nx + 1) + x];
            CHECK(std::isfinite(velocity));
            const double weight = (x == 0 || x == g.nx) ? 0.5 : 1.0;
            total += weight * modal_depth * velocity * velocity;
        }
    }
    for (int y = 0; y <= g.ny; ++y) {
        for (int x = 0; x < g.nx; ++x) {
            const double velocity = state.v[y * g.nx + x];
            CHECK(std::isfinite(velocity));
            const double weight = (y == 0 || y == g.ny) ? 0.5 : 1.0;
            total += weight * modal_depth * velocity * velocity;
        }
    }
    return total * (g.lx / g.nx) * (g.lz / g.ny);
}

void check_period(bool along_x, bool transpose, float depth = 1) {
    auto config = fixture();
    config.depth = depth;
    if (transpose) std::swap(config.grid.nx, config.grid.ny);
    water::Simulation simulation(config);
    water::upload(simulation, mode(config, along_x));
    const double k = pi / (along_x ? config.grid.lx : config.grid.lz);
    const double expected = 2 * pi / std::sqrt(config.gravity * k * std::tanh(k * config.depth));
    const float dt = time_step(config);
    double previous = 0.001;
    std::vector<double> crossings;
    for (int step = 1; step * dt < 2.5 * expected; ++step) {
        simulation.step(dt, 0.0f);
        const double current = amplitude(water::download(simulation), config, along_x);
        if (previous > 0.0 && current <= 0.0) {
            crossings.push_back((step - 1 + previous / (previous - current)) * dt);
        }
        previous = current;
    }
    CHECK(crossings.size() >= 2);
    const double measured = crossings[1] - crossings[0];
    std::cout << "period " << (along_x ? 'x' : 'z') << ' ' << config.grid.nx << 'x'
              << config.grid.ny << ": " << measured << " expected " << expected << '\n';
    CHECK(std::abs(measured / expected - 1.0) < 0.03);
}

} // namespace

TEST_CASE("numerics", rest_stays_exactly_zero_for_1000_steps) {
    const auto config = fixture();
    water::Simulation simulation(config);
    for (int i = 0; i < 1000; ++i) simulation.step(time_step(config), 0.15f);
    const auto state = water::download(simulation);
    for (const auto* values : {&state.eta, &state.u, &state.v}) {
        CHECK(std::all_of(values->begin(), values->end(), [](float v) { return v == 0.0f; }));
    }
}

TEST_CASE("numerics", reset_clears_every_staggered_array) {
    const auto config = fixture();
    water::Simulation simulation(config);
    auto state = zero_state(config);
    for (auto* values : {&state.eta, &state.u, &state.v}) {
        std::fill(values->begin(), values->end(), 0.125f);
    }
    water::upload(simulation, state);
    simulation.reset();
    state = water::download(simulation);
    for (const auto* values : {&state.eta, &state.u, &state.v}) {
        CHECK(std::all_of(values->begin(), values->end(), [](float v) { return v == 0.0f; }));
    }
}

TEST_CASE("numerics", wall_faces_are_zero_even_after_nonzero_upload) {
    const auto config = fixture();
    auto state = mode(config, true);
    std::fill(state.u.begin(), state.u.end(), 0.01f);
    std::fill(state.v.begin(), state.v.end(), -0.02f);
    water::Simulation simulation(config);
    water::upload(simulation, state);
    simulation.step(time_step(config), 0.0f);
    check_walls(water::download(simulation), config);
}

TEST_CASE("numerics", radial_wave_preserves_reflections_and_axis_symmetry) {
    auto config = fixture();
    config.grid = {96, 96, 12.0f, 12.0f};
    auto state = zero_state(config);
    for (int y = 0; y < 96; ++y) {
        for (int x = 0; x < 96; ++x) {
            const double px = (x + 0.5) / 8.0 - 6.0;
            const double pz = (y + 0.5) / 8.0 - 6.0;
            state.eta[y * 96 + x] = static_cast<float>(0.001 * std::exp(-(px * px + pz * pz)));
        }
    }
    water::Simulation simulation(config);
    water::upload(simulation, state);
    for (int i = 0; i < 600; ++i) simulation.step(time_step(config), 0.0f);
    state = water::download(simulation);
    double peak = 0.0, difference = 0.0;
    for (int y = 0; y < 96; ++y) {
        for (int x = 0; x < 96; ++x) {
            const double value = state.eta[y * 96 + x];
            peak = std::max(peak, std::abs(value));
            for (int reflected : {y * 96 + 95 - x, (95 - y) * 96 + x, x * 96 + y}) {
                difference = std::max(difference, std::abs(value - state.eta[reflected]));
            }
        }
    }
    std::cout << "symmetry peak-relative difference: " << difference / peak << '\n';
    CHECK(peak > 1e-5);
    CHECK(difference / peak < 1e-5);
}

TEST_CASE("numerics", rectangular_grid_x_mode_has_physical_period) { check_period(true, false); }
TEST_CASE("numerics", rectangular_grid_z_mode_has_physical_period) { check_period(false, false); }
TEST_CASE("numerics", transposed_grid_x_mode_has_physical_period) { check_period(true, true); }
TEST_CASE("numerics", transposed_grid_z_mode_has_physical_period) { check_period(false, true); }

TEST_CASE("numerics", damping_reduces_modal_energy_below_40_percent_after_five_seconds) {
    const auto config = fixture();
    water::Simulation control(config), damped(config);
    const auto initial = mode(config, true);
    water::upload(control, initial);
    water::upload(damped, initial);
    const float dt = time_step(config);
    for (int i = 0; i < static_cast<int>(std::ceil(5.0 / dt)); ++i) {
        control.step(dt, 0.0f);
        damped.step(dt, 0.5f);
    }
    const double ratio = energy(water::download(damped), config) / energy(water::download(control), config);
    std::cout << "damped energy ratio: " << ratio << '\n';
    CHECK(ratio < 0.4);
}

TEST_CASE("numerics", undamped_energy_stays_bounded_for_ten_periods) {
    const auto config = fixture();
    water::Simulation simulation(config);
    const auto initial = mode(config, true);
    water::upload(simulation, initial);
    const double initial_energy = energy(initial, config);
    const double k = pi / config.grid.lx;
    const double period = 2 * pi / std::sqrt(config.gravity * k * std::tanh(k * config.depth));
    const float dt = time_step(config);
    double maximum = initial_energy;
    for (int i = 0; i < static_cast<int>(std::ceil(10.0 * period / dt)); ++i) {
        simulation.step(dt, 0.0f);
        maximum = std::max(maximum, energy(water::download(simulation), config));
    }
    std::cout << "maximum energy ratio: " << maximum / initial_energy << '\n';
    CHECK(maximum / initial_energy <= 1.25);
}

TEST_CASE("numerics", invalid_steps_and_snapshot_sizes_are_rejected_without_state_changes) {
    const auto config = fixture();
    water::Simulation simulation(config);
    const auto initial = mode(config, true);
    water::upload(simulation, initial);
    const float dt = time_step(config);
    for (float invalid : {0.0f, -dt, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN(),
                          static_cast<float>(water::stable_dt(config) * 1.01)}) {
        CHECK_THROWS(simulation.step(invalid, 0.0f), std::invalid_argument);
    }
    for (float invalid : {-0.1f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        CHECK_THROWS(simulation.step(dt, invalid), std::invalid_argument);
    }
    auto truncated = initial;
    truncated.v.pop_back();
    CHECK_THROWS(water::upload(simulation, truncated), std::invalid_argument);
    const auto final = water::download(simulation);
    CHECK(final.eta == initial.eta);
    CHECK(final.u == initial.u);
    CHECK(final.v == initial.v);
}

TEST_CASE("numerics", deeper_water_has_physical_period) { check_period(true, false, 3); }
TEST_CASE("numerics", live_depth_preserves_fields_and_updates_validation_and_cfl) {
    auto config = fixture();
    config.depth = 3;
    water::Simulation simulation(config);
    auto initial = mode(config, true);
    std::fill(initial.u.begin(), initial.u.end(), .001f);
    std::fill(initial.v.begin(), initial.v.end(), -.002f);
    water::upload(simulation, initial);
    simulation.set_depth(.25f);
    auto after = water::download(simulation);
    CHECK(after.eta == initial.eta && after.u == initial.u && after.v == initial.v);
    CHECK(simulation.config().depth == .25f && simulation.config().impact_strength <= .005f);
    const water::Impact too_strong{10, 6, .75f, .006f};
    CHECK_THROWS(simulation.apply_impacts(&too_strong, 1), std::invalid_argument);
    for (float depth : {0.f, -1.f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        CHECK_THROWS(simulation.set_depth(depth), std::invalid_argument);
        CHECK(simulation.config().depth == .25f);
        after = water::download(simulation);
        CHECK(after.eta == initial.eta && after.u == initial.u && after.v == initial.v);
    }
    simulation.set_depth(100);
    float above = std::nextafter(float(water::stable_dt(simulation.config())), std::numeric_limits<float>::infinity());
    CHECK_THROWS(simulation.step(above, 0), std::invalid_argument);
    simulation.reset();
    CHECK(simulation.config().depth == 100);
    for (float depth : {.25f, 1.5178088f, 3.f, 100.f}) {
        simulation.set_depth(depth);
        for (int i = 0; i < 10; ++i) simulation.step(time_step(simulation.config()), 0);
        after = water::download(simulation);
        for (const auto* values : {&after.eta, &after.u, &after.v})
            CHECK(std::all_of(values->begin(), values->end(), [](float v) { return v == 0; }));
    }
}


namespace {
water::Config packet_config(int nx) {
    water::Config c;
    c.grid = {nx, 8, 100, 10}; c.impact_radius = 2.5f; c.depth = 6;
    return c;
}
water::Snapshot packet(const water::Config& c, double center, float dt, int steps) {
    auto state = zero_state(c);
    const auto& g = c.grid;
    for (int m = 36; m <= 92; ++m) {
        const double k = pi * m / g.lx;
        const double keff = 2 * g.nx / double(g.lx) * std::sin(pi * m / (2 * g.nx));
        const double omega = std::sqrt(c.gravity * k * std::tanh(k * c.depth));
        const double theta = 2 * std::asin(.5 * omega * dt);
        const double a = .0001 * std::exp(-.5 * std::pow((m - 64) / 5.0, 2));
        const double phase = k * center + steps * theta;
        // Stored momentum lags height by half the discrete modal phase.
        for (int y = 0; y < g.ny; ++y) {
            for (int x = 0; x < g.nx; ++x)
                state.eta[y * g.nx + x] += float(a * std::cos(phase) * std::cos(k * (x + .5) * g.lx / g.nx));
            for (int x = 1; x < g.nx; ++x)
                state.u[y * (g.nx + 1) + x] += float(a * c.gravity * keff / omega *
                    std::sin(phase - .5 * theta) * std::sin(pi * m * x / g.nx));
        }
    }
    return state;
}
double packet_centroid(const water::Snapshot& s, const water::Config& c) {
    const auto& g = c.grid;
    const double k = pi * 64 / g.lx;
    const double keff = 2 * g.nx / double(g.lx) * std::sin(pi * 64 / (2 * g.nx));
    const double impedance = std::sqrt(c.gravity * k * std::tanh(k * c.depth)) / (c.gravity * keff);
    double sum = 0, weighted = 0;
    for (int x = 0; x < g.nx; ++x) {
        const double velocity = .5 * (s.u[x] + s.u[x + 1]);
        const double energy = s.eta[x] * double(s.eta[x]) + std::pow(impedance * velocity, 2);
        sum += energy; weighted += energy * (x + .5) * g.lx / g.nx;
    }
    return weighted / sum;
}
double packet_phase(const water::Snapshot& s, const water::Config& c, float dt) {
    const auto& g = c.grid;
    const double k = pi * 64 / g.lx;
    const double keff = 2 * g.nx / double(g.lx) * std::sin(pi * 64 / (2 * g.nx));
    const double omega = std::sqrt(c.gravity * k * std::tanh(k * c.depth));
    double eta = 0, u = 0;
    for (int x = 0; x < g.nx; ++x) eta += s.eta[x] * std::cos(pi * 64 * (x + .5) / g.nx);
    for (int x = 1; x < g.nx; ++x) u += s.u[x] * std::sin(pi * 64 * x / g.nx) * omega / (c.gravity * keff);
    const double half = std::asin(.5 * omega * dt);
    return std::atan2((u + eta * std::sin(half)) / std::cos(half), eta);
}
}
TEST_CASE("numerics", narrow_band_packet_separates_group_and_crest_phase_speeds) {
    for (int refinement : {1, 2}) {
        const auto c = packet_config(256 * refinement);
        const float dt = time_step(c) / refinement;
        water::Simulation sim(c);
        const auto initial = packet(c, 30, dt, 0);
        water::upload(sim, initial);
        const double start = packet_centroid(initial, c);
        double previous = packet_phase(initial, c, dt), accumulated = 0;
        const int steps = int(4 / dt);
        for (int i = 0; i < steps; ++i) {
            sim.step(dt, 0);
            if ((i + 1) % 20 == 0 || i + 1 == steps) {
                const double phase = packet_phase(water::download(sim), c, dt);
                accumulated += std::remainder(phase - previous, 2 * pi);
                previous = phase;
            }
        }
        const auto final = water::download(sim);
        const double k = pi * 64 / c.grid.lx;
        const double omega = std::sqrt(c.gravity * k * std::tanh(k * c.depth));
        const double phase_speed = omega / k;
        const double kh = k * c.depth;
        const double group_speed = .5 * phase_speed * (1 + 2 * kh / std::sinh(2 * kh));
        const double measured_group = (packet_centroid(final, c) - start) / (steps * dt);
        const double measured_phase = accumulated / (steps * dt * k);
        std::cout << "PACKET refine=" << refinement << " group=" << measured_group << " expected=" << group_speed
                  << " phase=" << measured_phase << " expected=" << phase_speed << '\n';
        CHECK(std::abs(measured_group / group_speed - 1) < .05);
        CHECK(std::abs(measured_phase / phase_speed - 1) < .05);
        CHECK(std::abs(measured_group / measured_phase - .5) < .03);
    }
}
TEST_CASE("numerics", cosine_packet_reflects_at_wall_without_periodic_wrap) {
    const auto c = packet_config(512);
    const float dt = time_step(c);
    water::Simulation sim(c);
    water::upload(sim, packet(c, 90, dt, 0));
    const double k = pi * 64 / c.grid.lx;
    const double group = .5 * std::sqrt(c.gravity / k);
    int completed = 0;
    for (double target : {10 / group, 20 / group}) {
        const int steps = int(target / dt);
        for (; completed < steps; ++completed) sim.step(dt, 0);
        const auto got = water::download(sim), expected = packet(c, 90, dt, completed);
        double error = 0, norm = 0, left = 0, right = 0;
        for (int x = 0; x < c.grid.nx; ++x) {
            error += std::pow(got.eta[x] - expected.eta[x], 2);
            norm += std::pow(expected.eta[x], 2);
            if (x < c.grid.nx / 5) left += got.eta[x] * double(got.eta[x]);
            if (x > 4 * c.grid.nx / 5) right += got.eta[x] * double(got.eta[x]);
        }
        std::cout << "REFLECTION t=" << completed * dt << " relative_L2=" << std::sqrt(error / norm)
                  << " left/right=" << left / right << '\n';
        CHECK(std::sqrt(error / norm) < .001);
        CHECK(left < .01 * right);
        check_walls(got, c);
    }
}
