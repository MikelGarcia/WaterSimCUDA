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
            total += weight * config.depth * velocity * velocity;
        }
    }
    for (int y = 0; y <= g.ny; ++y) {
        for (int x = 0; x < g.nx; ++x) {
            const double velocity = state.v[y * g.nx + x];
            CHECK(std::isfinite(velocity));
            const double weight = (y == 0 || y == g.ny) ? 0.5 : 1.0;
            total += weight * config.depth * velocity * velocity;
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
    const double expected = 2.0 * (along_x ? config.grid.lx : config.grid.lz) /
                            std::sqrt(config.gravity * config.depth);
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

TEST_CASE("numerics", positive_pulse_reflects_with_positive_height_and_reversed_flow) {
    const auto config = fixture();
    const auto& g = config.grid;
    auto state = zero_state(config);
    const double speed = std::sqrt(config.gravity * config.depth);
    const auto pulse = [](double x) {
        return 0.001 * std::exp(-std::pow((x - 15.0) / 0.6, 2));
    };
    for (int y = 0; y < g.ny; ++y) {
        for (int x = 0; x < g.nx; ++x) {
            state.eta[y * g.nx + x] = static_cast<float>(pulse((x + 0.5) * g.lx / g.nx));
        }
        for (int x = 1; x < g.nx; ++x) {
            state.u[y * (g.nx + 1) + x] =
                static_cast<float>(speed / config.depth * pulse(x * g.lx / g.nx));
        }
    }
    water::Simulation simulation(config);
    water::upload(simulation, state);
    const float dt = time_step(config);
    const int steps = static_cast<int>(std::round(10.0 / speed / dt));
    for (int i = 0; i < steps; ++i) simulation.step(dt, 0.0f);
    state = water::download(simulation);
    double projection = 0.0, normalization = 0.0, flow = 0.0;
    for (int x = 0; x < g.nx; ++x) {
        const double shape = pulse((x + 0.5) * g.lx / g.nx);
        projection += state.eta[x] * shape;
        normalization += shape * shape;
        flow += state.u[x] * pulse(x * g.lx / g.nx);
    }
    std::cout << "reflected pulse amplitude ratio: " << projection / normalization << '\n';
    CHECK(projection / normalization > 0.8);
    CHECK(flow < 0.0);
    check_walls(state, config);
}

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
    const double period = 40.0 / std::sqrt(config.gravity * config.depth);
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
                        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::max()}) {
        CHECK_THROWS(simulation.set_depth(depth), std::invalid_argument);
        CHECK(simulation.config().depth == .25f);
        after = water::download(simulation);
        CHECK(after.eta == initial.eta && after.u == initial.u && after.v == initial.v);
    }
    simulation.set_depth(100);
    CHECK_THROWS(simulation.step(1.f / 120, 0), std::invalid_argument);
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

TEST_CASE("numerics", nonzero_fields_follow_independent_linear_swe) {
    water::Config c;
    c.grid = {80, 64, 4, 3};
    c.impact_radius = .15f;
    water::Simulation simulation(c);
    auto expected = water::download(simulation);
    const auto& g = c.grid;
    const double dx = double(g.lx) / g.nx, dz = double(g.lz) / g.ny;
    for (int z = 0; z < g.ny; ++z)
        for (int x = 0; x < g.nx; ++x)
            expected.eta[z * g.nx + x] = float(.001 * std::cos(pi * (x + .5) / g.nx) *
                                                    std::cos(pi * (z + .5) / g.ny));
    for (int z = 0; z < g.ny; ++z)
        for (int x = 1; x < g.nx; ++x)
            expected.u[z * (g.nx + 1) + x] = float(.0003 * std::sin(pi * x / g.nx));
    for (int z = 1; z < g.ny; ++z)
        for (int x = 0; x < g.nx; ++x)
            expected.v[z * g.nx + x] = float(-.0002 * std::sin(pi * z / g.ny));
    water::upload(simulation, expected);
    constexpr float dt = .001f;
    double max_error = 0;
    for (int step = 0; step < 200; ++step) {
        auto next = expected;
        for (int z = 0; z < g.ny; ++z) {
            for (int x = 1; x < g.nx; ++x) {
                const int i = z * (g.nx + 1) + x;
                const double gradient = (expected.eta[z * g.nx + x] -
                                         double(expected.eta[z * g.nx + x - 1])) / dx;
                next.u[i] = float((expected.u[i] - c.gravity * double(dt) * gradient) /
                                  (1 + c.damping * double(dt)));
            }
        }
        for (int z = 1; z < g.ny; ++z) {
            for (int x = 0; x < g.nx; ++x) {
                const int i = z * g.nx + x;
                const double gradient = (expected.eta[i] - double(expected.eta[i - g.nx])) / dz;
                next.v[i] = float((expected.v[i] - c.gravity * double(dt) * gradient) /
                                  (1 + c.damping * double(dt)));
            }
        }
        for (int z = 0; z < g.ny; ++z) {
            for (int x = 0; x < g.nx; ++x) {
                const double divergence =
                    (next.u[z * (g.nx + 1) + x + 1] - double(next.u[z * (g.nx + 1) + x])) / dx +
                    (next.v[(z + 1) * g.nx + x] - double(next.v[z * g.nx + x])) / dz;
                next.eta[z * g.nx + x] = float(expected.eta[z * g.nx + x] - c.depth * double(dt) * divergence);
            }
        }
        expected = next;
        simulation.step(dt, c.damping);
        const auto actual = water::download(simulation);
        for (size_t i = 0; i < actual.eta.size(); ++i)
            max_error = std::max(max_error, std::abs(double(actual.eta[i]) - expected.eta[i]));
        for (size_t i = 0; i < actual.u.size(); ++i)
            max_error = std::max(max_error, std::abs(double(actual.u[i]) - expected.u[i]));
        for (size_t i = 0; i < actual.v.size(); ++i)
            max_error = std::max(max_error, std::abs(double(actual.v[i]) - expected.v[i]));
    }
    std::cout << "CPU linear SWE maximum absolute error=" << max_error << '\n';
    CHECK(max_error < 1e-8);
}

TEST_CASE("numerics", released_one_dimensional_wave_has_analytic_speed) {
    water::Config c;
    c.grid = {512, 32, 20, 12};
    c.impact_radius = .8f;
    water::Simulation simulation(c);
    auto initial = water::download(simulation);
    const double speed = std::sqrt(c.gravity * c.depth);
    const auto pulse = [](double x) { return .001 * std::exp(-std::pow((x - 7) / .6, 2)); };
    for (int z = 0; z < c.grid.ny; ++z) {
        for (int x = 0; x < c.grid.nx; ++x)
            initial.eta[z * c.grid.nx + x] = float(pulse((x + .5) * c.grid.lx / c.grid.nx));
        for (int x = 1; x < c.grid.nx; ++x)
            initial.u[z * (c.grid.nx + 1) + x] = float(speed / c.depth * pulse(x * c.grid.lx / c.grid.nx));
    }
    water::upload(simulation, initial);
    const float dt = float(water::stable_dt(c)) * .8f;
    float elapsed = 0;
    for (float target : {.2f, .4f, .6f}) {
        while (elapsed < target) {
            simulation.step(dt, 0);
            elapsed += dt;
        }
        const auto actual = water::download(simulation);
        double squared_error = 0, norm = 0;
        int maximum = 0;
        for (int x = 0; x < c.grid.nx; ++x) {
            const double expected = pulse((x + .5) * c.grid.lx / c.grid.nx - speed * elapsed);
            squared_error += std::pow(actual.eta[x] - expected, 2);
            norm += expected * expected;
            if (actual.eta[x] > actual.eta[maximum]) maximum = x;
        }
        const double peak_position = (maximum + .5) * c.grid.lx / c.grid.nx;
        const double error = std::sqrt(squared_error / norm);
        std::cout << "ANALYTIC time=" << elapsed << " peak position=" << peak_position
                  << " expected=" << 7 + speed * elapsed << " relative L2=" << error << '\n';
        CHECK(std::abs(peak_position - (7 + speed * elapsed)) < 2 * c.grid.lx / c.grid.nx);
        CHECK(error < .03);
    }
}
