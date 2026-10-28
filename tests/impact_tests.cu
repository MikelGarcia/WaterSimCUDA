#include "simulation.hpp"
#include "validation.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

namespace {

water::Config fixture() {
    water::Config config;
    config.grid = {96, 64, 20.0f, 12.0f};
    config.impact_radius = 0.75f;
    config.depth = 1;
    return config;
}

void check_disturbance(const water::Config& config, const water::Impact& impact) {
    water::Simulation simulation(config);
    simulation.apply_impacts(&impact, 1);
    const auto state = water::download(simulation);
    const auto& g = config.grid;
    const double dx = static_cast<double>(g.lx) / g.nx;
    const double dz = static_cast<double>(g.lz) / g.ny;
    const double volume = std::accumulate(state.eta.begin(), state.eta.end(), 0.0) * dx * dz;
    CHECK(std::abs(volume) <= 1e-6);
    CHECK(*std::max_element(state.eta.begin(), state.eta.end()) > 1e-4f);
    CHECK(*std::min_element(state.eta.begin(), state.eta.end()) < -1e-4f);
    for (int y = 0; y < g.ny; ++y) {
        for (int x = 0; x < g.nx; ++x) {
            const float height = state.eta[y * g.nx + x];
            CHECK(std::isfinite(height));
            CHECK(std::abs(height) <= 0.02f * config.depth);
            const double px = (x + 0.5) * dx - impact.x;
            const double pz = (y + 0.5) * dz - impact.z;
            if (px * px + pz * pz > impact.radius * impact.radius) CHECK(height == 0.0f);
        }
    }
    for (float value : state.u) CHECK(value == 0.0f);
    for (float value : state.v) CHECK(value == 0.0f);
}

} // namespace

TEST_CASE("impacts", center_walls_corners_and_unaligned_positions_conserve_volume) {
    for (bool transpose : {false, true}) {
        auto config = fixture();
        if (transpose) std::swap(config.grid.nx, config.grid.ny);
        for (const auto& position : std::array<std::array<float, 2>, 12>{{
                 {10.0f, 6.0f}, {0.0f, 6.0f}, {20.0f, 6.0f}, {10.0f, 0.0f}, {10.0f, 12.0f},
                 {0.0f, 0.0f}, {20.0f, 0.0f}, {0.0f, 12.0f}, {20.0f, 12.0f},
                 {9.83f, 5.62f}, {0.03f, 0.07f}, {19.97f, 11.93f}}}) {
            check_disturbance(config, {position[0], position[1], 0.75f, 0.02f});
        }
    }
}

TEST_CASE("impacts", empty_and_zero_strength_batches_preserve_existing_state) {
    const auto config = fixture();
    water::Simulation simulation(config);
    const water::Impact impact{10.0f, 6.0f, 0.75f, 0.01f};
    simulation.apply_impacts(&impact, 1);
    const auto before = water::download(simulation);
    simulation.apply_impacts(nullptr, 0);
    const water::Impact zero{0.0f, 0.0f, 0.75f, 0.0f};
    simulation.apply_impacts(&zero, 1);
    const auto after = water::download(simulation);
    CHECK(after.eta == before.eta);
    CHECK(after.u == before.u);
    CHECK(after.v == before.v);
}

TEST_CASE("impacts", invalid_batch_is_rejected_before_any_valid_prefix_is_applied) {
    const auto config = fixture();
    water::Simulation simulation(config);
    const water::Impact valid{10.0f, 6.0f, 0.75f, 0.01f};
    simulation.apply_impacts(&valid, 1);
    const auto before = water::download(simulation);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (const auto& invalid : std::array<water::Impact, 17>{{
             {-0.1f, 6, .75f, .01f}, {20.1f, 6, .75f, .01f},
             {10, -0.1f, .75f, .01f}, {10, 12.1f, .75f, .01f},
             {nan, 6, .75f, .01f}, {10, nan, .75f, .01f},
             {inf, 6, .75f, .01f}, {10, inf, .75f, .01f},
             {10, 6, nan, .01f}, {10, 6, inf, .01f},
             {10, 6, 0, .01f}, {10, 6, .4f, .01f}, {10, 6, 3.1f, .01f},
             {10, 6, .75f, nan}, {10, 6, .75f, inf},
             {10, 6, .75f, -.01f}, {10, 6, .75f, .021f}}}) {
        const water::Impact batch[]{valid, invalid};
        CHECK_THROWS(simulation.apply_impacts(batch, 2), std::invalid_argument);
    }
    CHECK_THROWS(simulation.apply_impacts(nullptr, 1), std::invalid_argument);
    CHECK_THROWS(simulation.apply_impacts(&valid, 65), std::invalid_argument);
    const auto after = water::download(simulation);
    CHECK(after.eta == before.eta);
    CHECK(after.u == before.u);
    CHECK(after.v == before.v);
}

TEST_CASE("impacts", overlapping_batch_matches_serial_events_without_clamping) {
    const auto config = fixture();
    water::Simulation batched(config), serial(config);
    std::array<water::Impact, 64> events;
    events.fill({10.0f, 6.0f, 0.75f, 0.02f});
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (i % 2) events[i].x += 0.05f;
        serial.apply_impacts(&events[i], 1);
    }
    batched.apply_impacts(events.data(), events.size());
    const auto batch_state = water::download(batched);
    const auto serial_state = water::download(serial);
    for (std::size_t i = 0; i < batch_state.eta.size(); ++i) {
        CHECK_NEAR(batch_state.eta[i], serial_state.eta[i], 2e-6f);
    }
    CHECK(*std::max_element(batch_state.eta.begin(), batch_state.eta.end()) > 0.1f);
    const double volume = std::accumulate(batch_state.eta.begin(), batch_state.eta.end(), 0.0) *
                          (20.0 / 96) * (12.0 / 64);
    CHECK(std::abs(volume) <= 1e-6);
}

TEST_CASE("impacts", repeated_calls_own_event_lifetime_and_reset_clears_pending_impacts) {
    const auto config = fixture();
    water::Simulation simulation(config);
    for (int i = 0; i < 80; ++i) {
        water::Impact event{0.1f + i * .2f, 6.0f, .75f, .01f};
        simulation.apply_impacts(&event, 1);
        event = {0, 0, 0, 0};
    }
    const auto disturbed = water::download(simulation);
    CHECK(*std::max_element(disturbed.eta.begin(), disturbed.eta.end()) > 0.0f);
    simulation.reset();
    const auto state = water::download(simulation);
    for (const auto* values : {&state.eta, &state.u, &state.v}) {
        for (float value : *values) CHECK(value == 0.0f);
    }
}

TEST_CASE("impacts", thousand_seeded_edge_and_interior_impacts_keep_mean_height) {
    const auto config = fixture();
    water::Simulation simulation(config);
    std::mt19937 random(7);
    std::uniform_real_distribution<float> x(0.0f, 20.0f), z(0.0f, 12.0f);
    const float dt = static_cast<float>((1.0 / 120.0) / water::substeps_per_quantum(config));
    for (int i = 0; i < 1000; ++i) {
        water::Impact event{x(random), z(random), .75f, .01f};
        if (i % 5 == 0) event.x = 0.0f;
        if (i % 5 == 1) event.x = 20.0f;
        if (i % 5 == 2) event.z = 0.0f;
        if (i % 5 == 3) event.z = 12.0f;
        simulation.apply_impacts(&event, 1);
        simulation.step(dt, config.damping);
    }
    const auto state = water::download(simulation);
    const double mean = std::accumulate(state.eta.begin(), state.eta.end(), 0.0) / state.eta.size();
    std::cout << "1000-impact mean height: " << mean << '\n';
    CHECK(std::abs(mean) < 1e-5);
    CHECK(*std::max_element(state.eta.begin(), state.eta.end()) > 1e-4f);
    for (const auto* values : {&state.eta, &state.u, &state.v}) {
        for (float value : *values) CHECK(std::isfinite(value));
    }
}
