#include "app_controller.hpp"
#include "validation.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace {
water::SpawnView view() {
    return {{{17.5f, 21.8f, 23.2f}, {0, 0, 0}, {0, 1, 0}, 42,
             CAMERA_PERSPECTIVE}, 1.5f};
}

void sustained_rain(int nx, int ny, float depth = 3, float rate = 120, float damping = .15f) {
    water::Config config;
    config.grid = {nx, ny, 20.0f, 12.0f};
    config.depth = depth;
    config.impact_radius = 0.75f;
    config.impact_strength = 0.02f * config.depth;
    config.rain_rate = rate;
    config.damping = damping;
    config.seed = 1;
    water::Simulation simulation(config);
    water::AppController app(simulation, config);
    double max_mean = 0;
    float peak = 0, preceding_envelope = 0, final_envelope = 0;
    float distance_from_spawn = 0;
    bool reversed = false;

    for (int second = 1; second <= 120; ++second) {
        const int phase = (second - 1) % 32;
        water::BoatControl control;
        if (phase < 12) control = {1, 0};
        else if (phase < 20) control = {1, .55f};
        else if (phase < 28) control = {-1, -.65f};
        else control = {1, -1};
        for (int tick = 0; tick < 120; ++tick)
            app.frame(water::FixedClock::quantum, false, true, config.rain_rate,
                      config.impact_strength, config.damping, view(), control);
        const auto boat = water::download_boat(simulation);
        for (float value : {boat.x, boat.z, boat.yaw, boat.vx, boat.vz, boat.yaw_rate})
            CHECK(std::isfinite(value));
        const float extent_x = .5f * (water::boat_geometry.length * std::abs(std::cos(boat.yaw)) +
                                      water::boat_geometry.beam * std::abs(std::sin(boat.yaw)));
        const float extent_z = .5f * (water::boat_geometry.length * std::abs(std::sin(boat.yaw)) +
                                      water::boat_geometry.beam * std::abs(std::cos(boat.yaw)));
        CHECK(boat.x >= extent_x && boat.x <= config.grid.lx - extent_x);
        CHECK(boat.z >= extent_z && boat.z <= config.grid.lz - extent_z);
        CHECK(std::hypot(boat.vx, boat.vz) <= 1.87501f);
        distance_from_spawn = std::max(distance_from_spawn, std::hypot(boat.x - 10, boat.z - 6));
        reversed |= boat.vx * std::cos(boat.yaw) + boat.vz * std::sin(boat.yaw) < -.1f;
        const auto state = water::download(simulation);
        for (const auto* values : {&state.eta, &state.u, &state.v})
            for (float value : *values) CHECK(std::isfinite(value));

        const double mean = std::accumulate(state.eta.begin(), state.eta.end(), 0.0) /
                            state.eta.size();
        float amplitude = 0;
        for (float eta : state.eta) amplitude = std::max(amplitude, std::abs(eta));
        max_mean = std::max(max_mean, std::abs(mean));
        peak = std::max(peak, amplitude);
        CHECK(std::abs(mean) <= 1e-6 * std::max(depth, .25f));
        CHECK(amplitude < 0.1f * config.depth);
        if (second > 60 && second <= 90)
            preceding_envelope = std::max(preceding_envelope, amplitude);
        if (second > 90) final_envelope = std::max(final_envelope, amplitude);
    }

    CHECK_NEAR(app.sim_time(), 120.0, 1e-12);
    CHECK(distance_from_spawn > 1);
    CHECK(reversed);
    CHECK(peak > 0.001f * config.depth);
    CHECK(final_envelope <= 2 * preceding_envelope);
    std::cout << "SOAK " << nx << 'x' << ny << " seconds=120 rain=" << rate << "/s H=" << depth << " damping=" << damping << " seed=1"
              << " max_abs_mean=" << max_mean << " peak=" << peak
              << " envelope_61_90=" << preceding_envelope
              << " envelope_91_120=" << final_envelope << '\n';
}
}

TEST_CASE("soak", sustained_max_rain_preserves_rectangular_water) {
    sustained_rain(96, 64);
}

TEST_CASE("soak", sustained_max_rain_preserves_transposed_water) {
    sustained_rain(64, 96);
}

TEST_CASE("soak", moving_boat_with_shallow_and_deep_rain) {
    for (float depth : {.25f, 3.f, 6.f})
        for (float rate : {30.f, 120.f})
            for (float damping : {0.f, 2.5f})
                sustained_rain(96, 64, depth, rate, damping);
}
