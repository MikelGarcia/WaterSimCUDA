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

    for (int second = 1; second <= 120; ++second) {
        const float yaw = second * .2f;
        water::upload_boat(simulation, {10, 6, yaw, .75f * std::cos(yaw), .75f * std::sin(yaw), 0});
        for (int tick = 0; tick < 120; ++tick)
            app.frame(water::FixedClock::quantum, false, true, config.rain_rate,
                      config.impact_strength, config.damping, view());
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
