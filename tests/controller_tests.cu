#include "app_controller.hpp"
#include "validation.hpp"
#include "test_support.hpp"

#include <algorithm>

namespace {
water::Config fixture() {
    water::Config config;
    config.grid = {96, 64, 20.0f, 12.0f};
    config.impact_radius = 0.75f;
    return config;
}

void advance(water::AppController& app, int quanta) {
    for (int i = 0; i < quanta; ++i)
        app.frame(1.0 / 120.0, false, false, 0, 0.01f, 0.15f);
}

bool zero(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float x) { return x == 0; });
}
}

TEST_CASE("controller", stalled_frame_preserves_drop_and_limits_advancement) {
    const auto config = fixture();
    water::Simulation simulation(config);
    water::AppController app(simulation, config);
    CHECK(app.spawn(10, 6));
    app.frame(0.5, false, false, 0, 0.01f, 0.15f);
    CHECK_NEAR(app.sim_time(), 8.0 / 120.0, 1e-14);
    CHECK(app.drops().size() == 1);
    CHECK(app.drops()[0].start == 0);
    CHECK(app.drops()[0].arrival == 0.6);
    CHECK(zero(water::download(simulation).eta));
}

TEST_CASE("controller", pause_freezes_time_drops_and_all_water_arrays) {
    const auto config = fixture();
    water::Simulation simulation(config);
    water::AppController app(simulation, config);
    CHECK(app.spawn(10, 6));
    advance(app, 73);
    CHECK(app.spawn(4, 4));
    const auto before = water::download(simulation);
    CHECK(!zero(before.eta));
    const auto time = app.sim_time();
    const auto drop = app.drops()[0];
    app.frame(10, true, true, 60, 0.02f, 5);
    const auto after = water::download(simulation);
    CHECK(app.sim_time() == time);
    CHECK(app.drops().size() == 1);
    CHECK(app.drops()[0].start == drop.start);
    CHECK(app.drops()[0].arrival == drop.arrival);
    CHECK(before.eta == after.eta);
    CHECK(before.u == after.u);
    CHECK(before.v == after.v);
}

TEST_CASE("controller", arrival_applies_once_then_reset_clears_flight_and_water) {
    const auto config = fixture();
    water::Simulation simulation(config);
    water::AppController app(simulation, config);
    CHECK(app.spawn(10, 6));
    advance(app, 71);
    CHECK(zero(water::download(simulation).eta));
    advance(app, 2);
    CHECK(app.drops().empty());
    CHECK(!zero(water::download(simulation).eta));
    CHECK(app.spawn(4, 4));
    app.reset();
    CHECK(app.sim_time() == 0);
    CHECK(app.drops().empty());
    auto state = water::download(simulation);
    CHECK(zero(state.eta) && zero(state.u) && zero(state.v));
    advance(app, 150);
    state = water::download(simulation);
    CHECK(zero(state.eta) && zero(state.u) && zero(state.v));
}

TEST_CASE("controller", simultaneous_arrivals_above_batch_limit_are_all_applied) {
    const auto config = fixture();
    water::Simulation simulation(config);
    water::Simulation reference(config);
    water::AppController app(simulation, config);
    for (int i = 0; i < 256; ++i) CHECK(app.spawn(10, 6));
    CHECK(!app.spawn(10, 6));
    advance(app, 73);
    const water::Impact impact{10, 6, 0.75f, 0.01f};
    for (int i = 0; i < 256; ++i) reference.apply_impacts(&impact, 1);
    // The impact precedes the substep at quantum 72; quantum 73 adds one more step.
    auto actual = water::download(simulation);
    for (int i = 0; i < 2; ++i) reference.step(1.0f / 120.0f, 0.15f);
    auto expected = water::download(reference);
    CHECK(app.drops().empty());
    for (std::size_t i = 0; i < actual.eta.size(); ++i)
        CHECK_NEAR(actual.eta[i], expected.eta[i], 2e-5f);
}
