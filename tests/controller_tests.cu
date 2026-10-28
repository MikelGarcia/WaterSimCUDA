#include "app_controller.hpp"
#include "validation.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <limits>

namespace {
water::SpawnView view() {
    return {{{17.5f, 21.8f, 23.2f}, {0, 0, 0}, {0, 1, 0}, 42,
             CAMERA_PERSPECTIVE}, 1.5f};
}

water::Config fixture() {
    water::Config config;
    config.grid = {96, 64, 20.0f, 12.0f};
    config.impact_radius = 0.75f;
    config.depth = 1;
    return config;
}

void advance(water::AppController& app, int quanta) {
    for (int i = 0; i < quanta; ++i)
        app.frame(1.0 / 120.0, false, false, 0, 0.01f, 0.15f, view());
}

bool zero(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float x) { return x == 0; });
}

void same_boat(const water::BoatState& a, const water::BoatState& b, float tolerance = 1e-6f) {
    CHECK_NEAR(a.x, b.x, tolerance);
    CHECK_NEAR(a.z, b.z, tolerance);
    CHECK_NEAR(a.yaw, b.yaw, tolerance);
    CHECK_NEAR(a.vx, b.vx, tolerance);
    CHECK_NEAR(a.vz, b.vz, tolerance);
    CHECK_NEAR(a.yaw_rate, b.yaw_rate, tolerance);
}

void same_water(const water::Snapshot& a, const water::Snapshot& b) {
    for (const auto pair : {std::make_pair(&a.eta, &b.eta),
                            std::make_pair(&a.u, &b.u), std::make_pair(&a.v, &b.v)})
        for (std::size_t i = 0; i < pair.first->size(); ++i)
            CHECK_NEAR((*pair.first)[i], (*pair.second)[i], 2e-6f);
}
}

TEST_CASE("controller", held_boat_controls_follow_ticks_across_render_partitions) {
    const auto config = fixture();
    water::Simulation reference(config);
    const water::BoatControl controls[]{{1, 0}, {1, -1}, {0, 0}, {-1, 0}, {-1, 1}};
    const int substeps = water::substeps_per_quantum(config);
    for (auto control : controls)
        for (int tick = 0; tick < 120; ++tick)
            for (int substep = 0; substep < substeps; ++substep)
                reference.step(float(water::FixedClock::quantum / substeps), .15f, control);
    const auto expected = water::download_boat(reference);
    CHECK(std::hypot(expected.x - config.grid.lx / 2, expected.z - config.grid.lz / 2) > .1f);
    for (int ticks_per_frame : {1, 2, 4}) {
        water::Simulation simulation(config);
        water::AppController app(simulation, config);
        for (auto control : controls)
            for (int tick = 0; tick < 120; tick += ticks_per_frame)
                app.frame(ticks_per_frame / 120.0, false, false, 0, .01f, .15f, view(), control);
        CHECK_NEAR(app.sim_time(), 5, 1e-14);
        same_boat(water::download_boat(simulation), expected);
        same_water(water::download(simulation), water::download(reference));
    }
}

TEST_CASE("controller", overload_boat_motion_is_exactly_eight_ticks) {
    const auto config = fixture();
    water::Simulation simulation(config), reference(config);
    water::AppController app(simulation, config), ordinary(reference, config);
    app.frame(.5, false, false, 0, .01f, .15f, view(), {1, 1});
    for (int i = 0; i < 8; ++i)
        ordinary.frame(1.0 / 120, false, false, 0, .01f, .15f, view(), {1, 1});
    CHECK(app.sim_time() == ordinary.sim_time());
    CHECK(water::download_boat(simulation).vx > 0);
    same_boat(water::download_boat(simulation), water::download_boat(reference), 0);
    same_water(water::download(simulation), water::download(reference));
}

TEST_CASE("controller", held_boat_pause_depth_resume_and_reset) {
    const auto config = fixture();
    water::Simulation simulation(config), reference(config);
    water::AppController app(simulation, config), ordinary(reference, config);
    for (int tick = 0; tick < 120; ++tick) {
        app.frame(1.0 / 120, false, false, 0, .01f, .15f, view(), {1, 1});
        ordinary.frame(1.0 / 120, false, false, 0, .01f, .15f, view(), {1, 1});
    }
    const auto boat = water::download_boat(simulation);
    const auto water = water::download(simulation);
    const auto time = app.sim_time();
    CHECK(boat.yaw > 0 && !zero(water.eta));
    app.set_depth(.25f);
    ordinary.set_depth(.25f);
    app.frame(10, true, true, 120, .005f, 10, view(), {-1, -1});
    same_boat(water::download_boat(simulation), boat, 0);
    const auto frozen = water::download(simulation);
    CHECK(water.eta == frozen.eta && water.u == frozen.u && water.v == frozen.v);
    CHECK(app.sim_time() == time);
    app.frame(1.0 / 120, false, false, 0, .005f, .15f, view(), {-1, -1});
    ordinary.frame(1.0 / 120, false, false, 0, .005f, .15f, view(), {-1, -1});
    CHECK_NEAR(app.sim_time(), time + 1.0 / 120, 1e-14);
    same_boat(water::download_boat(simulation), water::download_boat(reference), 0);
    same_water(water::download(simulation), water::download(reference));
    CHECK(app.spawn(10, 6, view()));
    app.reset();
    same_boat(water::download_boat(simulation), {10, 6, 0, 0, 0, 0}, 0);
    const auto flat = water::download(simulation);
    CHECK(zero(flat.eta) && zero(flat.u) && zero(flat.v));
    CHECK(app.sim_time() == 0 && app.drops().empty());
    CHECK(simulation.config().depth == .25f);
    CHECK(simulation.config().rain_rate == config.rain_rate);
    CHECK(simulation.config().damping == config.damping);
    CHECK(simulation.config().reflection == config.reflection);
}

TEST_CASE("controller", controlled_boat_is_consistent_across_depth_selected_cfl_substeps) {
    auto shallow = fixture(), deep = fixture();
    shallow.depth = .25f;
    shallow.impact_strength = .005f;
    deep.depth = 100;
    CHECK(water::substeps_per_quantum(shallow) != water::substeps_per_quantum(deep));
    water::Simulation a(shallow), b(deep);
    water::AppController first(a, shallow), second(b, deep);
    for (auto control : {water::BoatControl{1, .6f}, {0, -.4f}, {-1, -.5f}, {1, .3f}})
        for (int tick = 0; tick < 30; ++tick) {
            first.frame(1.0 / 120, false, false, 0, .005f, .15f, view(), control);
            second.frame(1.0 / 120, false, false, 0, .005f, .15f, view(), control);
        }
    const auto x = water::download_boat(a), y = water::download_boat(b);
    CHECK(std::hypot(x.x - y.x, x.z - y.z) < .003f);
    CHECK(std::hypot(x.vx - y.vx, x.vz - y.vz) < .003f);
    CHECK(std::abs(x.yaw - y.yaw) < .005f);
    CHECK(first.sim_time() == second.sim_time());
}

TEST_CASE("controller", stalled_frame_preserves_drop_and_limits_advancement) {
    const auto config = fixture();
    water::Simulation simulation(config);
    water::AppController app(simulation, config);
    CHECK(app.spawn(10, 6, view()));
    app.frame(0.5, false, false, 0, 0.01f, 0.15f, view());
    CHECK_NEAR(app.sim_time(), 8.0 / 120.0, 1e-14);
    CHECK(app.drops().size() == 1);
    CHECK(app.drops()[0].start == 0);
    CHECK_NEAR(app.drops()[0].arrival, std::sqrt(2 * app.drops()[0].spawn_height / 9.81), 1e-9);
    CHECK(zero(water::download(simulation).eta));
}

TEST_CASE("controller", advances_valid_depth_at_float_timestep_rounding_boundary) {
    water::Config config;
    config.depth = 1.5178088f;
    water::Simulation simulation(config);
    const water::Impact impact{10, 6, config.impact_radius, config.impact_strength};
    simulation.apply_impacts(&impact, 1);
    water::AppController app(simulation, config);
    app.frame(1.0 / 120.0, false, false, 0, config.impact_strength, config.damping, view());
    CHECK_NEAR(app.sim_time(), 1.0 / 120.0, 1e-14);
    const auto state = water::download(simulation);
    CHECK(!zero(state.u));
    CHECK(!zero(state.v));
}

TEST_CASE("controller", pause_freezes_time_drops_and_all_water_arrays) {
    const auto config = fixture();
    water::Simulation simulation(config);
    water::AppController app(simulation, config);
    CHECK(app.spawn(10, 6, view()));
    advance(app, static_cast<int>(std::ceil(app.drops()[0].arrival * 120)));
    CHECK(app.spawn(4, 4, view()));
    const auto before = water::download(simulation);
    CHECK(!zero(before.eta));
    const auto time = app.sim_time();
    const auto drop = app.drops()[0];
    app.frame(10, true, true, 60, 0.02f, 5, view());
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
    CHECK(app.spawn(10, 6, view()));
    const int arrival_tick = static_cast<int>(std::ceil(app.drops()[0].arrival * 120));
    advance(app, arrival_tick - 1);
    CHECK(zero(water::download(simulation).eta));
    advance(app, 2);
    CHECK(app.drops().empty());
    CHECK(!zero(water::download(simulation).eta));
    CHECK(app.spawn(4, 4, view()));
    app.reset();
    CHECK(app.sim_time() == 0);
    CHECK(app.drops().empty());
    auto state = water::download(simulation);
    CHECK(zero(state.eta) && zero(state.u) && zero(state.v));
    advance(app, 150);
    state = water::download(simulation);
    CHECK(zero(state.eta) && zero(state.u) && zero(state.v));
}

TEST_CASE("controller", long_flights_preserve_capacity_and_batching) {
    const auto config = fixture();
    water::Simulation simulation(config);
    water::Simulation reference(config);
    water::AppController app(simulation, config);
    auto high = view();
    high.camera.position = {0, 75, 1};
    for (int i = 0; i < 256; ++i) CHECK(app.spawn(10, 6, high));
    CHECK(!app.spawn(4, 4, high));
    const int arrival_tick = static_cast<int>(std::ceil(app.drops()[0].arrival * 120));
    CHECK(arrival_tick > 120);
    for (int i = 0; i < arrival_tick - 1; ++i)
        app.frame(1.0 / 120.0, false, true, 60, 0.01f, 0.15f, high);
    CHECK(app.drops().size() == 256);
    CHECK(zero(water::download(simulation).eta));
    advance(app, 2);
    const water::Impact impact{10, 6, 0.75f, 0.01f};
    for (int i = 0; i < 256; ++i) reference.apply_impacts(&impact, 1);
    auto actual = water::download(simulation);
    for (int i = 0; i < 2; ++i) reference.step(1.0f / 120.0f, 0.15f);
    auto expected = water::download(reference);
    CHECK(app.drops().empty());
    for (const auto pair : {std::make_pair(&actual.eta, &expected.eta),
                            std::make_pair(&actual.u, &expected.u),
                            std::make_pair(&actual.v, &expected.v)})
        for (std::size_t i = 0; i < pair.first->size(); ++i)
            CHECK_NEAR((*pair.first)[i], (*pair.second)[i], 2e-5f);
    advance(app, arrival_tick);
    CHECK(app.drops().empty());
    for (int i = 0; i < arrival_tick; ++i) reference.step(1.0f / 120.0f, 0.15f);
    actual = water::download(simulation);
    expected = water::download(reference);
    for (std::size_t i = 0; i < actual.eta.size(); ++i)
        CHECK_NEAR(actual.eta[i], expected.eta[i], 2e-5f);
}

TEST_CASE("controller", manual_and_seeded_rain_births_keep_their_camera_trajectory) {
    const auto config = fixture();
    water::Simulation simulation(config);
    water::AppController app(simulation, config);
    auto initial = view();
    initial.camera.position = {0, 75, 1};
    CHECK(app.spawn(10, 6, initial));
    const auto manual = app.drops()[0];
    CHECK(manual.spawn_height > 75);
    CHECK_NEAR(manual.arrival - manual.start, std::sqrt(2 * manual.spawn_height / 9.81), 1e-9);
    for (int tick = 0; tick < 60; ++tick)
        app.frame(1.0 / 120.0, false, true, 60, 0.01f, 0.15f, initial);
    CHECK(app.drops().size() > 3);
    const auto births = app.drops();
    for (const auto& drop : births) {
        CHECK(drop.spawn_height > 75);
        const double duration = std::sqrt(2 * drop.spawn_height / 9.81);
        CHECK_NEAR(drop.arrival - drop.start, duration, 1e-9 * duration);
        CHECK_NEAR(water::drop_y(drop, drop.start + duration / 2),
                   0.75 * drop.spawn_height, 1e-9 * drop.spawn_height);
    }
    const double held_time = app.sim_time();
    app.frame(10, true, true, 60, 0.01f, 0.15f, view());
    CHECK(app.sim_time() == held_time);
    CHECK(app.drops().size() == births.size());
    app.frame(1.0 / 120.0, false, true, 60, 0.01f, 0.15f, view());
    for (std::size_t i = 0; i < births.size(); ++i) {
        const auto& drop = app.drops()[i];
        CHECK(drop.x == births[i].x && drop.z == births[i].z);
        CHECK(drop.start == births[i].start && drop.arrival == births[i].arrival);
        CHECK(drop.spawn_height == births[i].spawn_height);
        CHECK(water::drop_y(drop, held_time) == water::drop_y(births[i], held_time));
    }
    for (int tick = 0; tick < 60; ++tick)
        app.frame(1.0 / 120.0, false, true, 60, 0.01f, 0.15f, view());
    CHECK(app.drops().size() > births.size());
    CHECK(app.drops().back().spawn_height < 75);
    app.reset();
    CHECK(app.sim_time() == 0 && app.drops().empty());
    CHECK(zero(water::download(simulation).eta));
    CHECK(app.spawn(10, 6, initial));
    for (int tick = 0; tick < 60; ++tick)
        app.frame(1.0 / 120.0, false, true, 60, 0.01f, 0.15f, initial);
    CHECK(app.drops().size() == births.size());
    for (std::size_t i = 0; i < births.size(); ++i) {
        CHECK(app.drops()[i].x == births[i].x && app.drops()[i].z == births[i].z);
        CHECK(app.drops()[i].start == births[i].start);
        CHECK(app.drops()[i].arrival == births[i].arrival);
        CHECK(app.drops()[i].spawn_height == births[i].spawn_height);
    }
}

TEST_CASE("controller", paused_depth_change_preserves_state_and_flights_then_advances_new_cfl) {
    water::Config config;
    water::Simulation simulation(config);
    water::AppController app(simulation, config);
    const water::Impact impact{10, 6, config.impact_radius, .01f};
    simulation.apply_impacts(&impact, 1);
    app.frame(1.0 / 120, false, false, 0, .01f, .15f, view());
    CHECK(app.spawn(4, 5, view()));
    const auto before = water::download(simulation);
    const auto flight = app.drops()[0];
    const auto time = app.sim_time();
    app.set_depth(.25f);
    app.frame(10, true, false, 0, .005f, .15f, view());
    const auto after = water::download(simulation);
    CHECK(before.eta == after.eta && before.u == after.u && before.v == after.v);
    CHECK(app.sim_time() == time);
    const auto drop = app.drops()[0];
    CHECK(drop.x == flight.x && drop.z == flight.z && drop.start == flight.start);
    CHECK(drop.spawn_height == flight.spawn_height && drop.arrival == flight.arrival);
    CHECK(simulation.config().depth == .25f && simulation.config().impact_strength <= .005f);
    for (float invalid : {0.f, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::max()}) {
        CHECK_THROWS(app.set_depth(invalid), std::invalid_argument);
        CHECK(simulation.config().depth == .25f && app.sim_time() == time);
    }
    CHECK_THROWS(app.frame(1.0 / 120, false, false, 0, .006f, .15f, view()), std::invalid_argument);
    for (float depth : {1.5178088f, 100.f}) {
        app.set_depth(depth);
        app.frame(1.0 / 120, false, false, 0, .005f, .15f, view());
    }
    CHECK_NEAR(app.sim_time(), time + 2.0 / 120, 1e-14);
    app.reset();
    CHECK(simulation.config().depth == 100 && app.sim_time() == 0);
}

TEST_CASE("controller", exterior_automatic_arrival_leaves_entire_water_state_zero) {
    const auto config = fixture();
    water::Simulation simulation(config);
    water::AppController app(simulation, config);
    while (app.drops().empty()) app.frame(1.0/120, false, true, 60, .01f, .15f, view());
    CHECK(app.drops().size() == 1);
    const auto drop = app.drops()[0];
    CHECK(drop.x < 0 || drop.x > 20 || drop.z < 0 || drop.z > 12);
    advance(app, int(std::ceil((drop.arrival-app.sim_time())*120))+1);
    CHECK(app.drops().empty());
    const auto state = water::download(simulation);
    CHECK(zero(state.eta) && zero(state.u) && zero(state.v));
}
