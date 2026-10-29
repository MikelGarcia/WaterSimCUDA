#include "timeline.hpp"
#include "test_support.hpp"

#include <limits>
#include <algorithm>

namespace {
struct Vec { double x, y, z; };
Vec cross(Vec a, Vec b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec unit(Vec a) {
    const double length = std::sqrt(dot(a, a));
    return {a.x / length, a.y / length, a.z / length};
}
water::SpawnView view() {
    return {{{17.5f, 21.8f, 23.2f}, {0, 0, 0}, {0, 1, 0}, 42,
             CAMERA_PERSPECTIVE}, 1.5f};
}
}

TEST_CASE("timeline", full_drop_bounds_are_outside_spawn_frustum) {
    for (const water::Grid grid : {water::Grid{96, 64, 20, 12}, water::Grid{64, 96, 12, 20}})
    for (float pitch : {0.15f, 0.72f, 1.45f})
    for (float yaw : {0.0f, 0.65f, 1.8f, 3.4f, 4.9f})
    for (float scale : {0.4f, 1.65f, 4.0f})
    for (float aspect : {840.0f / 560.0f, 2.0f}) {
        const float distance = scale * std::max(grid.lx, grid.lz);
        auto pose = view();
        pose.aspect = aspect;
        pose.camera.position = {distance * std::cos(pitch) * std::sin(yaw),
                                distance * std::sin(pitch),
                                distance * std::cos(pitch) * std::cos(yaw)};
        const auto eye = pose.camera.position;
        const Vec forward = unit({-eye.x, -eye.y, -eye.z});
        const Vec right = unit(cross(forward, {0, 1, 0}));
        const Vec up = cross(right, forward);
        const double half_fov = pose.camera.fovy * 3.14159265358979323846 / 360;
        for (const Vector2 point : {Vector2{0, 0}, Vector2{grid.lx, 0},
                                   Vector2{0, grid.lz}, Vector2{grid.lx, grid.lz},
                                   Vector2{grid.lx / 2, grid.lz / 2}}) {
            const double height = water::drop_spawn_height(point.x, point.y, grid, pose);
            CHECK(std::isfinite(height));
            CHECK(height > eye.y);
            bool all_above = true, all_behind = true;
            for (double dx : {-0.065, 0.065})
            for (double dy : {0.0, 0.28})
            for (double dz : {-0.065, 0.065}) {
                const Vec relative{point.x - grid.lx / 2.0 + dx - eye.x,
                                   height + dy - eye.y, point.y - grid.lz / 2.0 + dz - eye.z};
                const double depth = dot(relative, forward);
                const double vertical = dot(relative, up);
                all_above &= vertical * std::cos(half_fov) - depth * std::sin(half_fov) >= 0.12;
                all_behind &= -depth >= 0.12;
            }
            CHECK(all_above || all_behind);
        }
    }
}

TEST_CASE("timeline", ballistic_endpoints_and_camera_change) {
    water::DropScheduler drops(water::Config{});
    CHECK(drops.spawn(4, 7, 1.25, view()));
    const auto born = drops.active()[0];
    const double duration = std::sqrt(2 * born.spawn_height / 9.81);
    CHECK_NEAR(born.arrival - born.start, duration, 1e-9 * duration);
    CHECK(water::drop_y(born, born.start) == born.spawn_height);
    CHECK(water::drop_y(born, born.start - 1) == born.spawn_height);
    CHECK_NEAR(water::drop_y(born, born.start + duration / 2), 0.75 * born.spawn_height,
               1e-9 * born.spawn_height);
    CHECK(water::drop_y(born, born.arrival) == 0);
    CHECK(water::drop_y(born, born.arrival + 10) == 0);
    auto moved = view();
    moved.camera.position = {0, 75, 1};
    const double position = water::drop_y(born, 1.5);
    CHECK(drops.tick(1.5, false, 0, 0.01f, moved).empty());
    const auto unchanged = drops.active()[0];
    CHECK(unchanged.spawn_height == born.spawn_height);
    CHECK(unchanged.start == born.start && unchanged.arrival == born.arrival);
    CHECK(water::drop_y(unchanged, 1.5) == position);
}

TEST_CASE("timeline", invalid_spawn_views_are_rejected) {
    const auto grid = water::Grid{};
    for (int fault = 0; fault < 11; ++fault) {
        auto invalid = view();
        switch (fault) {
        case 0: invalid.aspect = 0; break;
        case 1: invalid.aspect = std::numeric_limits<float>::infinity(); break;
        case 2: invalid.camera.fovy = 0; break;
        case 3: invalid.camera.fovy = 180; break;
        case 4: invalid.camera.projection = CAMERA_ORTHOGRAPHIC; break;
        case 5: invalid.camera.target = invalid.camera.position; break;
        case 6: invalid.camera.up = {0, 0, 0}; break;
        case 7: invalid.camera.up = {-17.5f, -21.8f, -23.2f}; break;
        case 8: invalid.camera.position.y = std::numeric_limits<float>::quiet_NaN(); break;
        case 9: invalid.camera.target = {17.5f, 50, 23.2f}; break;
        case 10: invalid.camera.fovy = std::numeric_limits<float>::quiet_NaN(); break;
        }
        CHECK_THROWS(water::drop_spawn_height(10, 6, grid, invalid), std::invalid_argument);
    }
}

TEST_CASE("timeline", clock_discards_stall_backlog_but_keeps_fraction) {
    water::FixedClock clock;
    CHECK(clock.advance(0.5, false) == 8);
    CHECK(clock.advance(0.0, false) == 0);
    clock.reset();
    CHECK(clock.advance(0.5 + 1.0 / 480.0, false) == 8);
    CHECK(clock.advance(0.0, false) == 0);
    CHECK(clock.advance(1.0 / 240.0, false) == 0);
    CHECK(clock.advance(1.0 / 480.0, false) == 1);
    CHECK(clock.advance(0.0, false) == 0);
}

TEST_CASE("timeline", pause_and_reset_do_not_accumulate_elapsed_time) {
    water::FixedClock clock;
    CHECK(clock.advance(1.0 / 240.0, false) == 0);
    CHECK(clock.advance(30.0, true) == 0);
    CHECK(clock.advance(1.0 / 240.0, false) == 1);
    CHECK(clock.advance(1.0 / 240.0, false) == 0);
    clock.reset();
    CHECK(clock.advance(1.0 / 240.0, false) == 0);
    CHECK(clock.advance(1.0 / 240.0, false) == 1);
}

TEST_CASE("timeline", clock_discards_integral_quanta_without_rounding_backlog) {
    water::FixedClock clock;
    CHECK(clock.advance(0.3, false) == 8);
    CHECK(clock.advance(1e-12, false) == 0);
    CHECK(clock.advance(1.0 / 120.0 - 1e-12, false) == 1);
}

TEST_CASE("timeline", clock_rejects_invalid_elapsed_time) {
    water::FixedClock clock;
    for (double value : {-1.0, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()}) {
        CHECK_THROWS(clock.advance(value, false), std::invalid_argument);
    }
}

TEST_CASE("timeline", arrival_once_at_stored_deadline) {
    water::Config config;
    water::DropScheduler drops(config);
    CHECK(drops.spawn(4.0f, 7.0f, 1.25, view()));
    CHECK(drops.active().size() == 1);
    CHECK_NEAR(drops.active()[0].start, 1.25, 1e-12);
    const double deadline = drops.active()[0].arrival;
    CHECK_NEAR(deadline - 1.25, std::sqrt(2 * drops.active()[0].spawn_height / 9.81), 1e-9);
    CHECK(drops.tick(deadline-1e-6, false, 8.0f, 0.015f, view()).empty());
    const auto events = drops.tick(deadline, false, 8.0f, 0.015f, view());
    CHECK(events.size() == 1);
    CHECK_NEAR(events[0].x, 4.0f, 1e-6f);
    CHECK_NEAR(events[0].z, 7.0f, 1e-6f);
    CHECK_NEAR(events[0].radius, config.impact_radius, 1e-6f);
    CHECK_NEAR(events[0].strength, 0.015f, 1e-6f);
    CHECK(drops.active().empty());
    CHECK(drops.tick(deadline + 1, false, 8.0f, 0.015f, view()).empty());
}

TEST_CASE("timeline", capacity_rejection_creates_no_later_impact) {
    water::DropScheduler drops(water::Config{});
    for (int i = 0; i < 256; ++i) CHECK(drops.spawn(5.0f, 6.0f, 0.0, view()));
    CHECK(!drops.spawn(7.0f, 8.0f, 0.0, view()));
    const double deadline = drops.active()[0].arrival;
    CHECK(drops.tick(deadline, false, 8.0f, 0.01f, view()).size() == 256);
    CHECK(drops.tick(deadline + 1, false, 8.0f, 0.01f, view()).empty());
    CHECK(drops.spawn(7.0f, 8.0f, deadline + 1, view()));
}

TEST_CASE("timeline", seeded_rain_is_reproducible_in_finite_ambient_field) {
    water::Config config;
    config.seed = 987;
    water::DropScheduler first(config), second(config);
    std::size_t impacts = 0;
    bool inside = false, outside = false;
    for (int step = 1; step <= 2400; ++step) {
        const double time = step / 120.0;
        const auto a = first.tick(time, true, 120.0f, 0.01f, view());
        const auto b = second.tick(time, true, 120.0f, 0.01f, view());
        CHECK(a.size() == b.size());
        impacts += a.size();
        CHECK(first.active().size() == second.active().size());
        for (std::size_t i = 0; i < first.active().size(); ++i) {
            const auto& drop = first.active()[i];
            const auto& repeated = second.active()[i];
            CHECK(drop.x >= -10 && drop.x <= 37.5f);
            CHECK(drop.z >= -10 && drop.z <= 39.2f);
            const bool pool = drop.x >= 0 && drop.x <= 20 && drop.z >= 0 && drop.z <= 12;
            inside |= pool;
            outside |= !pool;
            CHECK(drop.start == repeated.start);
            CHECK(drop.x == repeated.x && drop.z == repeated.z);
            CHECK_NEAR(drop.arrival - drop.start, std::sqrt(2 * drop.spawn_height / 9.81), 1e-9);
        }
        for (std::size_t i = 0; i < a.size(); ++i) {
            CHECK(a[i].x == b[i].x && a[i].z == b[i].z);
            CHECK(a[i].x >= 0 && a[i].x <= 20 && a[i].z >= 0 && a[i].z <= 12);
        }
    }
    CHECK(inside && outside);
    CHECK(impacts > 20 && impacts < 300);
}

TEST_CASE("timeline", rain_schedule_does_not_depend_on_tick_partition) {
    water::DropScheduler fine(water::Config{}), coarse(water::Config{});
    std::vector<water::Impact> events;
    for (int i = 1; i <= 480; ++i) {
        const auto arrivals = fine.tick(i / 120.0, true, 60.0f, 0.01f, view());
        events.insert(events.end(), arrivals.begin(), arrivals.end());
    }
    const auto arrivals = coarse.tick(4.0, true, 60.0f, 0.01f, view());
    CHECK(!events.empty());
    CHECK(events.size() == arrivals.size());
    for (std::size_t i = 0; i < events.size(); ++i) {
        CHECK(events[i].x == arrivals[i].x && events[i].z == arrivals[i].z);
    }
    CHECK(fine.active().size() == coarse.active().size());
    for (std::size_t i = 0; i < fine.active().size(); ++i) {
        CHECK(fine.active()[i].start == coarse.active()[i].start);
        CHECK(fine.active()[i].x == coarse.active()[i].x);
        CHECK(fine.active()[i].z == coarse.active()[i].z);
    }
}

TEST_CASE("timeline", rain_waits_vary_and_seed_changes_the_schedule) {
    water::Config config;
    water::DropScheduler first(config);
    config.seed = 2;
    water::DropScheduler other_seed(config);
    first.tick(0.5, true, 120.0f, 0.01f, view());
    other_seed.tick(0.5, true, 120.0f, 0.01f, view());
    CHECK(first.active().size() >= 3);
    CHECK(!other_seed.active().empty());
    CHECK(first.active()[0].start != other_seed.active()[0].start);
    const auto& drops = first.active();
    CHECK(drops[1].start > drops[0].start);
    CHECK(drops[2].start > drops[1].start);
    CHECK(std::abs((drops[1].start - drops[0].start) -
                   (drops[2].start - drops[1].start)) > 1e-6);
}

TEST_CASE("timeline", long_rain_interval_has_bounded_work_and_no_later_backlog) {
    water::DropScheduler drops(water::Config{});
    const auto events = drops.tick(10000.0, true, 120.0f, 0.01f, view());
    CHECK(events.size() <= 256);
    CHECK(!events.empty());
    CHECK(events.size() + drops.active().size() <= 256);
    const auto pending = std::count_if(drops.active().begin(), drops.active().end(), [](const water::Drop& drop) {
        return drop.x >= 0 && drop.x <= 20 && drop.z >= 0 && drop.z <= 12;
    });
    CHECK(drops.tick(10000.0, true, 120.0f, 0.01f, view()).empty());
    CHECK(drops.tick(10100.0, false, 120.0f, 0.01f, view()).size() == pending);
}

TEST_CASE("timeline", disabling_rain_keeps_falling_drops_without_backlog) {
    water::DropScheduler drops(water::Config{});
    CHECK(drops.tick(0.2, true, 120.0f, 0.01f, view()).empty());
    const auto falling = drops.active().size();
    CHECK(falling > 0);
    CHECK(drops.tick(0.21, false, 120.0f, 0.01f, view()).empty());
    CHECK(drops.active().size() == falling);
    const auto pool_flights = std::count_if(drops.active().begin(), drops.active().end(), [](const water::Drop& drop) {
        return drop.x >= 0 && drop.x <= 20 && drop.z >= 0 && drop.z <= 12;
    });
    CHECK(drops.tick(100.0, false, 120.0f, 0.01f, view()).size() == pool_flights);
    CHECK(drops.active().empty());
    CHECK(drops.tick(100.0, true, 120.0f, 0.01f, view()).empty());
    CHECK(drops.active().empty());
}

TEST_CASE("timeline", zero_rain_rate_does_not_build_a_backlog) {
    water::DropScheduler drops(water::Config{});
    CHECK(drops.tick(100.0, true, 0.0f, 0.01f, view()).empty());
    CHECK(drops.active().empty());
    CHECK(drops.tick(100.0, true, 120.0f, 0.01f, view()).empty());
    CHECK(drops.active().empty());
}

TEST_CASE("timeline", full_capacity_discards_automatic_rain_arrivals) {
    water::DropScheduler drops(water::Config{});
    for (int i = 0; i < 256; ++i) CHECK(drops.spawn(5.0f, 6.0f, 0.0, view()));
    CHECK(drops.tick(0.5, true, 120.0f, 0.01f, view()).empty());
    CHECK(drops.active().size() == 256);
    const double deadline = drops.active()[0].arrival;
    CHECK(drops.tick(deadline, false, 120.0f, 0.01f, view()).size() == 256);
    CHECK(drops.active().empty());
    CHECK(drops.tick(deadline, true, 120.0f, 0.01f, view()).empty());
    CHECK(drops.active().empty());
}

TEST_CASE("timeline", reset_restarts_seed_schedule_and_removes_manual_drops) {
    water::DropScheduler drops(water::Config{}), fresh(water::Config{});
    drops.tick(0.3, true, 120.0f, 0.01f, view());
    CHECK(drops.spawn(1.0f, 2.0f, 0.3, view()));
    drops.reset();
    CHECK(drops.active().empty());
    const auto a = drops.tick(4.0, true, 120.0f, 0.01f, view());
    const auto b = fresh.tick(4.0, true, 120.0f, 0.01f, view());
    CHECK(a.size() == b.size());
    CHECK(!a.empty());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].x == b[i].x && a[i].z == b[i].z);
            CHECK(a[i].x >= 0 && a[i].x <= 20 && a[i].z >= 0 && a[i].z <= 12);
    }
}

TEST_CASE("timeline", pause_freezes_in_flight_drop_until_simulation_resumes) {
    water::FixedClock clock;
    water::DropScheduler drops(water::Config{});
    CHECK(drops.spawn(5.0f, 6.0f, 0.0, view()));
    for (int i = 1; i <= 60; ++i) {
        CHECK(clock.advance(1.0 / 120.0, false) == 1);
        CHECK(drops.tick(i / 120.0, false, 8.0f, 0.01f, view()).empty());
    }
    CHECK(clock.advance(60.0, true) == 0);
    CHECK(clock.advance(0.0, false) == 0);
    CHECK(drops.active().size() == 1);
    const double deadline = drops.active()[0].arrival;
    const int arrival_tick = static_cast<int>(std::ceil(deadline * 120));
    for (int i = 61; i < arrival_tick; ++i) {
        CHECK(clock.advance(1.0 / 120.0, false) == 1);
        CHECK(drops.tick(i / 120.0, false, 8.0f, 0.01f, view()).empty());
    }
    CHECK(drops.tick(arrival_tick / 120.0, false, 8.0f, 0.01f, view()).size() == 1);
}

TEST_CASE("timeline", scheduler_rejects_invalid_controls_and_backward_time) {
    water::Config config;
    config.depth = 1;
    water::DropScheduler drops(config);
    drops.tick(1.0, false, 8.0f, 0.01f, view());
    CHECK_THROWS(drops.tick(0.9, false, 8.0f, 0.01f, view()), std::invalid_argument);
    CHECK_THROWS(drops.spawn(1.0f, 2.0f, 0.9, view()), std::invalid_argument);
    CHECK_THROWS(drops.tick(1.0, true, -1.0f, 0.01f, view()), std::invalid_argument);
    CHECK_THROWS(drops.tick(1.0, true, 120.1f, 0.01f, view()), std::invalid_argument);
    CHECK_THROWS(drops.tick(1.0, true, 8.0f, 0.021f, view()), std::invalid_argument);
    CHECK_THROWS(drops.tick(std::numeric_limits<double>::quiet_NaN(), false, 8.0f, 0.01f, view()),
                 std::invalid_argument);
    CHECK(!drops.spawn(-0.1f, 2.0f, 1.0, view()));
    CHECK(!drops.spawn(1.0f, 12.1f, 1.0, view()));
}

TEST_CASE("timeline", picking_rejects_parallel_upward_and_outside_rays) {
    const water::Grid grid;
    for (const Ray ray : {Ray{{0, 2, 0}, {1, 0, 0}}, Ray{{0, 2, 0}, {0, 1, 0}},
                          Ray{{0, -2, 0}, {0, -1, 0}}, Ray{{-10.01f, 2, 0}, {0, -1, 0}},
                          Ray{{10.01f, 2, 0}, {0, -1, 0}}, Ray{{0, 2, -6.01f}, {0, -1, 0}},
                          Ray{{0, 2, 6.01f}, {0, -1, 0}}}) {
        CHECK(!water::pick_water(ray, grid).has_value());
    }
    CHECK(!water::pick_water({{0, 2, 0}, {0, std::numeric_limits<float>::quiet_NaN(), 0}},
                             grid).has_value());
}

TEST_CASE("timeline", picking_maps_corners_and_oblique_camera_rays_to_basin) {
    const water::Grid grid;
    const Vector2 expected[] = {{0, 0}, {20, 0}, {0, 12}, {20, 12}, {13, 4}, {6, 9}};
    const Ray rays[] = {{{-10, 2, -6}, {0, -1, 0}}, {{10, 2, -6}, {0, -1, 0}},
                        {{-10, 2, 6}, {0, -1, 0}}, {{10, 2, 6}, {0, -1, 0}},
                        {{15, 8, 10}, {-1.5f, -1, -1.5f}},
                        {{-20, 12, -9}, {4, -3, 3}}};
    for (int i = 0; i < 6; ++i) {
        const auto point = water::pick_water(rays[i], grid);
        CHECK(point.has_value());
        CHECK_NEAR(point->x, expected[i].x, 1e-5f);
        CHECK_NEAR(point->y, expected[i].y, 1e-5f);
    }
}

TEST_CASE("timeline", depth_changes_preserve_flights_random_schedule_and_strength_validation) {
    water::Config config;
    water::DropScheduler actual(config), reference(config);
    actual.tick(.5, true, 60, .01f, view());
    reference.tick(.5, true, 60, .01f, view());
    CHECK(!actual.active().empty());
    const auto before = actual.active();
    actual.set_depth(.25f);
    CHECK(actual.active().size() == before.size());
    for (std::size_t i = 0; i < before.size(); ++i) {
        const auto a = actual.active()[i], b = before[i];
        CHECK(a.x == b.x && a.z == b.z && a.start == b.start);
        CHECK(a.spawn_height == b.spawn_height && a.arrival == b.arrival);
    }
    CHECK_THROWS(actual.tick(.6, true, 60, .006f, view()), std::invalid_argument);
    for (float invalid : {0.f, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()})
        CHECK_THROWS(actual.set_depth(invalid), std::invalid_argument);
    actual.set_depth(std::numeric_limits<float>::max());
    actual.set_depth(.25f);
    actual.tick(.6, true, 60, .005f, view());
    reference.tick(.6, true, 60, .005f, view());
    CHECK(actual.active().size() == reference.active().size());
    for (std::size_t i = 0; i < actual.active().size(); ++i) {
        const auto a = actual.active()[i], b = reference.active()[i];
        CHECK(a.x == b.x && a.z == b.z && a.start == b.start);
        CHECK(a.spawn_height == b.spawn_height && a.arrival == b.arrival);
    }
    actual.reset();
    CHECK_THROWS(actual.tick(.1, false, 0, .006f, view()), std::invalid_argument);
}

TEST_CASE("timeline", exterior_rain_disappears_without_impact) {
    water::DropScheduler drops(water::Config{});
    double time = 0;
    while (drops.active().empty()) drops.tick(time += 1.0 / 120, true, 60, .01f, view());
    CHECK(drops.active().size() == 1);
    const auto first = drops.active()[0];
    CHECK(first.x < 0 || first.x > 20 || first.z < 0 || first.z > 12);
    CHECK(drops.tick(first.arrival, false, 0, .01f, view()).empty());
    CHECK(drops.active().empty());
}

TEST_CASE("timeline", inclusive_water_edges_arrive_but_outside_clicks_fail) {
    water::DropScheduler drops(water::Config{});
    for (Vector2 p : {Vector2{0,0}, {20,0}, {0,12}, {20,12}, {10,0}, {20,6}})
        CHECK(drops.spawn(p.x, p.y, 0, view()));
    CHECK(!drops.spawn(-.001f, 0, 0, view()));
    CHECK(!drops.spawn(20.001f, 0, 0, view()));
    CHECK(drops.tick(100, false, 0, .01f, view()).size() == 6);
    CHECK(drops.active().empty());
}

TEST_CASE("timeline", camera_motion_changes_only_future_ambient_births) {
    water::DropScheduler actual(water::Config{}), reference(water::Config{});
    actual.tick(.2, true, 60, .01f, view());
    reference.tick(.2, true, 60, .01f, view());
    const auto before = actual.active();
    auto moved = view();
    moved.camera.position = {-70, 60, -50};
    actual.tick(.5, true, 60, .01f, moved);
    reference.tick(.5, true, 60, .01f, view());
    CHECK(actual.active().size() > before.size());
    for (std::size_t i = 0; i < before.size(); ++i) {
        const auto drop = actual.active()[i], previous = before[i];
        CHECK(drop.x == previous.x && drop.z == previous.z && drop.start == previous.start);
        CHECK(drop.arrival == previous.arrival && drop.spawn_height == previous.spawn_height);
    }
    bool changed = false;
    for (std::size_t i = before.size(); i < actual.active().size(); ++i) {
        const auto drop = actual.active()[i];
        CHECK(drop.x >= -70 && drop.x <= 30 && drop.z >= -54 && drop.z <= 22);
        CHECK(drop.start == reference.active()[i].start);
        changed |= drop.x != reference.active()[i].x || drop.z != reference.active()[i].z;
    }
    CHECK(changed);
}
