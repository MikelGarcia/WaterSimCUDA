#include "timeline.hpp"
#include "test_support.hpp"

#include <limits>

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

TEST_CASE("timeline", drop_arrives_once_at_its_simulation_deadline) {
    water::Config config;
    water::DropScheduler drops(config);
    CHECK(drops.spawn(4.0f, 7.0f, 1.25));
    CHECK(drops.active().size() == 1);
    CHECK_NEAR(drops.active()[0].start, 1.25, 1e-12);
    CHECK_NEAR(drops.active()[0].arrival, 1.85, 1e-12);
    CHECK(drops.tick(1.849, false, 8.0f, 0.015f).empty());
    const auto events = drops.tick(1.85, false, 8.0f, 0.015f);
    CHECK(events.size() == 1);
    CHECK_NEAR(events[0].x, 4.0f, 1e-6f);
    CHECK_NEAR(events[0].z, 7.0f, 1e-6f);
    CHECK_NEAR(events[0].radius, config.impact_radius, 1e-6f);
    CHECK_NEAR(events[0].strength, 0.015f, 1e-6f);
    CHECK(drops.active().empty());
    CHECK(drops.tick(2.0, false, 8.0f, 0.015f).empty());
}

TEST_CASE("timeline", capacity_rejection_creates_no_later_impact) {
    water::DropScheduler drops(water::Config{});
    for (int i = 0; i < 256; ++i) CHECK(drops.spawn(5.0f, 6.0f, 0.0));
    CHECK(!drops.spawn(7.0f, 8.0f, 0.0));
    CHECK(drops.tick(0.6, false, 8.0f, 0.01f).size() == 256);
    CHECK(drops.tick(1.2, false, 8.0f, 0.01f).empty());
    CHECK(drops.spawn(7.0f, 8.0f, 1.2));
}

TEST_CASE("timeline", seeded_rain_is_reproducible_and_stays_inside_basin) {
    water::Config config;
    config.seed = 987;
    water::DropScheduler first(config), second(config);
    std::size_t impacts = 0;
    for (int step = 1; step <= 1200; ++step) {
        const double time = step / 120.0;
        const auto a = first.tick(time, true, 60.0f, 0.01f);
        const auto b = second.tick(time, true, 60.0f, 0.01f);
        CHECK(a.size() == b.size());
        impacts += a.size();
        CHECK(first.active().size() == second.active().size());
        for (std::size_t i = 0; i < first.active().size(); ++i) {
            const auto& drop = first.active()[i];
            const auto& repeated = second.active()[i];
            CHECK(drop.x >= 0.0f && drop.x <= 20.0f);
            CHECK(drop.z >= 0.0f && drop.z <= 12.0f);
            CHECK(drop.start == repeated.start);
            CHECK(drop.x == repeated.x && drop.z == repeated.z);
            CHECK_NEAR(drop.arrival - drop.start, 0.6, 1e-12);
        }
        for (std::size_t i = 0; i < a.size(); ++i) {
            CHECK(a[i].x == b[i].x && a[i].z == b[i].z);
        }
    }
    CHECK(impacts > 450 && impacts < 750);
}

TEST_CASE("timeline", rain_schedule_does_not_depend_on_tick_partition) {
    water::DropScheduler fine(water::Config{}), coarse(water::Config{});
    std::vector<water::Impact> events;
    for (int i = 1; i <= 120; ++i) {
        const auto arrivals = fine.tick(i / 120.0, true, 60.0f, 0.01f);
        events.insert(events.end(), arrivals.begin(), arrivals.end());
    }
    const auto arrivals = coarse.tick(1.0, true, 60.0f, 0.01f);
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
    first.tick(0.5, true, 60.0f, 0.01f);
    other_seed.tick(0.5, true, 60.0f, 0.01f);
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
    const auto events = drops.tick(10000.0, true, 60.0f, 0.01f);
    CHECK(events.size() == 256);
    CHECK(drops.active().empty());
    CHECK(drops.tick(10000.0, true, 60.0f, 0.01f).empty());
}

TEST_CASE("timeline", disabling_rain_keeps_falling_drops_without_backlog) {
    water::DropScheduler drops(water::Config{});
    CHECK(drops.tick(0.2, true, 60.0f, 0.01f).empty());
    const auto falling = drops.active().size();
    CHECK(falling > 0);
    CHECK(drops.tick(0.21, false, 60.0f, 0.01f).empty());
    CHECK(drops.active().size() == falling);
    CHECK(drops.tick(100.0, false, 60.0f, 0.01f).size() == falling);
    CHECK(drops.active().empty());
    CHECK(drops.tick(100.0, true, 60.0f, 0.01f).empty());
    CHECK(drops.active().empty());
}

TEST_CASE("timeline", zero_rain_rate_does_not_build_a_backlog) {
    water::DropScheduler drops(water::Config{});
    CHECK(drops.tick(100.0, true, 0.0f, 0.01f).empty());
    CHECK(drops.active().empty());
    CHECK(drops.tick(100.0, true, 60.0f, 0.01f).empty());
    CHECK(drops.active().empty());
}

TEST_CASE("timeline", full_capacity_discards_automatic_rain_arrivals) {
    water::DropScheduler drops(water::Config{});
    for (int i = 0; i < 256; ++i) CHECK(drops.spawn(5.0f, 6.0f, 0.0));
    CHECK(drops.tick(0.5, true, 60.0f, 0.01f).empty());
    CHECK(drops.active().size() == 256);
    CHECK(drops.tick(0.6, false, 60.0f, 0.01f).size() == 256);
    CHECK(drops.active().empty());
    CHECK(drops.tick(0.6, true, 60.0f, 0.01f).empty());
    CHECK(drops.active().empty());
}

TEST_CASE("timeline", reset_restarts_seed_schedule_and_removes_manual_drops) {
    water::DropScheduler drops(water::Config{}), fresh(water::Config{});
    drops.tick(0.3, true, 60.0f, 0.01f);
    CHECK(drops.spawn(1.0f, 2.0f, 0.3));
    drops.reset();
    CHECK(drops.active().empty());
    const auto a = drops.tick(1.0, true, 60.0f, 0.01f);
    const auto b = fresh.tick(1.0, true, 60.0f, 0.01f);
    CHECK(a.size() == b.size());
    CHECK(!a.empty());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].x == b[i].x && a[i].z == b[i].z);
    }
}

TEST_CASE("timeline", pause_freezes_in_flight_drop_until_simulation_resumes) {
    water::FixedClock clock;
    water::DropScheduler drops(water::Config{});
    CHECK(drops.spawn(5.0f, 6.0f, 0.0));
    for (int i = 1; i <= 60; ++i) {
        CHECK(clock.advance(1.0 / 120.0, false) == 1);
        CHECK(drops.tick(i / 120.0, false, 8.0f, 0.01f).empty());
    }
    CHECK(clock.advance(60.0, true) == 0);
    CHECK(clock.advance(0.0, false) == 0);
    CHECK(drops.active().size() == 1);
    for (int i = 61; i < 72; ++i) {
        CHECK(clock.advance(1.0 / 120.0, false) == 1);
        CHECK(drops.tick(i / 120.0, false, 8.0f, 0.01f).empty());
    }
    CHECK(drops.tick(0.6, false, 8.0f, 0.01f).size() == 1);
}

TEST_CASE("timeline", scheduler_rejects_invalid_controls_and_backward_time) {
    water::DropScheduler drops(water::Config{});
    drops.tick(1.0, false, 8.0f, 0.01f);
    CHECK_THROWS(drops.tick(0.9, false, 8.0f, 0.01f), std::invalid_argument);
    CHECK_THROWS(drops.spawn(1.0f, 2.0f, 0.9), std::invalid_argument);
    CHECK_THROWS(drops.tick(1.0, true, -1.0f, 0.01f), std::invalid_argument);
    CHECK_THROWS(drops.tick(1.0, true, 61.0f, 0.01f), std::invalid_argument);
    CHECK_THROWS(drops.tick(1.0, true, 8.0f, 0.021f), std::invalid_argument);
    CHECK_THROWS(drops.tick(std::numeric_limits<double>::quiet_NaN(), false, 8.0f, 0.01f),
                 std::invalid_argument);
    CHECK(!drops.spawn(-0.1f, 2.0f, 1.0));
    CHECK(!drops.spawn(1.0f, 12.1f, 1.0));
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
