#include "config.hpp"
#include "test_support.hpp"

#include <initializer_list>
#include <limits>
#include <string>

namespace {

water::Config parse(std::initializer_list<const char*> arguments) {
    std::vector<const char*> argv{"water_demo"};
    argv.insert(argv.end(), arguments);
    return water::parse_config(static_cast<int>(argv.size()), argv.data());
}

} // namespace

TEST_CASE("config", defaults_are_valid) {
    const auto config = parse({});
    water::validate(config);
    CHECK(config.grid.nx == 512);
    CHECK(config.grid.ny == 512);
    CHECK_NEAR(config.grid.lx, 20.0f, 1e-6f);
    CHECK_NEAR(config.grid.lz, 12.0f, 1e-6f);
    CHECK_NEAR(config.depth, 1.0f, 1e-6f);
    CHECK_NEAR(config.gravity, 9.81f, 1e-6f);
    CHECK_NEAR(config.damping, 0.15f, 1e-6f);
    CHECK_NEAR(config.impact_radius, 0.35f, 1e-6f);
    CHECK_NEAR(config.impact_strength, 0.01f, 1e-6f);
    CHECK_NEAR(config.rain_rate, 8.0f, 1e-6f);
    CHECK(config.seed == 1);
}

TEST_CASE("config", parses_rectangular_grid_and_all_options) {
    const auto config = parse({"--nx", "96", "--ny", "64", "--lx", "20",
                               "--lz", "12", "--radius", "0.75", "--depth", "2",
                               "--damping", "0.3", "--strength", "0.025",
                               "--rain-rate", "12", "--seed", "42"});
    CHECK(config.grid.nx == 96);
    CHECK(config.grid.ny == 64);
    CHECK_NEAR(config.grid.lx, 20.0f, 1e-6f);
    CHECK_NEAR(config.grid.lz, 12.0f, 1e-6f);
    CHECK_NEAR(config.impact_radius, 0.75f, 1e-6f);
    CHECK_NEAR(config.depth, 2.0f, 1e-6f);
    CHECK_NEAR(config.damping, 0.3f, 1e-6f);
    CHECK_NEAR(config.impact_strength, 0.025f, 1e-6f);
    CHECK_NEAR(config.rain_rate, 12.0f, 1e-6f);
    CHECK(config.seed == 42);
}

TEST_CASE("config", rejects_unknown_missing_and_malformed_options) {
    CHECK_THROWS(parse({"--unknown", "1"}), std::invalid_argument);
    CHECK_THROWS(parse({"--nx"}), std::invalid_argument);
    for (const char* value : {"", "abc", "96x", "1.5", "99999999999999999999"}) {
        CHECK_THROWS(parse({"--nx", value}), std::invalid_argument);
    }
    for (const char* value : {"", "abc", "1x", "1e999", "1e-999"}) {
        CHECK_THROWS(parse({"--depth", value}), std::invalid_argument);
    }
    for (const char* value : {"-1", "1.5", "4294967296"}) {
        CHECK_THROWS(parse({"--seed", value}), std::invalid_argument);
    }
    try {
        parse({"--nx", "banana"});
        CHECK(false);
    } catch (const std::invalid_argument& error) {
        CHECK(std::string(error.what()).find("--nx") != std::string::npos);
    }
}

TEST_CASE("config", rejects_invalid_grid_and_physics) {
    const water::Config valid;
    for (int water::Grid::* field : {&water::Grid::nx, &water::Grid::ny}) {
        for (int value : {7, 2049}) {
            auto config = valid;
            config.grid.*field = value;
            CHECK_THROWS(water::validate(config), std::invalid_argument);
        }
    }
    for (float water::Grid::* field : {&water::Grid::lx, &water::Grid::lz}) {
        for (float value : {0.0f, -1.0f}) {
            auto config = valid;
            config.grid.*field = value;
            CHECK_THROWS(water::validate(config), std::invalid_argument);
        }
    }
    for (float water::Config::* field : {&water::Config::depth, &water::Config::gravity,
                                        &water::Config::impact_radius}) {
        for (float value : {0.0f, -1.0f}) {
            auto config = valid;
            config.*field = value;
            CHECK_THROWS(water::validate(config), std::invalid_argument);
        }
    }
    for (float water::Config::* field : {&water::Config::damping, &water::Config::rain_rate,
                                        &water::Config::impact_strength}) {
        auto config = valid;
        config.*field = -0.01f;
        CHECK_THROWS(water::validate(config), std::invalid_argument);
    }
    CHECK_THROWS(parse({"--rain-rate", "60.1"}), std::invalid_argument);
    CHECK_THROWS(parse({"--strength", "0.0201"}), std::invalid_argument);
    CHECK_THROWS(parse({"--radius", "0.01"}), std::invalid_argument);
    CHECK_THROWS(parse({"--radius", "3.1"}), std::invalid_argument);
    CHECK_THROWS(parse({"--nx", "96", "--ny", "64"}), std::invalid_argument);
}

TEST_CASE("config", rejects_nonfinite_values_in_every_float_field) {
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
        for (float water::Grid::* field : {&water::Grid::lx, &water::Grid::lz}) {
            water::Config config;
            config.grid.*field = value;
            CHECK_THROWS(water::validate(config), std::invalid_argument);
        }
        for (float water::Config::* field : {&water::Config::depth, &water::Config::gravity,
                                            &water::Config::damping, &water::Config::impact_radius,
                                            &water::Config::impact_strength, &water::Config::rain_rate}) {
            water::Config config;
            config.*field = value;
            CHECK_THROWS(water::validate(config), std::invalid_argument);
        }
    }
    CHECK_THROWS(parse({"--depth", "nan"}), std::invalid_argument);
    CHECK_THROWS(parse({"--damping", "inf"}), std::invalid_argument);
}

TEST_CASE("config", accepts_validation_boundaries) {
    auto config = parse({"--nx", "8", "--ny", "12", "--lx", "4", "--lz", "6",
                         "--radius", "1", "--strength", "0.02", "--damping", "0",
                         "--rain-rate", "60", "--seed", "0"});
    water::validate(config);
    config.grid = {2048, 2048, 20.0f, 12.0f};
    config.impact_radius = 3.0f;
    config.impact_strength = 0.0f;
    config.rain_rate = 0.0f;
    water::validate(config);
    CHECK(parse({"--seed", "4294967295"}).seed == std::numeric_limits<unsigned>::max());
}

TEST_CASE("config", timestep_respects_both_axes_and_quantum) {
    for (const water::Grid grid : {water::Grid{}, water::Grid{96, 64, 20.0f, 12.0f},
                                   water::Grid{64, 96, 20.0f, 12.0f}}) {
        water::Config config;
        config.grid = grid;
        config.impact_radius = 0.75f;
        const double dx = static_cast<double>(grid.lx) / grid.nx;
        const double dz = static_cast<double>(grid.lz) / grid.ny;
        const double rate = std::sqrt(static_cast<double>(config.gravity) * config.depth) *
                            std::sqrt(1.0 / (dx * dx) + 1.0 / (dz * dz));
        const double dt = water::stable_dt(config);
        CHECK(dt > 0.0);
        CHECK(dt * rate <= 0.8 + 1e-12);
        CHECK_NEAR(dt * rate, 0.8, 1e-12);
        const int substeps = water::substeps_per_quantum(config);
        CHECK(substeps >= 1);
        CHECK((1.0 / 120.0) / substeps <= dt);
        if (substeps > 1) CHECK((1.0 / 120.0) / (substeps - 1) > dt);
    }
    CHECK(water::substeps_per_quantum(water::Config{}) == 2);
    auto coarse = parse({"--nx", "96", "--ny", "64", "--radius", "0.75"});
    CHECK(water::substeps_per_quantum(coarse) == 1);
    CHECK_NEAR(water::stable_dt(coarse), 0.0355973328, 1e-9);
    coarse.damping = 100.0f;
    CHECK_NEAR(water::stable_dt(coarse), 0.0355973328, 1e-9);
}

TEST_CASE("config", timestep_rejects_invalid_or_unrepresentable_substep_counts) {
    water::Config config;
    config.grid.nx = 0;
    CHECK_THROWS(water::stable_dt(config), std::invalid_argument);
    CHECK_THROWS(water::substeps_per_quantum(config), std::invalid_argument);
    config = water::Config{};
    config.depth = std::numeric_limits<float>::max();
    CHECK_THROWS(water::substeps_per_quantum(config), std::invalid_argument);
}

TEST_CASE("config", float_timestep_stays_within_cfl_bound_at_subdivision_boundary) {
    const auto config = parse({"--depth", "1.5178088"});
    const int substeps = water::substeps_per_quantum(config);
    const float dt = static_cast<float>((1.0 / 120.0) / substeps);
    CHECK(dt > 0.0f);
    CHECK(dt <= water::stable_dt(config));
}
