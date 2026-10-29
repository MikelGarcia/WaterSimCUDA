#include "dispersion_reference.hpp"
#include "validation.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>

namespace {
water::Config fixture() {
    water::Config c;
    c.grid = {80, 64, 4, 3};
    c.impact_radius = .15f;
    return c;
}
double ramp(double x) {
    x = std::clamp(x, 0.0, 1.0);
    return x * x * (3 - 2 * x);
}
double head(double x, double z, const water::BoatState& b, const water::Config& c) {
    const double vx = b.vx, vz = b.vz, yaw = b.yaw;
    const double speed = std::hypot(vx, vz);
    if (speed <= 1e-4)
        return 0;
    const double cell = std::max(double(c.grid.lx) / c.grid.nx, double(c.grid.lz) / c.grid.ny);
    const double length = std::max(double(water::boat_geometry.length), 4 * cell);
    const double width = std::max(.07, 2 * cell);
    const double longitudinal = (x - b.x) * std::cos(yaw) + (z - b.z) * std::sin(yaw);
    const double transverse = -(x - b.x) * std::sin(yaw) + (z - b.z) * std::cos(yaw);
    auto ridge = [&](double l, double sign) {
        const double aft = .5 * water::boat_geometry.length - l;
        const double q = (transverse - sign * .7 * aft) / width;
        if (aft <= 0 || aft >= 2 * length || std::abs(q) >= 1) return 0.0;
        const double envelope = ramp(aft / (.2 * length)) *
            (1 - ramp((aft - 1.5 * length) / (.5 * length)));
        return envelope * std::pow(1 - q * q, 3);
    };
    const double blend = ramp((vx * std::cos(yaw) + vz * std::sin(yaw) + .01) / .02);
    const double pressure = blend * (ridge(longitudinal, -1) + ridge(longitudinal, 1)) +
        (1 - blend) * (ridge(-longitudinal, -1) + ridge(-longitudinal, 1));
    return .012 * (speed * speed / (1.875 * 1.875)) * ramp((speed - .0001) / .0099) *
        std::min(1.0, double(c.depth) / 3.0) * std::sqrt(std::min(1.0, double(c.depth) / 3.0)) * (1 + .3 * std::sin(b.wake_phase)) * pressure;
}
water::Snapshot forced(water::Simulation& s, const water::Config& c, water::BoatState b, float dt,
                       float damping) {
    s.reset();
    water::upload_boat(s, b);
    s.step(dt, damping);
    b = water::download_boat(s);
    const auto got = water::download(s);
    const auto& g = c.grid;
    const double dx = double(g.lx) / g.nx, dz = double(g.lz) / g.ny;
    std::vector<double> p(g.nx * g.ny);
    for (int z = 0; z < g.ny; ++z)
        for (int x = 0; x < g.nx; ++x)
            p[z * g.nx + x] = head((x + .5) * dx, (z + .5) * dz, b, c);
    water::Snapshot initial{std::vector<float>(got.eta.size()), std::vector<float>(got.u.size()), std::vector<float>(got.v.size())};
    const auto expected = reference::step(c, initial, dt, damping, p);
    auto close = [](double actual, double expected) {
        CHECK(std::abs(actual - expected) <= 2e-6 + 2e-5 * std::abs(expected));
    };
    for (size_t i = 0; i < got.u.size(); ++i) close(got.u[i], expected.u[i]);
    for (size_t i = 0; i < got.v.size(); ++i) close(got.v[i], expected.v[i]);
    for (size_t i = 0; i < got.eta.size(); ++i) close(got.eta[i], expected.eta[i]);
    return got;
}
double intensity(const water::Snapshot& s) {
    double sum = 0;
    for (const auto* a : {&s.u, &s.v})
        for (float x : *a)
            sum += double(x) * x;
    return sum;
}
} // namespace
TEST_CASE("wake", pressure_starts_at_leading_bow_and_opens_aft_on_both_sides) {
    auto c = fixture();
    c.grid = {320, 320, 6, 6};
    water::Simulation simulation(c);
    constexpr float dt = .00001f;
    for (int orientation = 0; orientation < 3; ++orientation) {
        simulation.reset();
        const float yaw = orientation == 1 ? 1.57079632679f : 0;
        const float direction = orientation == 2 ? -1.f : 1.f;
        water::upload_boat(simulation, {3, 3, yaw, direction * 1.5f * std::cos(yaw),
                                       direction * 1.5f * std::sin(yaw), 0, 1.57079632679f});
        simulation.step(dt, 0);
        const auto state = water::download(simulation);
        const auto boat = water::download_boat(simulation);
        const float dx = c.grid.lx / c.grid.nx;
        std::vector<double> pressure(state.eta.size());
        for (int z = 0; z < c.grid.ny; ++z)
            for (int x = 1; x < c.grid.nx; ++x)
                pressure[z * c.grid.nx + x] = pressure[z * c.grid.nx + x - 1] -
                    dx * state.u[z * (c.grid.nx + 1) + x] / (c.gravity * dt);
        auto sample = [&](double longitudinal, double transverse) {
            longitudinal *= direction;
            const double x = boat.x + longitudinal * std::cos(yaw) - transverse * std::sin(yaw);
            const double z = boat.z + longitudinal * std::sin(yaw) + transverse * std::cos(yaw);
            return pressure[int(z / dx) * c.grid.nx + int(x / dx)];
        };
        for (double aft : {.15, .45, .9}) {
            const double l = .3 - aft, t = .7 * aft;
            const double left = sample(l, -t), right = sample(l, t);
            CHECK(left > .0003 && right > .0003);
            CHECK(std::abs(left - right) < .0002);
            CHECK(left > 2 * std::abs(sample(l, 0)));
            CHECK(left > 2 * std::abs(sample(.3 + aft, t)));
        }
        CHECK(sample(.24, .042) > .0001);
        CHECK(std::abs(sample(.5, 0)) < 1e-6);
    }
}

TEST_CASE("wake", independent_pressure_gradient_and_continuity) {
    for (bool transpose : {false, true}) {
        auto c = fixture();
        if (transpose) {
            std::swap(c.grid.nx, c.grid.ny);
            std::swap(c.grid.lx, c.grid.lz);
        }
        water::Simulation s(c);
        for (float damping : {0.f, 2.5f})
            for (auto b : {water::BoatState{1.5f, 1.5f, .4f, .9f, .3f, 0},
                           {1.5f, 1.5f, 0, -.75f, 0, 0},
                           {.30f, .12f, 0, .4f, .3f, 0},
                           {1.5f, 1.5f, 0, 0, .375f, 0},
                           {1.5f, 1.5f, 0, -.75f, 0, 0, .7f},
                           {1.5f, 1.5f, .4f, .9f, .3f, 0, 4.2f}})
                CHECK(intensity(forced(s, c, b, .001f, damping)) > 1e-10);
    }
}
TEST_CASE("wake", coarse_grid_pressure_remains_anchored_to_the_physical_leading_bow) {
    auto c = fixture();
    c.grid = {32, 24, 6, 4};
    c.impact_radius = .4f;
    water::Simulation simulation(c);
    for (const auto b : {water::BoatState{3, 2, 0, 1.5f, 0, 0},
                         {3, 2, 0, -1.5f, 0, 0},
                         {3, 2, 1.57079632679f, 0, 1.5f, 0}})
        CHECK(intensity(forced(simulation, c, b, .001f, 0)) > 1e-10);
}
TEST_CASE("wake", rest_coasting_speed_and_lateral_continuity) {
    auto c = fixture();
    water::Simulation s(c);
    auto rest = forced(s, c, {2, 1.5f, 0, 0, 0, 0}, .001f, 2.5f);
    CHECK(intensity(rest) == 0);
    for (float e : rest.eta)
        CHECK(e == 0);
    auto insignificant = forced(s, c, {2, 1.5f, 0, .0001f, 0, 0}, .001f, 0);
    CHECK(intensity(insignificant) == 0);
    for (float e : insignificant.eta)
        CHECK(e == 0);
    auto slow = forced(s, c, {2, 1.5f, 0, .375f, 0, 0}, .001f, 0);
    auto fast = forced(s, c, {2, 1.5f, 0, .75f, 0, 0}, .001f, 0);
    CHECK(intensity(fast) > 15 * intensity(slow));
    auto forward = forced(s, c, {2, 1.5f, 0, 1.5f, 0, 0}, .001f, 0);
    CHECK(intensity(forward) > 15 * intensity(fast));
    auto middle = forced(s, c, {2, 1.5f, 0, 0, .3f, 0}, .001f, 0);
    for (float vx : {-1e-6f, 1e-6f}) {
        auto side = forced(s, c, {2, 1.5f, 0, vx, .3f, 0}, .001f, 0);
        CHECK(intensity(side) > 0);
        for (size_t i = 0; i < side.u.size(); ++i)
            CHECK(std::abs(side.u[i] - middle.u[i]) < 1e-7);
    }
    auto reverse = forced(s, c, {2, 1.5f, 0, -.75f, 0, 0}, .001f, 0);
    CHECK(std::abs(intensity(reverse) / intensity(fast) - 1) < .02);
    for (int z = 0; z < c.grid.ny; ++z) {
        for (int x = 0; x < c.grid.nx; ++x)
            CHECK_NEAR(reverse.eta[z * c.grid.nx + x], fast.eta[z * c.grid.nx + c.grid.nx - 1 - x],
                       2e-6);
        for (int x = 0; x <= c.grid.nx; ++x)
            CHECK_NEAR(reverse.u[z * (c.grid.nx + 1) + x],
                       -fast.u[z * (c.grid.nx + 1) + c.grid.nx - x], 2e-6);
    }
}

TEST_CASE("wake", moving_pressure_preserves_existing_waves_and_wake_history) {
    const auto c = fixture();
    const auto& g = c.grid;
    for (float damping : {0.f, 2.5f}) {
        water::Simulation background(c), wake(c), combined(c), instantaneous(c);
        auto initial = water::download(background);
        constexpr float pi = 3.14159265359f;
        for (int z = 0; z < g.ny; ++z) {
            for (int x = 0; x < g.nx; ++x)
                initial.eta[z * g.nx + x] = .0004f + .0007f *
                    std::cos(pi * (x + .5f) / g.nx) * std::cos(pi * (z + .5f) / g.ny);
            for (int x = 1; x < g.nx; ++x)
                initial.u[z * (g.nx + 1) + x] = .0003f * std::sin(pi * x / g.nx);
        }
        for (int z = 1; z < g.ny; ++z)
            for (int x = 0; x < g.nx; ++x)
                initial.v[z * g.nx + x] = -.0002f * std::sin(pi * z / g.ny);
        water::upload(background, initial);
        water::upload(combined, initial);
        const water::BoatState boat{1.5f, 1.5f, .4f, .9f, .3f, 0};
        water::upload_boat(wake, boat);
        water::upload_boat(combined, boat);
        constexpr float dt = .001f;
        for (int step = 0; step < 200; ++step) {
            instantaneous.reset();
            water::upload_boat(instantaneous, water::download_boat(wake));
            instantaneous.step(dt, damping, {1, .2f});
            background.step(dt, damping);
            wake.step(dt, damping, {1, .2f});
            combined.step(dt, damping, {1, .2f});
            const auto a = water::download(background);
            const auto b = water::download(wake);
            const auto sum = water::download(combined);
            CHECK_NEAR(std::accumulate(sum.eta.begin(), sum.eta.end(), 0.0) / sum.eta.size(),
                       .0004, 1e-9);
            for (auto fields : {std::array<const std::vector<float>*, 3>{&a.eta, &b.eta, &sum.eta},
                                {&a.u, &b.u, &sum.u}, {&a.v, &b.v, &sum.v}}) {
                double error = 0, scale = 0;
                for (size_t i = 0; i < fields[0]->size(); ++i) {
                    const double expected = double((*fields[0])[i]) + (*fields[1])[i];
                    error = std::max(error, std::abs(double((*fields[2])[i]) - expected));
                    scale = std::max(scale, std::abs(expected));
                }
                if (step == 199) std::cout << "HISTORY damping=" << damping << " error=" << error << " scale=" << scale << '\n';
                CHECK(error <= 2e-8 + 2e-6 * scale);
            }
        }
        const auto accumulated = water::download(wake);
        const auto latest = water::download(instantaneous);
        CHECK(intensity(accumulated) > 10 * intensity(latest));
        const double mean = std::accumulate(accumulated.eta.begin(), accumulated.eta.end(), 0.0) /
                            accumulated.eta.size();
        CHECK(std::abs(mean) < 1e-9);
        std::cout << "WAKE HISTORY damping=" << damping << " accumulated/latest velocity energy="
                  << intensity(accumulated) / intensity(latest) << '\n';
    }
}

namespace {
void boat_soak(water::Config c) {
    water::Simulation s(c);
    const int subdivisions = water::substeps_per_quantum(c);
    const float dt = float((1.0 / 120) / subdivisions);
    double maximum_mean = 0;
    float peak = 0;
    for (int second = 1; second <= 120; ++second) {
        // Each leg starts with the real moving boat near a different wall/corner.
        // Water is never reset, so every forcing episode accumulates in the same field.
        if (second % 15 == 1) {
            const int leg = (second - 1) / 15;
            const float yaw = float(leg) * .7853981634f;
            const float ex = .5f * (water::boat_geometry.length * std::abs(std::cos(yaw)) +
                                    water::boat_geometry.beam * std::abs(std::sin(yaw)));
            const float ez = .5f * (water::boat_geometry.length * std::abs(std::sin(yaw)) +
                                    water::boat_geometry.beam * std::abs(std::cos(yaw)));
            const float x = (leg & 1) ? ex + .001f : c.grid.lx - ex - .001f;
            const float z = (leg & 2) ? ez + .001f : c.grid.lz - ez - .001f;
            water::upload_boat(s, {x, z, yaw, 1.875f * std::cos(yaw), 1.875f * std::sin(yaw), 0});
        }
        for (int tick = 0; tick < 120; ++tick) {
            water::BoatControl control{1, 0};
            if (second > 40)
                control.rudder = 1;
            if (second > 80)
                control = {-1, -1};
            for (int sub = 0; sub < subdivisions; ++sub)
                s.step(dt, c.damping, control);
        }
        const auto state = water::download(s);
        for (const auto* values : {&state.eta, &state.u, &state.v})
            for (float value : *values)
                CHECK(std::isfinite(value));
        for (int z = 0; z < c.grid.ny; ++z) {
            CHECK(state.u[z * (c.grid.nx + 1)] == 0);
            CHECK(state.u[z * (c.grid.nx + 1) + c.grid.nx] == 0);
        }
        for (int x = 0; x < c.grid.nx; ++x) {
            CHECK(state.v[x] == 0);
            CHECK(state.v[c.grid.ny * c.grid.nx + x] == 0);
        }
        const double mean =
            std::accumulate(state.eta.begin(), state.eta.end(), 0.0) / state.eta.size();
        maximum_mean = std::max(maximum_mean, std::abs(mean));
        CHECK(std::abs(mean) <= 1e-6 * std::max(c.depth, .25f));
        for (float eta : state.eta)
            peak = std::max(peak, std::abs(eta));
        if (peak >= .02f * c.depth)
            std::cout << "BOAT SOAK bound exceeded grid=" << c.grid.nx << 'x' << c.grid.ny
                      << " depth=" << c.depth << " damping=" << c.damping
                      << " second=" << second << " peak=" << peak << '\n';
        CHECK(peak < .02f * c.depth);
    }
    std::cout << "BOAT SOAK " << c.grid.nx << 'x' << c.grid.ny << " H = " << c.depth
              << " damping = " << c.damping << " seconds = 120 max_mean = " << maximum_mean
              << " peak = " << peak << '\n';
}
} // namespace
TEST_CASE("wake", prolonged_boat_conserves_volume_and_stays_small) {
    for (float depth : {.25f, 3.f, 6.f})
        for (float damping : {0.f, 2.5f})
            for (auto grid : {water::Grid{64, 48, 20, 12}, water::Grid{48, 64, 12, 20},
                              water::Grid{512, 512, 20, 12}}) {
                auto c = fixture();
                c.grid = grid;
                c.depth = depth;
                c.damping = damping;
                c.impact_radius = .9f;
                c.impact_strength = .001f;
                boat_soak(c);
            }
}

TEST_CASE("wake", sustained_motion_radiates_resolved_trailing_waves) {
    water::Config c;
    water::Simulation s(c);
    water::upload_boat(s, {5, 6, 0, 0, 0, 0});
    const float dt = float(water::stable_dt(c)) * .8f;
    for (float t = 0; t < 3; t += dt)
        s.step(dt, c.damping, {1, 0});
    const auto b = water::download_boat(s);
    const auto state = water::download(s);
    auto height = [&](double l, double t) {
        const int x = int((b.x + l) * c.grid.nx / c.grid.lx);
        const int z = int((b.z + t) * c.grid.ny / c.grid.lz);
        return state.eta[z * c.grid.nx + x];
    };
    // The first visible shoulders are ahead of the hull centre and outside its waterline.
    for (double aft : {.2, .45, .8}) {
        const double l = .3 - aft, t = .7 * aft;
        for (double side : {-1., 1.}) {
            CHECK(height(l, side * t) < -.001);
            CHECK(std::abs(height(l, side * t)) > 3 * std::abs(height(l, 0)));
            CHECK(std::abs(height(l, side * t)) > 3 * std::abs(height(.3 + aft, side * t)));
        }
    }
    double trailing = 0;
    for (int z = 0; z < c.grid.ny; ++z)
        for (int x = 0; x < c.grid.nx; ++x) {
            const float l = (x + .5f) * c.grid.lx / c.grid.nx - b.x;
            if (l < -.9f)
                trailing = std::max(trailing, double(std::abs(state.eta[z * c.grid.nx + x])));
        }
    std::cout << "trailing peak=" << trailing << '\n';
    // One default click measured .568mm here; sustained propulsion must emit a resolved fraction.
    CHECK(trailing > .0001);
}

TEST_CASE("wake", shedding_phase_tracks_travel_and_reset) {
    auto c = fixture();
    water::Simulation whole(c), split(c);
    const water::BoatState start{1, 1.5f, 0, .75f, 0, 0};
    water::upload_boat(whole, start);
    water::upload_boat(split, start);
    whole.step(.001f, c.damping, {1, 0});
    split.step(.0005f, c.damping, {1, 0});
    split.step(.0005f, c.damping, {1, 0});
    const auto a = water::download_boat(whole), b = water::download_boat(split);
    CHECK_NEAR(a.wake_phase, 6.28318530718f * (a.x - start.x) / water::boat_geometry.length, 2e-6);
    CHECK_NEAR(a.wake_phase, b.wake_phase, 2e-5);
    whole.set_depth(.5f);
    CHECK(water::download_boat(whole).wake_phase == a.wake_phase);
    water::upload_boat(whole, {a.x, a.z, a.yaw, 0, 0, 0, a.wake_phase});
    whole.step(.001f, c.damping);
    CHECK(water::download_boat(whole).wake_phase == a.wake_phase);
    whole.reset();
    CHECK(water::download_boat(whole).wake_phase == 0);
    for (float h : water::download(whole).eta) CHECK(h == 0);
    water::upload_boat(whole, {1, 1.5f, 0, .75f, 0, 0, 6.282f});
    whole.step(.001f, c.damping);
    const auto wrapped = water::download_boat(whole);
    CHECK(wrapped.wake_phase >= 0 && wrapped.wake_phase < .02f);
}

TEST_CASE("wake", released_waves_follow_autonomous_dispersion_after_source_stops) {
    auto c = fixture();
    c.grid = {32, 24, 8, 6}; c.impact_radius = .5f;
    for (bool click : {false, true}) {
        water::Simulation simulation(c);
        water::upload_boat(simulation, {3, 3, 0, 1, 0, 0});
        const float dt = .008f;
        for (int step = 0; step < 100; ++step) simulation.step(dt, c.damping);
        const auto boat = water::download_boat(simulation);
        if (click) {
            simulation.reset();
            const water::Impact impact{boat.x, boat.z, c.impact_radius, c.impact_strength};
            simulation.apply_impacts(&impact, 1);
        }
        water::upload_boat(simulation, {boat.x, boat.z, boat.yaw, 0, 0, 0});
        const auto initial = water::download(simulation);
        auto expected = initial;
        for (int step = 0; step < 100; ++step) {
            expected = reference::step(c, expected, dt, 0);
            simulation.step(dt, 0);
        }
        const auto actual = water::download(simulation);
        double error = 0, scale = 0, outside_change = 0;
        for (size_t i = 0; i < actual.eta.size(); ++i) {
            error = std::max(error, std::abs(double(actual.eta[i]) - expected.eta[i]));
            scale = std::max(scale, std::abs(double(expected.eta[i])));
            const double x = (i % c.grid.nx + .5) * c.grid.lx / c.grid.nx;
            const double z = (i / c.grid.nx + .5) * c.grid.lz / c.grid.ny;
            if (std::hypot(x - boat.x, z - boat.z) > 2.2)
                outside_change = std::max(outside_change, std::abs(double(actual.eta[i]) - initial.eta[i]));
        }
        std::cout << "RELEASE click=" << click << " reference_error=" << error << " outside_change=" << outside_change << '\n';
        CHECK(error <= 2e-7 + 3e-5 * scale);
        CHECK(outside_change > 1e-7);
    }
}
