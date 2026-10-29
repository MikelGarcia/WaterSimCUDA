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
    const double width = std::max(double(water::boat_geometry.beam), 2 * cell);
    const double longitudinal = (x - b.x) * std::cos(yaw) + (z - b.z) * std::sin(yaw);
    const double transverse = -(x - b.x) * std::sin(yaw) + (z - b.z) * std::cos(yaw);
    auto bump = [](double q) { return std::abs(q) < 1 ? std::pow(1 - q * q, 3) : 0.0; };
    auto pattern = [&](double l) {
        return bump((l - .25 * length) / (.5 * length)) * bump(transverse / width) -
               .5 * bump((l + .35 * length) / (.75 * length)) * bump(transverse / (1.5 * width));
    };
    const double blend = ramp((vx * std::cos(yaw) + vz * std::sin(yaw) + .01) / .02);
    const double shedding = .03 * (speed * speed / (1.875 * 1.875)) * ramp((speed - .0001) / .0099) *
        std::min(1.0, double(c.depth) / 3.0) * std::sin(b.wake_phase) *
        (blend * bump((longitudinal + .8 * length) / (.5 * length)) +
         (1-blend) * bump((longitudinal - .8 * length) / (.5 * length))) * bump(transverse / (1.5 * width));
    return shedding + .002 * (speed * speed / (1.875 * 1.875)) * ramp((speed - .0001) / .0099) *
           std::min(1.0, double(c.depth) / 1.0) *
           (blend * pattern(longitudinal) + (1 - blend) * pattern(-longitudinal));
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
    std::vector<double> p(g.nx * g.ny), u(got.u.size()), v(got.v.size());
    for (int z = 0; z < g.ny; ++z)
        for (int x = 0; x < g.nx; ++x)
            p[z * g.nx + x] = head((x + .5) * dx, (z + .5) * dz, b, c);
    for (int z = 0; z < g.ny; ++z)
        for (int x = 1; x < g.nx; ++x)
            u[z * (g.nx + 1) + x] =
                -c.gravity * dt * (p[z * g.nx + x] - p[z * g.nx + x - 1]) / dx / (1 + damping * dt);
    for (int z = 1; z < g.ny; ++z)
        for (int x = 0; x < g.nx; ++x)
            v[z * g.nx + x] = -c.gravity * dt * (p[z * g.nx + x] - p[(z - 1) * g.nx + x]) / dz /
                              (1 + damping * dt);
    auto close = [](double actual, double expected) {
        CHECK(std::abs(actual - expected) <= 2e-6 + 2e-5 * std::abs(expected));
    };
    for (size_t i = 0; i < u.size(); ++i)
        close(got.u[i], u[i]);
    for (size_t i = 0; i < v.size(); ++i)
        close(got.v[i], v[i]);
    for (int z = 0; z < g.ny; ++z)
        for (int x = 0; x < g.nx; ++x) {
            double eta = -c.depth * dt *
                         ((u[z * (g.nx + 1) + x + 1] - u[z * (g.nx + 1) + x]) / dx +
                          (v[(z + 1) * g.nx + x] - v[z * g.nx + x]) / dz);
            close(got.eta[z * g.nx + x], eta);
        }
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
                                {&a.u, &b.u, &sum.u}, {&a.v, &b.v, &sum.v}})
                for (size_t i = 0; i < fields[0]->size(); ++i)
                    CHECK_NEAR((*fields[2])[i], (*fields[0])[i] + (*fields[1])[i], 1e-8f);
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

TEST_CASE("wake", released_waves_travel_beyond_the_complete_forcing_footprint) {
    water::Config c;
    for (bool click : {false, true}) {
        water::Simulation simulation(c);
        water::upload_boat(simulation, {5, 6, 0, 0, 0, 0});
        const float dt = float(water::stable_dt(c)) * .8f;
        for (float t = 0; t < 3; t += dt)
            simulation.step(dt, c.damping, {1, 0});
        const auto boat = water::download_boat(simulation);
        const double stern_x = boat.x - .8 * water::boat_geometry.length;
        if (click) {
            simulation.reset();
            const water::Impact impact{float(stern_x), boat.z, c.impact_radius, c.impact_strength};
            simulation.apply_impacts(&impact, 1);
        }
        water::upload_boat(simulation, {boat.x, boat.z, boat.yaw, 0, 0, 0});
        float elapsed = 0;
        double initial_radius = 0;
        for (float target : {0.f, .2f, .4f, .6f, .8f}) {
            while (elapsed < target) {
                simulation.step(dt, c.damping);
                elapsed += dt;
            }
            const auto state = water::download(simulation);
            double far_peak = 0, mass = 0, weighted = 0, total = 0;
            for (int z = 0; z < c.grid.ny; ++z) {
                for (int x = 0; x < c.grid.nx; ++x) {
                    const double radius = std::hypot((x + .5) * c.grid.lx / c.grid.nx - stern_x,
                                                     (z + .5) * c.grid.lz / c.grid.ny - boat.z);
                    const double height = state.eta[z * c.grid.nx + x];
                    // One metre excludes both pressure lobes, including their transverse support.
                    if (radius > 1)
                        far_peak = std::max(far_peak, std::abs(height));
                    weighted += radius * radius * height * height;
                    total += height * height;
                    mass += height;
                }
            }
            const double radius = std::sqrt(weighted / total);
            CHECK(std::abs(mass / state.eta.size()) < 1e-9);
            if (target == 0) {
                initial_radius = radius;
            } else {
                CHECK(far_peak > .0001);
                const double travel = std::sqrt(c.gravity * c.depth) * elapsed;
                CHECK(radius - initial_radius > .65 * travel);
                CHECK(radius - initial_radius < 1.2 * travel);
            }
            std::cout << "RELEASE click=" << click << " time=" << elapsed
                      << " outside peak=" << far_peak << " rms radius=" << radius << '\n';
        }
    }
}
