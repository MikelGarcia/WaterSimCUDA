#include "validation.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <limits>

namespace {
constexpr float pi = 3.14159265358979323846f;
water::Config fixture() {
    water::Config c;
    c.grid = {16, 16, 40, 30};
    c.impact_radius = 5;
    return c;
}
float speed(const water::BoatState& b) {
    return std::hypot(b.vx, b.vz);
}
float longitudinal(const water::BoatState& b) {
    return b.vx * std::cos(b.yaw) + b.vz * std::sin(b.yaw);
}
void advance(water::Simulation& s, int count, water::BoatControl control = {},
    float dt = 1.0f / 120, float damping = 0) {
    for (int i = 0; i < count; ++i)
        s.step(dt, damping, control);
}
void inside(const water::BoatState& b, const water::Grid& g) {
    CHECK(std::isfinite(b.x) && std::isfinite(b.z) && std::isfinite(b.yaw));
    CHECK(std::isfinite(b.yaw_rate));
    CHECK(speed(b) <= 1.87501f);
    for (int i : {-1, 1})
        for (int j : {-1, 1}) {
            const double x = b.x + i * 0.5 * water::boat_geometry.length * std::cos(double(b.yaw)) -
                             j * 0.5 * water::boat_geometry.beam * std::sin(double(b.yaw));
            const double z = b.z + i * 0.5 * water::boat_geometry.length * std::sin(double(b.yaw)) +
                             j * 0.5 * water::boat_geometry.beam * std::cos(double(b.yaw));
            CHECK(x >= -2e-6 && x <= g.lx + 2e-6);
            CHECK(z >= -2e-6 && z <= g.lz + 2e-6);
        }
}
} // namespace
TEST_CASE("boat", requested_hull_dimensions) {
    CHECK_NEAR(water::boat_geometry.length, .6f, 1e-7f);
    CHECK_NEAR(water::boat_geometry.beam, .227371384f, 1e-7f);
}
TEST_CASE("boat", propulsion_braking_and_coasting) {
    water::Simulation s(fixture());
    advance(s, 1200, {1, 0});
    auto b = water::download_boat(s);
    CHECK_NEAR(longitudinal(b), 1.875f, 1e-5f);
    CHECK(speed(b) <= 1.87501f);
    water::upload_boat(s, {20, 15, 0, 1, 0, 0});
    s.step(1.0f / 120, 0, {-1, 0});
    CHECK(longitudinal(water::download_boat(s)) >= 0);
    advance(s, 200, {-1, 0});
    b = water::download_boat(s);
    CHECK(longitudinal(b) < 0);
    CHECK(longitudinal(b) >= -0.93751f);
    advance(s, 400, {-1, 0});
    b = water::download_boat(s);
    CHECK_NEAR(longitudinal(b), -.9375f, 1e-5f);
    CHECK(speed(b) <= 0.93751f);
    s.step(1.0f / 120, 0, {1, 0});
    CHECK(longitudinal(water::download_boat(s)) < 0);
    advance(s, 200, {1, 0});
    CHECK(longitudinal(water::download_boat(s)) > 0);
    water::upload_boat(s, {20, 15, 0, 1, 0.2f, 0});
    float old = speed(water::download_boat(s));
    for (int i = 0; i < 120; ++i) {
        s.step(1.0f / 120, 0);
        b = water::download_boat(s);
        CHECK(speed(b) <= old);
        old = speed(b);
    }
    CHECK(b.x > 20.1f);
    CHECK(old > 0);
}
TEST_CASE("boat", rudder_rest_reverse_and_high_speed) {
    water::Simulation s(fixture());
    auto rest = water::download_boat(s);
    for (float rudder : {-1.0f, 0.0f, 1.0f}) {
        advance(s, 120, {0, rudder});
        const auto b = water::download_boat(s);
        CHECK(b.x == rest.x && b.z == rest.z && b.yaw == rest.yaw && b.yaw_rate == 0);
    }
    water::BoatState b;
    for (float sign : {-1.0f, 1.0f}) {
        water::upload_boat(s, {20, 15, 0, sign * .5f, 0, 0});
        advance(s, 12, {0, 1});
        b = water::download_boat(s);
        CHECK(sign * b.yaw > 0);
    }
    float rates[2];
    int n = 0;
    for (float v : {.9375f, 1.875f}) {
        water::upload_boat(s, {20, 15, 0, v, 0, 0});
        for (int i = 0; i < 120; ++i) {
            const auto current = water::download_boat(s);
            water::upload_boat(s, {20, 15, current.yaw, v * std::cos(current.yaw),
                                      v * std::sin(current.yaw), current.yaw_rate});
            s.step(1.0f / 120, 0, {0, 1});
        }
        rates[n++] = water::download_boat(s).yaw_rate;
    }
    CHECK(rates[1] < rates[0]);
}
TEST_CASE("boat", damping_independence_and_subdivision) {
    auto c = fixture();
    water::Simulation a(c), b(c);
    for (auto control : {water::BoatControl{1, .6f}, {0, -.4f}, {-1, -.5f}, {1, .3f}}) {
        advance(a, 30, control);
        advance(b, 60, control, 1.0f / 240, 10);
    }
    auto x = water::download_boat(a), y = water::download_boat(b);
    CHECK(std::hypot(x.x - y.x, x.z - y.z) < .003f);
    CHECK(std::hypot(x.vx - y.vx, x.vz - y.vz) < .003f);
    CHECK(std::abs(x.yaw - y.yaw) < .005f);
    water::upload_boat(a, {20, 15, .3f, .4f, .2f, .1f});
    water::upload_boat(b, {20, 15, .3f, .4f, .2f, .1f});
    advance(a, 120);
    advance(b, 120, {}, 1.0f / 120, 10);
    x = water::download_boat(a);
    y = water::download_boat(b);
    CHECK(x.x == y.x && x.vx == y.vx && x.vz == y.vz && x.yaw == y.yaw);
}
TEST_CASE("boat", walls_corners_and_contact_energy) {
    auto c = fixture();
    water::Simulation s(c);
    const std::array<water::BoatState, 9> states{
        {{.34f, 15, pi, -1.5f, 0, 0}, {39.66f, 15, 0, 1.5f, 0, 0}, {20, .34f, -pi / 2, 0, -1.5f, 0},
            {20, 29.66f, pi / 2, 0, 1.5f, 0}, {39.7f, 29.7f, pi / 4, 1.06f, 1.06f, .5f},
            {20, .114f, 0, .7f, -.1f, -1.5f}, {.114f, 15, pi / 2, -.1f, .7f, 1.5f},
            {39.886f, 15, pi / 2, .1f, .7f, 1.5f}, {20, 29.886f, 0, .7f, .1f, 1.5f}}};
    for (auto state : states) {
        water::upload_boat(s, state);
        float old = speed(state);
        bool lost = false;
        for (int i = 0; i < 160; ++i) {
            s.step(1.0f / 120, 0);
            auto b = water::download_boat(s);
            inside(b, c.grid);
            CHECK(speed(b) <= old + 1e-6f);
            if (i == 4)
                CHECK(speed(b) < speed(state) * .3f);
            if (speed(b) < speed(state) * .3f)
                lost = true;
            old = speed(b);
        }
        CHECK(lost);
    }
}
TEST_CASE("boat", contact_escape_and_blocked_yaw_recovery) {
    auto c = fixture();
    water::Simulation s(c);
    const float right_start = 39.7f;
    water::upload_boat(s, {right_start, 15, 0, 0, 0, 0});
    advance(s, 120, {-1, 0});
    auto b = water::download_boat(s);
    CHECK(right_start - b.x > .15f);
    inside(b, c.grid);
    const float left_start = .30f;
    water::upload_boat(s, {left_start, 15, 0, 0, 0, 0});
    advance(s, 120, {1, 0});
    b = water::download_boat(s);
    CHECK(b.x - left_start > .15f);
    inside(b, c.grid);
    const float bottom_start = .114f;
    water::upload_boat(s, {20, bottom_start, 0, .5f, 0, -1});
    advance(s, 60, {1, -1});
    inside(water::download_boat(s), c.grid);
    advance(s, 600, {1, 1});
    b = water::download_boat(s);
    inside(b, c.grid);
    CHECK(b.z - bottom_start > .043f);
    CHECK(b.yaw > 0);
    water::upload_boat(s, {20, water::boat_geometry.beam / 2, 0, .5f, 0, 0});
    advance(s, 120, {1, 0});
    b = water::download_boat(s);
    CHECK(b.x > 20.5f);
    inside(b, c.grid);
    water::upload_boat(s, {39.7f, 15, 0, 0, 0, 0});
    for (int i = 0; i < 300; ++i) {
        s.step(1.0f / 120, 0, {1, 1});
        inside(water::download_boat(s), c.grid);
    }
}
TEST_CASE("boat", translating_impact_after_angular_contact_loses_energy) {
    auto c = fixture();
    water::Simulation s(c);
    water::upload_boat(s, {20, water::boat_geometry.beam / 2 + .001f, 0, .7f, -.18f, -2});
    s.step(1.0f / 120, 0);
    const auto b = water::download_boat(s);
    inside(b, c.grid);
    CHECK(speed(b) < .3f);
}
TEST_CASE("boat", fit_reset_and_invalid_inputs) {
    for (bool transpose : {false, true}) {
        auto c = fixture();
        c.grid.lx = transpose ? .26f : .50f;
        c.grid.lz = transpose ? .50f : .26f;
        c.grid.nx = 64;
        c.grid.ny = 64;
        c.impact_radius = .04f;
        bool caught = false;
        try {
            water::Simulation s(c);
        } catch (const std::invalid_argument& e) {
            std::string text = e.what();
            caught = text.find("boat footprint") != std::string::npos &&
                     text.find("required") != std::string::npos;
        }
        CHECK(caught);
        c.grid.lx = transpose ? .28f : .68f;
        c.grid.lz = transpose ? .68f : .28f;
        water::Simulation s(c);
        auto b = water::download_boat(s);
        CHECK_NEAR(b.yaw, transpose ? pi / 2 : 0, 1e-7f);
        inside(b, c.grid);
        water::upload_boat(s, {b.x, b.z, b.yaw, .5f, .2f, 1});
        s.reset();
        b = water::download_boat(s);
        CHECK(b.vx == 0 && b.vz == 0 && b.yaw_rate == 0);
        CHECK(b.x == c.grid.lx / 2 && b.z == c.grid.lz / 2);
        CHECK_NEAR(b.yaw, transpose ? pi / 2 : 0, 1e-7f);
    }
    water::Simulation s(fixture());
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS(s.step(nan, 0), std::invalid_argument);
    CHECK_THROWS(s.step(0, 0), std::invalid_argument);
    CHECK_THROWS(s.step(10, 0), std::invalid_argument);
    CHECK_THROWS(s.step(1.0f / 120, -1), std::invalid_argument);
    for (auto control : {water::BoatControl{nan, 0}, {0, nan}, {1.01f, 0}, {0, -1.01f}})
        CHECK_THROWS(s.step(1.0f / 120, 0, control), std::invalid_argument);
}
TEST_CASE("boat", matrix_orientation_translation_and_size) {
    water::Simulation s(fixture());
    float* device = nullptr;
    CHECK(cudaMalloc(&device, 16 * sizeof(float)) == cudaSuccess);
    CHECK_THROWS(s.write_boat_transform(device, 15), std::invalid_argument);
    CHECK_THROWS(s.write_boat_transform(nullptr, 16), std::invalid_argument);
    for (float yaw : {0.0f, pi / 2}) {
        water::upload_boat(s, {21, 13, yaw, 0, 0, 0});
        s.write_boat_transform(device, 16);
        float m[16];
        CHECK(cudaMemcpyAsync(m, device, sizeof(m), cudaMemcpyDeviceToHost, s.stream()) ==
              cudaSuccess);
        CHECK(cudaStreamSynchronize(s.stream()) == cudaSuccess);
        CHECK_NEAR(m[0], std::cos(yaw), 1e-6f);
        CHECK_NEAR(m[2], std::sin(yaw), 1e-6f);
        CHECK_NEAR(m[8], -std::sin(yaw), 1e-6f);
        CHECK_NEAR(m[10], std::cos(yaw), 1e-6f);
        CHECK(m[12] == 1 && m[13] == 0 && m[14] == -2 && m[15] == 1 && m[5] == 1);
        for (int i : {1, 3, 4, 6, 7, 9, 11})
            CHECK(m[i] == 0);
    }
    s.reset();
    s.write_boat_transform(device, 16);
    float m[16];
    CHECK(cudaMemcpyAsync(m, device, sizeof(m), cudaMemcpyDeviceToHost, s.stream()) == cudaSuccess);
    CHECK(cudaStreamSynchronize(s.stream()) == cudaSuccess);
    CHECK(m[12] == 0 && m[14] == 0);
    CHECK(cudaFree(device) == cudaSuccess);
}
TEST_CASE("boat", oversized_diagnostics_are_bounded) {
    auto c = fixture();
    c.grid = {8, 8, 1000, 1000};
    c.impact_radius = 250;
    water::Simulation s(c);
    water::upload_boat(s, {500, 500, 0, 10000, 10000, 10000});
    s.step(1, 0);
    auto b = water::download_boat(s);
    inside(b, c.grid);
    CHECK(std::hypot(b.x - 500, b.z - 500) < .005f);
    CHECK(std::abs(b.yaw) < .005f);
}
