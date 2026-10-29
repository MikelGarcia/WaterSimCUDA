#include "dispersive_continuity.hpp"
#include "dispersion_reference.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <cctype>
#include <limits>
#include <windows.h>
#include <tlhelp32.h>

namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

struct DeviceField {
    float* p = nullptr;
    explicit DeviceField(std::size_t n) { checked(cudaMalloc(&p, n * sizeof(float))); }
    ~DeviceField() {
        if (cudaFree(p) != cudaSuccess)
            std::terminate();
    }
};

void report_modules() {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    CHECK(snapshot != INVALID_HANDLE_VALUE);
    MODULEENTRY32 module{};
    module.dwSize = sizeof(module);
    CHECK(Module32First(snapshot, &module));
    do {
        std::string name = module.szModule;
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char ch) { return char(std::tolower(ch)); });
        if (name.find("cufft") != std::string::npos || name.find("nvjit") != std::string::npos ||
            name.find("nvrtc") != std::string::npos || name.find("cudart") != std::string::npos)
            std::cout << "FFT_MODULE " << module.szExePath << '\n';
    } while (Module32Next(snapshot, &module));
    CHECK(CloseHandle(snapshot));
}

void compare(water::Grid grid, float depth, int mode, int repeats = 1) {
    water::Config config;
    config.grid = grid;
    config.depth = depth;
    const int nx = grid.nx, ny = grid.ny;
    constexpr double pi = 3.14159265358979323846;
    std::vector<float> u((nx + 1) * ny), v(nx * (ny + 1)), eta(nx * ny);
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double d = 0.5;
            if (mode == 1)
                d = 0.37 * std::cos(pi * (x + 0.5) / nx) - 0.29 * std::cos(3 * pi * (y + 0.5) / ny) +
                    0.41 * std::cos(2 * pi * (x + 0.5) / nx) * std::cos(5 * pi * (y + 0.5) / ny) +
                    0.13 * std::sin(1.7 * (x + 3 * y));
            if (mode == 2)
                d = std::cos(pi * (nx - 1) * (x + 0.5) / nx) * std::cos(pi * (ny - 1) * (y + 0.5) / ny);
            if (mode == 3)
                d = std::cos(pi * (nx - 1) * (x + 0.5) / nx);
            if (mode == 4)
                d = std::cos(pi * (ny - 1) * (y + 0.5) / ny);
            u[y * (nx + 1) + x + 1] = u[y * (nx + 1) + x] + float(d * grid.lx / nx);
            eta[y * nx + x] = float(0.07 * std::sin(x + 2.1 * y));
        }
    if (mode == 1)
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = float(0.07 * std::sin(1.3 * i));
    // Use rounded device inputs in the independent divergence calculation.
    std::vector<double> divergence(nx * ny);
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x)
            divergence[y * nx + x] =
                (double(u[y * (nx + 1) + x + 1]) - u[y * (nx + 1) + x]) / (double(grid.lx) / nx) +
                (double(v[(y + 1) * nx + x]) - v[y * nx + x]) / (double(grid.lz) / ny);
    const auto response = reference::apply_depth(config, divergence);
    DeviceField du(u.size()), dv(v.size()), de(eta.size());
    cudaStream_t stream;
    checked(cudaStreamCreate(&stream));
    {
        water::DispersiveContinuity continuity(config, stream);
        report_modules();
        checked(cudaMemcpyAsync(du.p, u.data(), u.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        checked(cudaMemcpyAsync(dv.p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        checked(
            cudaMemcpyAsync(de.p, eta.data(), eta.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        constexpr float dt = 0.125f;
        std::vector<float> actual(eta.size());
        for (int repeat = 1; repeat <= repeats; ++repeat) {
            continuity.advance(de.p, du.p, dv.p, dt);
            checked(cudaMemcpyAsync(actual.data(), de.p, actual.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
            checked(cudaStreamSynchronize(stream));
            double scale = 0, error = 0, mean = 0;
            for (std::size_t i = 0; i < eta.size(); ++i)
                scale = std::max(scale, std::abs(eta[i] - repeat * dt * response[i]));
            for (std::size_t i = 0; i < eta.size(); ++i) {
                error = std::max(error, std::abs(actual[i] - (eta[i] - repeat * dt * response[i])));
                mean += actual[i] - eta[i];
            }
            std::cout << "FFT " << nx << 'x' << ny << " H=" << depth << " error=" << error << '\n';
            CHECK(error <= 2e-6 * scale + 2e-8);
            CHECK(std::abs(mean) / eta.size() < 2e-8);
        }
        std::vector<float> unchanged(u.size());
        checked(cudaMemcpyAsync(unchanged.data(), du.p, u.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                stream));
        checked(cudaStreamSynchronize(stream));
        CHECK(unchanged == u);
        unchanged.resize(v.size());
        checked(cudaMemcpyAsync(unchanged.data(), dv.p, v.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                stream));
        checked(cudaStreamSynchronize(stream));
        CHECK(unchanged == v);
    }
    checked(cudaStreamDestroy(stream));
}
} // namespace

TEST_CASE("fft", constant_divergence_has_zero_response) {
    compare({8, 12, 3, 11}, 3, 0);
}
TEST_CASE("fft", mixed_fields_match_independent_cosines) {
    for (float h : {0.25f, 3.0f, 6.0f})
        compare({8, 12, 3, 11}, h, 1);
}
TEST_CASE("fft", highest_modes_and_transposes_preserve_normalization) {
    for (float h : {0.25f, 3.0f, 6.0f}) {
        compare({8, 12, 3, 11}, h, 2);
        compare({12, 8, 11, 3}, h, 2);
        compare({8, 12, 3, 11}, h, 3);
        compare({12, 8, 11, 3}, h, 4);
        compare({8, 32, 2, 19}, h, 1);
        compare({32, 8, 19, 2}, h, 1);
    }
}
TEST_CASE("fft", repeated_inverse_does_not_reuse_destroyed_spectrum) {
    compare({12, 8, 11, 3}, 3, 1, 2);
}

namespace water {
struct ContinuityObservation {
    static std::size_t workspace(const DispersiveContinuity& op) { return op.workspace_bytes_; }
    static cufftHandle forward(const DispersiveContinuity& op) { return op.forward_; }
    static const float* real(const DispersiveContinuity& op) { return op.real_; }
    static float depth(const DispersiveContinuity& op) { return op.depth_; }
};
} // namespace water

TEST_CASE("fft", independent_streams_and_depth_refresh_reuse_resources) {
    water::Config config;
    config.grid = {8, 12, 3, 11};
    cudaStream_t a, b;
    checked(cudaStreamCreate(&a));
    checked(cudaStreamCreate(&b));
    {
        water::DispersiveContinuity first(config, a), second(config, b);
        const auto plan = water::ContinuityObservation::forward(first);
        const auto real = water::ContinuityObservation::real(first);
        const auto workspace = water::ContinuityObservation::workspace(first);
        CHECK(real != water::ContinuityObservation::real(second));
        DeviceField u(9 * 12), v(8 * 13), eta_a(8 * 12), eta_b(8 * 12);
        std::vector<float> host_u(9 * 12), zeros(8 * 13);
        for (int y = 0; y < 12; ++y)
            for (int x = 0; x <= 8; ++x)
                host_u[y * 9 + x] = float(0.2 * std::sin(0.6 * x + 0.3 * y));
        checked(cudaMemcpy(u.p, host_u.data(), host_u.size() * sizeof(float), cudaMemcpyHostToDevice));
        checked(cudaMemcpy(v.p, zeros.data(), zeros.size() * sizeof(float), cudaMemcpyHostToDevice));
        checked(cudaMemsetAsync(eta_a.p, 0, 8 * 12 * sizeof(float), a));
        checked(cudaMemsetAsync(eta_b.p, 0, 8 * 12 * sizeof(float), b));
        first.set_depth(0.25f);
        CHECK_THROWS(first.set_depth(std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);
        CHECK_THROWS(first.set_depth(0), std::invalid_argument);
        CHECK(water::ContinuityObservation::depth(first) == 0.25f);
        first.advance(eta_a.p, u.p, v.p, 0.125f);
        second.advance(eta_b.p, u.p, v.p, 0.125f);
        checked(cudaStreamSynchronize(a));
        checked(cudaStreamSynchronize(b));
        std::vector<float> aa(96), bb(96);
        checked(cudaMemcpy(aa.data(), eta_a.p, 96 * sizeof(float), cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(bb.data(), eta_b.p, 96 * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<double> div(96);
        for (int y = 0; y < 12; ++y)
            for (int x = 0; x < 8; ++x)
                div[y * 8 + x] = (double(host_u[y * 9 + x + 1]) - host_u[y * 9 + x]) * 8 / 3;
        auto reference_a = config;
        reference_a.depth = 0.25f;
        const auto expected_a = reference::apply_depth(reference_a, div);
        const auto expected_b = reference::apply_depth(config, div);
        for (int i = 0; i < 96; ++i) {
            CHECK_NEAR(aa[i], -0.125 * expected_a[i], 2e-7);
            CHECK_NEAR(bb[i], -0.125 * expected_b[i], 2e-7);
        }
        first.set_depth(6);
        first.set_depth(3);
        CHECK(water::ContinuityObservation::forward(first) == plan);
        CHECK(water::ContinuityObservation::real(first) == real);
        CHECK(water::ContinuityObservation::workspace(first) == workspace);
    }
    checked(cudaStreamDestroy(a));
    checked(cudaStreamDestroy(b));
}

TEST_CASE("fft", repeated_construction_recovers_device_memory) {
    water::Config config;
    config.grid = {32, 24, 7, 11};
    cudaStream_t stream;
    checked(cudaStreamCreate(&stream));
    {
        water::DispersiveContinuity warmup(config, stream);
    }
    std::size_t before, total, after;
    checked(cudaMemGetInfo(&before, &total));
    for (int i = 0; i < 12; ++i) {
        water::DispersiveContinuity op(config, stream);
        op.set_depth(0.25f);
    }
    checked(cudaMemGetInfo(&after, &total));
    std::cout << "FFT_MEMORY before=" << before << " after=" << after << '\n';
    CHECK(after + 2 * 1024 * 1024 >= before);
    checked(cudaStreamDestroy(stream));
}

TEST_CASE("fft", workspace_sizes_are_measured) {
    cudaStream_t stream;
    checked(cudaStreamCreate(&stream));
    for (int size : {512, 2048}) {
        water::Config config;
        config.grid.nx = size;
        config.grid.ny = size;
        std::size_t before, total, after;
        checked(cudaMemGetInfo(&before, &total));
        {
            water::DispersiveContinuity op(config, stream);
            checked(cudaStreamSynchronize(stream));
            checked(cudaMemGetInfo(&after, &total));
            std::cout << "FFT_RESOURCES " << size
                      << " workspace=" << water::ContinuityObservation::workspace(op)
                      << " device_delta=" << (before - after) << '\n';
        }
    }
    checked(cudaStreamDestroy(stream));
}

TEST_CASE("fft", large_prime_plans_execute_with_deployed_runtime) {
    water::Config config;
    config.grid = {131, 137, 23, 17};
    constexpr double pi = 3.14159265358979323846;
    const int nx = config.grid.nx, ny = config.grid.ny, m = 17;
    std::vector<float> u((nx + 1) * ny), v(nx * (ny + 1)), eta(nx * ny);
    const double dx = double(config.grid.lx) / nx;
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x <= nx; ++x)
            u[y * (nx + 1) + x] = float(dx * std::sin(pi * m * x / nx) / (2 * std::sin(pi * m / (2 * nx))));
    DeviceField du(u.size()), dv(v.size()), de(eta.size());
    checked(cudaMemcpy(du.p, u.data(), u.size() * sizeof(float), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(dv.p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    checked(cudaMemset(de.p, 0, eta.size() * sizeof(float)));
    cudaStream_t stream;
    checked(cudaStreamCreate(&stream));
    {
        water::DispersiveContinuity op(config, stream);
        report_modules();
        op.advance(de.p, du.p, dv.p, 0.125f);
        op.advance(de.p, du.p, dv.p, 0.125f);
        checked(
            cudaMemcpyAsync(eta.data(), de.p, eta.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checked(cudaStreamSynchronize(stream));
        const double k = pi * m / config.grid.lx;
        const double eigenvalue = std::pow(2 * std::sin(pi * m / (2 * nx)) / dx, 2);
        const double amplitude = -0.25 * k * std::tanh(k * config.depth) / eigenvalue;
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                CHECK_NEAR(eta[y * nx + x], amplitude * std::cos(pi * m * (x + 0.5) / nx), 2e-7);
    }
    checked(cudaStreamDestroy(stream));
}
TEST_CASE("dispersion", complete_staggered_steps_match_cosine_reference) {
    constexpr double pi = 3.14159265358979323846;
    for (water::Grid grid : {water::Grid{8, 12, 4, 6}, water::Grid{12, 8, 6, 4}})
        for (float depth : {.25f, 3.f, 6.f})
            for (float damping : {0.f, 2.5f}) {
                water::Config config;
                config.grid = grid;
                config.depth = depth;
                config.impact_radius = 1;
                config.impact_strength = .001f;
                water::Simulation simulation(config);
                auto expected = water::download(simulation);
                for (int y = 0; y < grid.ny; ++y)
                    for (int x = 0; x < grid.nx; ++x)
                        expected.eta[y * grid.nx + x] = float(.001 + .003 * std::cos(pi * (x + .5) / grid.nx) +
                            .002 * std::cos(pi * (grid.nx - 1) * (x + .5) / grid.nx) *
                            std::cos(pi * (grid.ny - 1) * (y + .5) / grid.ny));
                for (size_t i = 0; i < expected.u.size(); ++i) expected.u[i] = float(.002 * std::sin(.7 * i));
                for (size_t i = 0; i < expected.v.size(); ++i) expected.v[i] = float(.003 * std::cos(.9 * i));
                water::upload(simulation, expected);
                const float dt = .003f;
                double error = 0, scale = 0;
                for (int step = 0; step < 6; ++step) {
                    expected = reference::step(config, expected, dt, damping);
                    simulation.step(dt, damping);
                    const auto actual = water::download(simulation);
                    for (int y = 0; y < grid.ny; ++y) {
                        CHECK(actual.u[y * (grid.nx + 1)] == 0);
                        CHECK(actual.u[y * (grid.nx + 1) + grid.nx] == 0);
                    }
                    for (int x = 0; x < grid.nx; ++x) {
                        CHECK(actual.v[x] == 0);
                        CHECK(actual.v[grid.ny * grid.nx + x] == 0);
                    }
                    const std::vector<float>* want[] = {&expected.eta, &expected.u, &expected.v};
                    const std::vector<float>* got[] = {&actual.eta, &actual.u, &actual.v};
                    for (int a = 0; a < 3; ++a)
                        for (size_t i = 0; i < got[a]->size(); ++i) {
                            error = std::max(error, std::abs(double((*got[a])[i]) - (*want[a])[i]));
                            scale = std::max(scale, std::abs(double((*want[a])[i])));
                        }
                }
                std::cout << "COMPLETE " << grid.nx << 'x' << grid.ny << " H=" << depth
                          << " damping=" << damping << " error=" << error << " scale=" << scale << '\n';
                CHECK(error <= 2e-7 + 3e-5 * scale);
            }
}
namespace {
double modal_period_error(water::Config c, int m, int n, int refinement) {
    constexpr double pi = 3.14159265358979323846;
    water::Simulation sim(c);
    auto state = water::download(sim);
    for (int y = 0; y < c.grid.ny; ++y)
        for (int x = 0; x < c.grid.nx; ++x)
            state.eta[y * c.grid.nx + x] = float(.001 * std::cos(pi * m * (x + .5) / c.grid.nx) *
                std::cos(pi * n * (y + .5) / c.grid.ny));
    water::upload(sim, state);
    const double k = std::hypot(pi * m / c.grid.lx, pi * n / c.grid.lz);
    const double omega = std::sqrt(c.gravity * k * std::tanh(k * c.depth));
    const double period = 2 * pi / omega;
    const float dt = float((1.0 / 120) / water::substeps_per_quantum(c) / refinement);
    double previous = .001, crossings[2]{};
    int count = 0;
    for (int step = 1; step * dt < 2.5 * period && count < 2; ++step) {
        sim.step(dt, 0);
        state = water::download(sim);
        double projection = 0;
        for (int y = 0; y < c.grid.ny; ++y)
            for (int x = 0; x < c.grid.nx; ++x)
                projection += state.eta[y * c.grid.nx + x] * std::cos(pi * m * (x + .5) / c.grid.nx) *
                    std::cos(pi * n * (y + .5) / c.grid.ny);
        projection *= (m ? 2.0 : 1) * (n ? 2.0 : 1) / (c.grid.nx * c.grid.ny);
        if (previous > 0 && projection <= 0)
            crossings[count++] = (step - 1 + previous / (previous - projection)) * dt;
        previous = projection;
    }
    CHECK(count == 2);
    const double error = std::abs((crossings[1] - crossings[0]) / period - 1);
    std::cout << "MODE " << m << ',' << n << " omega=" << omega << " dt=" << dt << " period_error=" << error << '\n';
    return error;
}
}
TEST_CASE("dispersion", modal_periods_refine_toward_finite_depth_dispersion) {
    water::Config c;
    c.grid = {32, 24, 4, 3};
    c.impact_radius = .25f;
    for (auto mn : {std::pair<int,int>{1,0}, {0,2}, {3,4}, {31,23}}) {
        const double coarse = modal_period_error(c, mn.first, mn.second, 1);
        const double fine = modal_period_error(c, mn.first, mn.second, 2);
        CHECK(coarse < .03);
        CHECK(fine < coarse * .5);
    }
}
TEST_CASE("dispersion", genuinely_long_modes_recover_shallow_water_limit) {
    water::Config c;
    c.grid = {16, 8, 400, 200};
    c.depth = .25f;
    c.impact_radius = 50;
    c.impact_strength = .001f;
    constexpr double pi = 3.14159265358979323846;
    const double k = pi / c.grid.lx;
    CHECK(k * c.depth < .002);
    CHECK_NEAR(std::sqrt(c.gravity * k * std::tanh(k * c.depth)) /
               (k * std::sqrt(c.gravity * c.depth)), 1, 1e-6);
    // Increase gravity to observe the same small-kH mode in a short deterministic run.
    c.gravity = 98100;
    CHECK(modal_period_error(c, 1, 0, 1) < .03);
}
TEST_CASE("dispersion", depth_refresh_preserves_state_and_reuses_fft_resources) {
    water::Config c;
    c.grid = {8, 12, 4, 6};
    c.impact_radius = 1;
    c.impact_strength = .001f;
    water::Simulation sim(c), other(c);
    auto state = water::download(sim);
    for (size_t i = 0; i < state.eta.size(); ++i) state.eta[i] = float(.001 * std::cos(.3 * i));
    for (size_t i = 0; i < state.u.size(); ++i) state.u[i] = float(.002 * std::sin(.7 * i));
    water::upload(sim, state);
    const auto boat = water::download_boat(sim);
    const auto resources = water::fft_resources(sim), separate = water::fft_resources(other);
    CHECK(resources.forward_plan != separate.forward_plan);
    CHECK(resources.real_bytes == 4 * 8 * 12 * sizeof(float));
    CHECK(resources.spectral_bytes == 2 * 12 * 9 * 2 * sizeof(float));
    CHECK(resources.weights_bytes == 2 * 2 * 12 * 9 * sizeof(float));
    for (float h : {.25f, 6.f, 3.f, std::numeric_limits<float>::max()}) {
        sim.set_depth(h);
        const auto current = water::download(sim);
        CHECK(current.eta == state.eta && current.u == state.u && current.v == state.v);
        const auto current_boat = water::download_boat(sim);
        CHECK(std::memcmp(&boat, &current_boat, sizeof(boat)) == 0);
        const auto now = water::fft_resources(sim);
        CHECK(now.forward_plan == resources.forward_plan && now.inverse_plan == resources.inverse_plan);
        CHECK(now.workspace == resources.workspace && now.workspace_bytes == resources.workspace_bytes);
    }
    for (float h : {0.f, -1.f, std::numeric_limits<float>::quiet_NaN()})
        CHECK_THROWS(sim.set_depth(h), std::invalid_argument);
    sim.set_depth(.25f);
    c = sim.config();
    const auto expected = reference::step(c, state, .003f, 2.5f);
    sim.step(.003f, 2.5f);
    const auto actual = water::download(sim);
    for (size_t i = 0; i < actual.eta.size(); ++i) CHECK_NEAR(actual.eta[i], expected.eta[i], 2e-7);
    for (size_t i = 0; i < actual.u.size(); ++i) CHECK_NEAR(actual.u[i], expected.u[i], 2e-7);
    for (size_t i = 0; i < actual.v.size(); ++i) CHECK_NEAR(actual.v[i], expected.v[i], 2e-7);
    sim.reset();
    CHECK(water::fft_resources(sim).forward_plan == resources.forward_plan);
    CHECK(water::fft_resources(sim).workspace == resources.workspace);
}
TEST_CASE("dispersion", borrowed_fft_events_time_only_the_queued_transform_pair) {
    water::Config c;
    c.grid = {8, 12, 4, 6}; c.impact_radius = 1;
    water::Simulation sim(c);
    cudaEvent_t begin, end;
    checked(cudaEventCreate(&begin)); checked(cudaEventCreate(&end));
    CHECK_THROWS(water::set_fft_timing_events(sim, begin, nullptr), std::invalid_argument);
    water::set_fft_timing_events(sim, begin, end);
    sim.step(.003f, 0);
    checked(cudaEventSynchronize(end));
    float ms;
    checked(cudaEventElapsedTime(&ms, begin, end)); CHECK(ms >= 0);
    water::set_fft_timing_events(sim, nullptr, nullptr);
    checked(cudaStreamSynchronize(sim.stream()));
    checked(cudaEventDestroy(end)); checked(cudaEventDestroy(begin));
    sim.step(.003f, 0);
}
