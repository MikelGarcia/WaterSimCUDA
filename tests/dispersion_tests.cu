#include "dispersive_continuity.hpp"
#include "dispersion_reference.hpp"
#include "test_support.hpp"
#include <algorithm>
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
