#include "simulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace water {
namespace {

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void check_cleanup(cudaError_t error, const char* operation) noexcept {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    }
}

__global__ void update_velocity(const float* eta, float* u, float* v,
                                int nx, int ny, float dx, float dz,
                                float gravity_dt, float drag) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ny * (nx + 1)) {
        const int x = i % (nx + 1);
        const int y = i / (nx + 1);
        u[i] = (x == 0 || x == nx) ? 0.0f :
            (u[i] - gravity_dt * (eta[y * nx + x] - eta[y * nx + x - 1]) / dx) / drag;
    }
    if (i < (ny + 1) * nx) {
        const int x = i % nx;
        const int y = i / nx;
        v[i] = (y == 0 || y == ny) ? 0.0f :
            (v[i] - gravity_dt * (eta[y * nx + x] - eta[(y - 1) * nx + x]) / dz) / drag;
    }
}

__global__ void update_height(float* eta, const float* u, const float* v,
                              int nx, int ny, float dx, float dz, float depth_dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx * ny) return;
    const int x = i % nx;
    const int y = i / nx;
    const float divergence = (u[y * (nx + 1) + x + 1] - u[y * (nx + 1) + x]) / dx +
                             (v[(y + 1) * nx + x] - v[y * nx + x]) / dz;
    eta[i] -= depth_dt * divergence;
}

} // namespace

Simulation::Simulation(const Config& config) : config_(config), max_dt_(stable_dt(config)) {
    const auto& g = config_.grid;
    try {
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create simulation stream");
        check(cudaMalloc(&eta_, sizeof(float) * g.nx * g.ny), "allocate heights");
        check(cudaMalloc(&u_, sizeof(float) * (g.nx + 1) * g.ny), "allocate x velocities");
        check(cudaMalloc(&v_, sizeof(float) * g.nx * (g.ny + 1)), "allocate z velocities");
        reset();
    } catch (...) {
        release();
        throw;
    }
}

Simulation::~Simulation() { release(); }

void Simulation::release() noexcept {
    if (eta_) check_cleanup(cudaFree(eta_), "free heights");
    if (u_) check_cleanup(cudaFree(u_), "free x velocities");
    if (v_) check_cleanup(cudaFree(v_), "free z velocities");
    if (stream_) check_cleanup(cudaStreamDestroy(stream_), "destroy simulation stream");
}

void Simulation::reset() {
    const auto& g = config_.grid;
    check(cudaMemsetAsync(eta_, 0, sizeof(float) * g.nx * g.ny, stream_), "reset heights");
    check(cudaMemsetAsync(u_, 0, sizeof(float) * (g.nx + 1) * g.ny, stream_), "reset x velocities");
    check(cudaMemsetAsync(v_, 0, sizeof(float) * g.nx * (g.ny + 1), stream_), "reset z velocities");
}

void Simulation::step(float dt, float damping) {
    if (!std::isfinite(dt) || dt <= 0.0f || dt > max_dt_) {
        throw std::invalid_argument("step dt must be finite, positive, and within the startup CFL bound");
    }
    if (!std::isfinite(damping) || damping < 0.0f) {
        throw std::invalid_argument("step damping must be finite and nonnegative");
    }
    const auto& g = config_.grid;
    constexpr int threads = 256;
    const int faces = std::max((g.nx + 1) * g.ny, g.nx * (g.ny + 1));
    const float dx = g.lx / g.nx;
    const float dz = g.lz / g.ny;
    update_velocity<<<(faces + threads - 1) / threads, threads, 0, stream_>>>(
        eta_, u_, v_, g.nx, g.ny, dx, dz, config_.gravity * dt, 1.0f + damping * dt);
    check(cudaGetLastError(), "launch velocity update");
    update_height<<<(g.nx * g.ny + threads - 1) / threads, threads, 0, stream_>>>(
        eta_, u_, v_, g.nx, g.ny, dx, dz, config_.depth * dt);
    check(cudaGetLastError(), "launch height update");
}

} // namespace water
