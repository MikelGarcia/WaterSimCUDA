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

constexpr int max_impacts = 64;

struct ImpactBatch {
    Impact events[max_impacts];
    int count;
};

__device__ double2 impact_weights(const Impact& event, double x, double z) {
    const double px = x - event.x;
    const double pz = z - event.z;
    const double radius_squared = static_cast<double>(event.radius) * event.radius;
    const double distance_squared = px * px + pz * pz;
    if (distance_squared > radius_squared) return make_double2(0.0, 0.0);
    const double scaled = distance_squared / radius_squared;
    return make_double2(exp(-4.5 * scaled), exp(-1.125 * scaled));
}

__global__ void sum_impact_support(ImpactBatch batch, double2* sums,
                                   int nx, int ny, double dx, double dz) {
    const Impact event = batch.events[blockIdx.x];
    const double radius = event.radius;
    const int first_x = max(0, static_cast<int>(ceil((event.x - radius) / dx - 0.5)));
    const int last_x = min(nx - 1, static_cast<int>(floor((event.x + radius) / dx - 0.5)));
    const int first_z = max(0, static_cast<int>(ceil((event.z - radius) / dz - 0.5)));
    const int last_z = min(ny - 1, static_cast<int>(floor((event.z + radius) / dz - 0.5)));
    const int width = last_x - first_x + 1;
    double2 total = make_double2(0.0, 0.0);
    for (int i = threadIdx.x; i < width * (last_z - first_z + 1); i += blockDim.x) {
        const int x = first_x + i % width;
        const int z = first_z + i / width;
        const double2 weights = impact_weights(event, (x + 0.5) * dx, (z + 0.5) * dz);
        total.x += weights.x;
        total.y += weights.y;
    }
    __shared__ double2 partial[256];
    partial[threadIdx.x] = total;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x].x += partial[threadIdx.x + stride].x;
            partial[threadIdx.x].y += partial[threadIdx.x + stride].y;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[blockIdx.x] = partial[0];
}

__global__ void apply_impact_batch(float* eta, ImpactBatch batch, const double2* sums,
                                   int nx, int ny, double dx, double dz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx * ny) return;
    const double x = (i % nx + 0.5) * dx;
    const double z = (i / nx + 0.5) * dz;
    float height = eta[i];
    for (int e = 0; e < batch.count; ++e) {
        const Impact event = batch.events[e];
        const double2 weights = impact_weights(event, x, z);
        height += static_cast<float>(event.strength *
            (weights.x - weights.y * (sums[e].x / sums[e].y)));
    }
    eta[i] = height;
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

__device__ float node_height(const float* eta, int x, int z, int nx, int ny) {
    float height = 0.0f;
    int count = 0;
    for (int row = max(0, z - 1); row <= min(ny - 1, z); ++row) {
        for (int column = max(0, x - 1); column <= min(nx - 1, x); ++column) {
            height += eta[row * nx + column];
            ++count;
        }
    }
    return height / count;
}

__global__ void build_surface(const float* eta, Vertex* positions, Vertex* normals,
                              int nx, int ny, float lx, float lz) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nx * ny) return;
    const int x = cell % nx, z = cell / nx;
    const float dx = lx / nx, dz = lz / ny;
    Vertex corners[4], corner_normals[4];
    for (int corner = 0; corner < 4; ++corner) {
        const int node_x = x + corner % 2, node_z = z + corner / 2;
        const int left = max(0, node_x - 1), right = min(nx, node_x + 1);
        const int near = max(0, node_z - 1), far = min(ny, node_z + 1);
        const float sx = (node_height(eta, right, node_z, nx, ny) -
                          node_height(eta, left, node_z, nx, ny)) / ((right - left) * dx);
        const float sz = (node_height(eta, node_x, far, nx, ny) -
                          node_height(eta, node_x, near, nx, ny)) / ((far - near) * dz);
        const float inverse_length = rsqrtf(sx * sx + 1.0f + sz * sz);
        corners[corner] = {node_x * dx - lx / 2, node_height(eta, node_x, node_z, nx, ny),
                           node_z * dz - lz / 2};
        corner_normals[corner] = {-sx * inverse_length, inverse_length, -sz * inverse_length};
    }
    const int order[6] = {0, 2, 1, 1, 2, 3};
    for (int vertex = 0; vertex < 6; ++vertex) {
        positions[cell * 6 + vertex] = corners[order[vertex]];
        normals[cell * 6 + vertex] = corner_normals[order[vertex]];
    }
}

} // namespace

Simulation::Simulation(const Config& config) : config_(config), max_dt_(stable_dt(config)) {
    const auto& g = config_.grid;
    try {
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create simulation stream");
        check(cudaMalloc(&eta_, sizeof(float) * g.nx * g.ny), "allocate heights");
        check(cudaMalloc(&u_, sizeof(float) * (g.nx + 1) * g.ny), "allocate x velocities");
        check(cudaMalloc(&v_, sizeof(float) * g.nx * (g.ny + 1)), "allocate z velocities");
        check(cudaMalloc(&impact_sums_, sizeof(double2) * max_impacts), "allocate impact sums");
        reset();
    } catch (...) {
        release();
        throw;
    }
}

Simulation::~Simulation() { release(); }

void Simulation::write_surface(Vertex* device_positions, Vertex* device_normals,
                               std::size_t vertex_count) {
    const auto& g = config_.grid;
    if (!device_positions || !device_normals ||
        vertex_count != std::size_t{6} * g.nx * g.ny) {
        throw std::invalid_argument("surface requires two device buffers with exactly 6*nx*ny vertices");
    }
    constexpr int threads = 128;
    build_surface<<<(g.nx * g.ny + threads - 1) / threads, threads, 0, stream_>>>(
        eta_, device_positions, device_normals, g.nx, g.ny, g.lx, g.lz);
    check(cudaGetLastError(), "launch surface generation");
}

void Simulation::apply_impacts(const Impact* host_events, std::size_t count) {
    if (count > max_impacts) throw std::invalid_argument("at most 64 impacts may be submitted per call");
    if (count == 0) return;
    if (!host_events) throw std::invalid_argument("nonempty impact batch requires events");
    const auto& g = config_.grid;
    const double dx = static_cast<double>(g.lx) / g.nx;
    const double dz = static_cast<double>(g.lz) / g.ny;
    ImpactBatch batch{};
    batch.count = static_cast<int>(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& event = host_events[i];
        if (!std::isfinite(event.x) || event.x < 0.0f || event.x > g.lx ||
            !std::isfinite(event.z) || event.z < 0.0f || event.z > g.lz) {
            throw std::invalid_argument("impact position must be finite and inside the basin");
        }
        if (!std::isfinite(event.radius) || event.radius <= 0.0f ||
            event.radius < 2.0 * std::max(dx, dz) ||
            event.radius > 0.25 * std::min(g.lx, g.lz)) {
            throw std::invalid_argument("impact radius must span at least two cells and at most one quarter of the basin");
        }
        if (!std::isfinite(event.strength) || event.strength < 0.0f ||
            event.strength > 0.02f * config_.depth) {
            throw std::invalid_argument("impact strength must be finite and in [0, 0.02 * depth]");
        }
        batch.events[i] = event;
    }
    constexpr int threads = 256;
    // Launch parameters snapshot the compact batch before the caller can reuse its events.
    sum_impact_support<<<batch.count, threads, 0, stream_>>>(batch, impact_sums_, g.nx, g.ny, dx, dz);
    check(cudaGetLastError(), "launch impact support reduction");
    apply_impact_batch<<<(g.nx * g.ny + threads - 1) / threads, threads, 0, stream_>>>(
        eta_, batch, impact_sums_, g.nx, g.ny, dx, dz);
    check(cudaGetLastError(), "launch impact application");
}

void Simulation::release() noexcept {
    if (eta_) check_cleanup(cudaFree(eta_), "free heights");
    if (u_) check_cleanup(cudaFree(u_), "free x velocities");
    if (v_) check_cleanup(cudaFree(v_), "free z velocities");
    if (impact_sums_) check_cleanup(cudaFree(impact_sums_), "free impact sums");
    if (stream_) check_cleanup(cudaStreamDestroy(stream_), "destroy simulation stream");
}

void Simulation::set_depth(float depth) {
    Config next = config_;
    next.depth = depth;
    next.impact_strength = std::min(next.impact_strength, 0.02f * depth);
    substeps_per_quantum(next);
    const double max_dt = stable_dt(next);
    config_ = next;
    max_dt_ = max_dt;
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
