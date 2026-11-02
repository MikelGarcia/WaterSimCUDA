#include "simulation.hpp"
#include "dispersive_continuity.hpp"

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

__global__ void reset_boat(BoatState* boat, float lx, float lz) {
    *boat = {lx / 2, lz / 2, lx >= lz ? 0.0f : 1.57079632679f, 0, 0, 0};
}

__device__ float driven_speed(float speed, float throttle, float dt) {
    constexpr float drag = 0.35f;
    if (speed * throttle < 0) {
        const float stop_time = log1pf(drag * fabsf(speed) / 1.5f) / drag;
        if (dt < stop_time) {
            return copysignf((fabsf(speed) + 1.5f / drag) * expf(-drag * dt) -
                             1.5f / drag, speed);
        }
        dt -= stop_time;
        speed = 0;
    }
    const float acceleration = throttle * (throttle >= 0 ? 0.75f : 0.50f);
    const float decay = expf(-drag * dt);
    const float next = speed * decay + acceleration * (-expm1f(-drag * dt)) / drag;
    // A reverse propulsion cap must not delete a pre-existing rebound impulse.
    if (throttle < 0) return fmaxf(next, fminf(speed, -0.9375f));
    return next;
}

__device__ void footprint(float yaw, float& ex, float& ez) {
    const float c = fabsf(cosf(yaw)), s = fabsf(sinf(yaw));
    ex = 0.5f * (boat_geometry.length * c + boat_geometry.beam * s);
    ez = 0.5f * (boat_geometry.length * s + boat_geometry.beam * c);
}

__global__ void update_boat(BoatState* boat, BoatControl control, float dt, float lx, float lz) {
    BoatState b = *boat;
    const float radius = 0.5f * hypotf(boat_geometry.length, boat_geometry.beam);
    const float travel_limit = fminf(boat_geometry.length, boat_geometry.beam) / 8;
    const float corner_speed = hypotf(b.vx, b.vz) + 0.75f * dt +
                               radius * fmaxf(2.0f, fabsf(b.yaw_rate));
    const float safe_dt = fminf(1.0f / 240, fminf(travel_limit / corner_speed,
                               0.034906585f / fmaxf(2.0f, fabsf(b.yaw_rate))));
    const float consumed = fminf(dt, 256 * safe_dt);
    const int count = min(256, max(1, static_cast<int>(ceilf(consumed / safe_dt))));
    const float h = consumed / count;
    const float initial_speed = hypotf(b.vx, b.vz);
    if (initial_speed > 1.875f) {
        b.vx *= 1.875f / initial_speed;
        b.vz *= 1.875f / initial_speed;
    }
    b.yaw_rate = fminf(2.0f, fmaxf(-2.0f, b.yaw_rate));
    int previous_contact = 0;
    int previous_translation_contact = 0;
    for (int i = 0; i < count; ++i) {
        const float c = cosf(b.yaw), s = sinf(b.yaw);
        const float longitudinal = b.vx * c + b.vz * s;
        const float lateral = -b.vx * s + b.vz * c;
        const float forward = driven_speed(longitudinal, control.throttle, h);
        const float sideways = lateral * expf(-4.0f * h);
        float vx = forward * c - sideways * s;
        float vz = forward * s + sideways * c;
        const float magnitude = hypotf(vx, vz);
        if (magnitude > 1.875f) {
            vx *= 1.875f / magnitude;
            vz *= 1.875f / magnitude;
        }
        const float mid_speed = 0.5f * (longitudinal + forward);
        const float ratio = mid_speed / 1.875f;
        const float target = control.rudder * 2.5f * mid_speed / (1 + 3 * ratio * ratio);
        float rate = target + (b.yaw_rate - target) * expf(-8 * h);
        rate = fminf(2.0f, fmaxf(-2.0f, rate));
        float yaw = b.yaw + target * h + (b.yaw_rate - target) * (-expm1f(-8 * h)) / 8;
        if (magnitude < 1e-4f) {
            rate = 0;
            yaw = b.yaw;
        }
        float x = b.x + 0.5f * (b.vx + vx) * h;
        float z = b.z + 0.5f * (b.vz + vz) * h;
        float ex, ez;
        footprint(yaw, ex, ez);
        int contact = 0;
        // Reject rotation that would sweep the footprint through a wall at this pose.
        if (b.x < ex || b.x > lx - ex || b.z < ez || b.z > lz - ez) {
            if (b.x < ex) contact |= 1;
            if (b.x > lx - ex) contact |= 2;
            if (b.z < ez) contact |= 4;
            if (b.z > lz - ez) contact |= 8;
            yaw = b.yaw;
            footprint(yaw, ex, ez);
        }
        int translation_contact = 0;
        if (x < ex) translation_contact |= 1;
        if (x > lx - ex) translation_contact |= 2;
        if (z < ez) translation_contact |= 4;
        if (z > lz - ez) translation_contact |= 8;
        contact |= translation_contact;
        x = fminf(lx - ex, fmaxf(ex, x));
        z = fminf(lz - ez, fmaxf(ez, z));
        if (((translation_contact & 1) && vx < 0) ||
            ((translation_contact & 2) && vx > 0)) vx *= -0.10f;
        if (((translation_contact & 4) && vz < 0) ||
            ((translation_contact & 8) && vz > 0)) vz *= -0.10f;
        if (translation_contact & ~previous_translation_contact) {
            // A rejected turn alone must not drain motion parallel to the wall.
            if (!(translation_contact & 3)) vx *= 0.25f;
            if (!(translation_contact & 12)) vz *= 0.25f;
        }
        if (contact & ~previous_contact) rate *= 0.20f;
        previous_contact = contact;
        previous_translation_contact = translation_contact;
        b = {x, z, yaw, vx, vz, rate};
    }
    *boat = b;
}

__global__ void boat_transform(const BoatState* boat, float* matrix, float lx, float lz) {
    const BoatState b = *boat;
    const float c = cosf(b.yaw), s = sinf(b.yaw);
    const float values[16] = {c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0,
                              b.x - lx / 2, 0, b.z - lz / 2, 1};
    for (int i = 0; i < 16; ++i) matrix[i] = values[i];
}

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

__device__ float hull_head(float x, float z, const BoatState& b,
                           float dx, float dz, float depth) {
    const float c = cosf(b.yaw), s = sinf(b.yaw);
    const float speed = hypotf(b.vx, b.vz);
    const float offset = speed > 0 ? .3f * boat_geometry.length * (b.vx * c + b.vz * s) / speed : 0;
    const float longitudinal = (x - b.x) * c + (z - b.z) * s - offset;
    const float transverse = -(x - b.x) * s + (z - b.z) * c;
    const float cell = fmaxf(dx, dz);
    const float a = fmaxf(.5f * boat_geometry.length, 2 * cell);
    const float d = fmaxf(.5f * boat_geometry.beam, 1.5f * cell);
    const float rho_squared = longitudinal * longitudinal / (a * a) +
                              transverse * transverse / (d * d);
    if (rho_squared >= 1) return 0;
    const float bump = 1 - rho_squared;
    const float speed_squared = (b.vx * b.vx + b.vz * b.vz) / (1.875f * 1.875f);
    const float relative_depth = fminf(1, depth / 3);
    return .045f * relative_depth * sqrtf(relative_depth) *
        (speed_squared / (1 + speed_squared)) * bump * bump * bump;
}

__global__ void update_velocity(const float* eta, float* u, float* v,
                                int nx, int ny, float dx, float dz,
                                float gravity_dt, float drag, const BoatState* boat, float depth) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ny * (nx + 1)) {
        const int x = i % (nx + 1);
        const int y = i / (nx + 1);
        u[i] = (x == 0 || x == nx) ? 0.0f :
            (u[i] - gravity_dt * (eta[y * nx + x] - eta[y * nx + x - 1] +
                hull_head((x + .5f) * dx, (y + .5f) * dz, *boat, dx, dz, depth) -
                hull_head((x - .5f) * dx, (y + .5f) * dz, *boat, dx, dz, depth)) / dx) / drag;
    }
    if (i < (ny + 1) * nx) {
        const int x = i % nx;
        const int y = i / nx;
        v[i] = (y == 0 || y == ny) ? 0.0f :
            (v[i] - gravity_dt * (eta[y * nx + x] - eta[(y - 1) * nx + x] +
                hull_head((x + .5f) * dx, (y + .5f) * dz, *boat, dx, dz, depth) -
                hull_head((x + .5f) * dx, (y - .5f) * dz, *boat, dx, dz, depth)) / dz) / drag;
    }
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
    const float required_length = boat_geometry.length + 2 * boat_geometry.spawn_clearance;
    const float required_beam = boat_geometry.beam + 2 * boat_geometry.spawn_clearance;
    if (std::max(g.lx, g.lz) < required_length || std::min(g.lx, g.lz) < required_beam) {
        throw std::invalid_argument("boat footprint does not fit: required basin dimensions at least " +
            std::to_string(required_length) + " m along spawn heading and " +
            std::to_string(required_beam) + " m across it (including clearance)");
    }
    try {
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create simulation stream");
        continuity_ = std::make_unique<DispersiveContinuity>(config_, stream_);
        check(cudaMalloc(&eta_, sizeof(float) * g.nx * g.ny), "allocate heights");
        check(cudaMalloc(&u_, sizeof(float) * (g.nx + 1) * g.ny), "allocate x velocities");
        check(cudaMalloc(&v_, sizeof(float) * g.nx * (g.ny + 1)), "allocate z velocities");
        check(cudaMalloc(&impact_sums_, sizeof(double2) * max_impacts), "allocate impact sums");
        check(cudaMalloc(&boat_, sizeof(BoatState)), "allocate boat");
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
    continuity_.reset();
    if (boat_) check_cleanup(cudaFree(boat_), "free boat");
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
    continuity_->set_depth(depth);
    config_ = next;
    max_dt_ = max_dt;
}

void Simulation::reset() {
    const auto& g = config_.grid;
    reset_boat<<<1, 1, 0, stream_>>>(boat_, g.lx, g.lz);
    check(cudaGetLastError(), "reset boat");
    check(cudaMemsetAsync(eta_, 0, sizeof(float) * g.nx * g.ny, stream_), "reset heights");
    check(cudaMemsetAsync(u_, 0, sizeof(float) * (g.nx + 1) * g.ny, stream_), "reset x velocities");
    check(cudaMemsetAsync(v_, 0, sizeof(float) * g.nx * (g.ny + 1), stream_), "reset z velocities");
}

void Simulation::write_boat_transform(float* device_column_major, std::size_t float_count) {
    if (!device_column_major || float_count != 16) {
        throw std::invalid_argument("boat transform requires exactly 16 device floats");
    }
    const auto& g = config_.grid;
    boat_transform<<<1, 1, 0, stream_>>>(boat_, device_column_major, g.lx, g.lz);
    check(cudaGetLastError(), "launch boat transform");
}

void Simulation::step(float dt, float damping, BoatControl control) {
    if (!std::isfinite(dt) || dt <= 0.0f || dt > max_dt_) {
        throw std::invalid_argument("step dt must be finite, positive, and within the current stability bound");
    }
    if (!std::isfinite(damping) || damping < 0.0f) {
        throw std::invalid_argument("step damping must be finite and nonnegative");
    }
    if (!std::isfinite(control.throttle) || !std::isfinite(control.rudder) ||
        std::abs(control.throttle) > 1 || std::abs(control.rudder) > 1) {
        throw std::invalid_argument("boat controls must be finite and in [-1, 1]");
    }
    const auto& g = config_.grid;
    update_boat<<<1, 1, 0, stream_>>>(boat_, control, dt, g.lx, g.lz);
    check(cudaGetLastError(), "launch boat update");
    constexpr int threads = 256;
    const int faces = std::max((g.nx + 1) * g.ny, g.nx * (g.ny + 1));
    const float dx = g.lx / g.nx;
    const float dz = g.lz / g.ny;
    update_velocity<<<(faces + threads - 1) / threads, threads, 0, stream_>>>(
        eta_, u_, v_, g.nx, g.ny, dx, dz, config_.gravity * dt, 1.0f + damping * dt,
        boat_, config_.depth);
    check(cudaGetLastError(), "launch velocity update");
    continuity_->advance(eta_, u_, v_, dt);
}

} // namespace water
