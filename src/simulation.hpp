#pragma once

#include "config.hpp"
#include "boat.hpp"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <memory>

namespace water {

struct Snapshot;
struct FftResources;
class DispersiveContinuity;

struct Impact {
    float x, z, radius, strength;
};

struct Vertex {
    float x, y, z;
};

class Simulation {
public:
    void set_depth(float depth);
    explicit Simulation(const Config& config);
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    void reset();
    void step(float dt, float damping, BoatControl control = {});
    void write_boat_transform(float* device_column_major, std::size_t float_count);
    void apply_impacts(const Impact* host_events, std::size_t count);
    void write_surface(Vertex* device_positions, Vertex* device_normals, std::size_t vertex_count);
    cudaStream_t stream() const { return stream_; }
    const Config& config() const { return config_; }

private:
    friend BoatState download_boat(const Simulation& simulation);
    friend void upload_boat(Simulation& simulation, const BoatState& state);
    friend Snapshot download(const Simulation& simulation);
    friend void upload(Simulation& simulation, const Snapshot& state);
    friend FftResources fft_resources(const Simulation&);
    friend void set_fft_timing_events(Simulation&, cudaEvent_t, cudaEvent_t);
    void release() noexcept;

    Config config_;
    double max_dt_;
    cudaStream_t stream_ = nullptr;
    std::unique_ptr<DispersiveContinuity> continuity_;
    float* eta_ = nullptr;
    float* u_ = nullptr;
    float* v_ = nullptr;
    double2* impact_sums_ = nullptr;
    BoatState* boat_ = nullptr;
};

} // namespace water
