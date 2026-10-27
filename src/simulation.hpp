#pragma once

#include "config.hpp"
#include <cuda_runtime_api.h>
#include <cstddef>

namespace water {

struct Snapshot;

struct Impact {
    float x, z, radius, strength;
};

class Simulation {
public:
    explicit Simulation(const Config& config);
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    void reset();
    void step(float dt, float damping);
    void apply_impacts(const Impact* host_events, std::size_t count);
    cudaStream_t stream() const { return stream_; }
    const Config& config() const { return config_; }

private:
    friend Snapshot download(const Simulation& simulation);
    friend void upload(Simulation& simulation, const Snapshot& state);
    void release() noexcept;

    const Config config_;
    const double max_dt_;
    cudaStream_t stream_ = nullptr;
    float* eta_ = nullptr;
    float* u_ = nullptr;
    float* v_ = nullptr;
    double2* impact_sums_ = nullptr;
};

} // namespace water
