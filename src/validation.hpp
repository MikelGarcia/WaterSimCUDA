#pragma once

#include "simulation.hpp"
#include <vector>

namespace water {

struct Snapshot {
    std::vector<float> eta, u, v;
};

struct FftResources {
    int forward_plan, inverse_plan;
    const void* workspace;
    std::size_t real_bytes, spectral_bytes, weights_bytes, workspace_bytes;
};
FftResources fft_resources(const Simulation& simulation);
// Events are borrowed and must outlive queued work. A null pair disables timing.
void set_fft_timing_events(Simulation& simulation, cudaEvent_t begin, cudaEvent_t end);

Snapshot download(const Simulation& simulation);
void upload(Simulation& simulation, const Snapshot& state);
BoatState download_boat(const Simulation& simulation);
void upload_boat(Simulation& simulation, const BoatState& state);

} // namespace water
