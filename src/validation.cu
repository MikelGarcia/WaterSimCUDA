#include "validation.hpp"
#include "dispersive_continuity.hpp"

#include <stdexcept>
#include <string>

namespace water {
namespace {

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

} // namespace

FftResources fft_resources(const Simulation& simulation) {
    const auto& op = *simulation.continuity_;
    const auto& g = op.grid_;
    const std::size_t spectral_count = 2 * std::size_t(g.ny) * (g.nx + 1);
    return {op.forward_, op.inverse_, op.workspace_, 4 * std::size_t(g.nx) * g.ny * sizeof(float),
            spectral_count * sizeof(cufftComplex), 2 * spectral_count * sizeof(float), op.workspace_bytes_};
}

void set_fft_timing_events(Simulation& simulation, cudaEvent_t begin, cudaEvent_t end) {
    if (bool(begin) != bool(end)) throw std::invalid_argument("FFT timing requires an event pair");
    simulation.continuity_->timing_begin_ = begin;
    simulation.continuity_->timing_end_ = end;
}

BoatState download_boat(const Simulation& simulation) {
    BoatState state;
    check(cudaMemcpyAsync(&state, simulation.boat_, sizeof(state), cudaMemcpyDeviceToHost,
                          simulation.stream_), "download boat");
    check(cudaStreamSynchronize(simulation.stream_), "synchronize boat download");
    return state;
}

void upload_boat(Simulation& simulation, const BoatState& state) {
    check(cudaMemcpyAsync(simulation.boat_, &state, sizeof(state), cudaMemcpyHostToDevice,
                          simulation.stream_), "upload boat");
    check(cudaStreamSynchronize(simulation.stream_), "synchronize boat upload");
}

Snapshot download(const Simulation& simulation) {
    const auto& g = simulation.config_.grid;
    Snapshot state{std::vector<float>(g.nx * g.ny),
                   std::vector<float>((g.nx + 1) * g.ny),
                   std::vector<float>(g.nx * (g.ny + 1))};
    check(cudaMemcpyAsync(state.eta.data(), simulation.eta_, state.eta.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, simulation.stream_), "download heights");
    check(cudaMemcpyAsync(state.u.data(), simulation.u_, state.u.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, simulation.stream_), "download x velocities");
    check(cudaMemcpyAsync(state.v.data(), simulation.v_, state.v.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, simulation.stream_), "download z velocities");
    check(cudaStreamSynchronize(simulation.stream_), "synchronize state download");
    return state;
}

void upload(Simulation& simulation, const Snapshot& state) {
    const auto& g = simulation.config_.grid;
    if (state.eta.size() != static_cast<std::size_t>(g.nx * g.ny) ||
        state.u.size() != static_cast<std::size_t>((g.nx + 1) * g.ny) ||
        state.v.size() != static_cast<std::size_t>(g.nx * (g.ny + 1))) {
        throw std::invalid_argument("snapshot sizes must match the simulation grid");
    }
    check(cudaMemcpyAsync(simulation.eta_, state.eta.data(), state.eta.size() * sizeof(float),
                          cudaMemcpyHostToDevice, simulation.stream_), "upload heights");
    check(cudaMemcpyAsync(simulation.u_, state.u.data(), state.u.size() * sizeof(float),
                          cudaMemcpyHostToDevice, simulation.stream_), "upload x velocities");
    check(cudaMemcpyAsync(simulation.v_, state.v.data(), state.v.size() * sizeof(float),
                          cudaMemcpyHostToDevice, simulation.stream_), "upload z velocities");
    check(cudaStreamSynchronize(simulation.stream_), "synchronize state upload");
}

} // namespace water
