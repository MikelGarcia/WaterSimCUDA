#include "validation.hpp"

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
