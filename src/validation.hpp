#pragma once

#include "simulation.hpp"
#include <vector>

namespace water {

struct Snapshot {
    std::vector<float> eta, u, v;
};

Snapshot download(const Simulation& simulation);
void upload(Simulation& simulation, const Snapshot& state);
BoatState download_boat(const Simulation& simulation);
void upload_boat(Simulation& simulation, const BoatState& state);

} // namespace water
