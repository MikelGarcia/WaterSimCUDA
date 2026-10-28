#include "app_controller.hpp"
#include <algorithm>

namespace water {

AppController::AppController(Simulation& simulation, const Config& config)
    : simulation_(simulation), scheduler_(config), substeps_(substeps_per_quantum(config)),
      dt_(static_cast<float>(FixedClock::quantum / substeps_)) {}

void AppController::frame(double wall_seconds, bool paused, bool rain_enabled, float rate,
                          float strength, float damping, const SpawnView& view, BoatControl control) {
    const int quanta = clock_.advance(wall_seconds, paused);
    for (int tick = 0; tick < quanta; ++tick) {
        const auto impacts = scheduler_.tick((ticks_ + 1) * FixedClock::quantum,
                                              rain_enabled, rate, strength, view);
        for (std::size_t first = 0; first < impacts.size(); first += 64) {
            simulation_.apply_impacts(impacts.data() + first,
                                     std::min<std::size_t>(64, impacts.size() - first));
        }
        for (int i = 0; i < substeps_; ++i) simulation_.step(dt_, damping, control);
        ++ticks_;
    }
}

void AppController::set_depth(float depth) {
    Config next = simulation_.config();
    next.depth = depth;
    next.impact_strength = std::min(next.impact_strength, 0.02f * depth);
    const int substeps = substeps_per_quantum(next);
    const float dt = static_cast<float>(FixedClock::quantum / substeps);
    simulation_.set_depth(depth);
    scheduler_.set_depth(depth);
    substeps_ = substeps;
    dt_ = dt;
}

void AppController::reset() {
    simulation_.reset();
    clock_.reset();
    scheduler_.reset();
    ticks_ = 0;
}

bool AppController::spawn(float x, float z, const SpawnView& view) {
    return scheduler_.spawn(x, z, sim_time(), view);
}

} // namespace water
