#pragma once

#include "timeline.hpp"
#include <cstdint>

namespace water {

class AppController {
public:
    void set_depth(float depth);
    AppController(Simulation& simulation, const Config& config);
    void frame(double wall_seconds, bool paused, bool rain_enabled, float rate,
               float strength, float damping, const SpawnView&);
    void reset();
    bool spawn(float x, float z, const SpawnView&);
    const std::vector<Drop>& drops() const { return scheduler_.active(); }
    double sim_time() const { return ticks_ * FixedClock::quantum; }

private:
    Simulation& simulation_;
    FixedClock clock_;
    DropScheduler scheduler_;
    int substeps_;
    float dt_;
    std::uint64_t ticks_ = 0;
};

} // namespace water
