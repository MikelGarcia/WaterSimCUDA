#pragma once

#include "timeline.hpp"
#include <cstdint>

namespace water {

class AppController {
public:
    AppController(Simulation& simulation, const Config& config);
    void frame(double wall_seconds, bool paused, bool rain_enabled, float rate,
               float strength, float damping);
    void reset();
    bool spawn(float x, float z);
    const std::vector<Drop>& drops() const { return scheduler_.active(); }
    double sim_time() const { return ticks_ * FixedClock::quantum; }

private:
    Simulation& simulation_;
    FixedClock clock_;
    DropScheduler scheduler_;
    const int substeps_;
    const float dt_;
    std::uint64_t ticks_ = 0;
};

} // namespace water
