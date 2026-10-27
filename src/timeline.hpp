#pragma once

#include "simulation.hpp"
#include <raylib.h>
#include <optional>
#include <random>
#include <vector>

namespace water {

class FixedClock {
public:
    static constexpr double quantum = 1.0 / 120.0;
    int advance(double wall_seconds, bool paused);
    void reset();

private:
    double remainder_ = 0.0;
};

struct Drop {
    float x, z;
    double start, arrival;
};

class DropScheduler {
public:
    explicit DropScheduler(const Config& config);
    bool spawn(float x, float z, double sim_time);
    std::vector<Impact> tick(double end_time, bool rain_enabled, float rain_rate, float strength);
    const std::vector<Drop>& active() const { return active_; }
    void reset();

private:
    void collect_arrivals(double end_time, float strength, std::vector<Impact>& events);

    const Config config_;
    std::mt19937 random_;
    std::vector<Drop> active_;
    double time_ = 0.0;
    double next_rain_ = 0.0;
    float rain_rate_ = 0.0f;
};

std::optional<Vector2> pick_water(Ray ray, const Grid& grid);

} // namespace water
