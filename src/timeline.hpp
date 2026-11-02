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

struct SpawnView { Camera3D camera; float aspect; };

inline constexpr float drop_radius = 0.065f, drop_height = 0.28f, drop_spawn_margin = 0.12f;
inline constexpr double drop_acceleration = 88.29;

struct Drop {
    float x, z;
    double spawn_height, start, arrival;
};

double drop_spawn_height(float x, float z, const Grid&, const SpawnView&);
double drop_y(const Drop&, double sim_time);

class DropScheduler {
public:
    void set_depth(float depth);
    explicit DropScheduler(const Config& config);
    bool spawn(float x, float z, double sim_time, const SpawnView&);
    std::vector<Impact> tick(double end_time, bool rain_enabled, float rain_rate, float strength,
                             const SpawnView&);
    const std::vector<Drop>& active() const { return active_; }
    void reset();

private:
    bool add_drop(float x, float z, double sim_time, const SpawnView&);
    void collect_arrivals(double end_time, float strength, std::vector<Impact>& events);

    Config config_;
    std::mt19937 random_;
    std::vector<Drop> active_;
    double time_ = 0.0;
    double next_rain_ = 0.0;
    float rain_rate_ = 0.0f;
};

std::optional<Vector2> pick_water(Ray ray, const Grid& grid);

} // namespace water
