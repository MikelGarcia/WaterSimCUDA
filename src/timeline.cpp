#include "timeline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace water {

int FixedClock::advance(double wall_seconds, bool paused) {
    if (!std::isfinite(wall_seconds) || wall_seconds < 0.0) {
        throw std::invalid_argument("elapsed time must be finite and nonnegative");
    }
    if (paused) return 0;
    const double total = remainder_ + wall_seconds;
    double quanta;
    remainder_ = std::modf(total / quantum, &quanta) * quantum;
    return static_cast<int>(std::min(quanta, 8.0));
}

void FixedClock::reset() {
    remainder_ = 0.0;
}

DropScheduler::DropScheduler(const Config& config) : config_(config), random_(config.seed) {
    validate(config_);
    active_.reserve(256);
    reset();
}

bool DropScheduler::spawn(float x, float z, double sim_time) {
    if (!std::isfinite(sim_time) || sim_time < time_) {
        throw std::invalid_argument("drop time must be finite and monotonic");
    }
    if (!std::isfinite(x) || !std::isfinite(z) ||
        x < 0.0f || x > config_.grid.lx || z < 0.0f || z > config_.grid.lz ||
        active_.size() == 256) return false;
    active_.push_back({x, z, sim_time, sim_time + 0.6});
    return true;
}

void DropScheduler::collect_arrivals(double end_time, float strength, std::vector<Impact>& events) {
    const auto remaining = std::remove_if(active_.begin(), active_.end(), [&](const Drop& drop) {
        if (drop.arrival > end_time) return false;
        events.push_back({drop.x, drop.z, config_.impact_radius, strength});
        return true;
    });
    active_.erase(remaining, active_.end());
}

std::vector<Impact> DropScheduler::tick(double end_time, bool rain_enabled, float rain_rate,
                                      float strength) {
    if (!std::isfinite(end_time) || end_time < time_) {
        throw std::invalid_argument("tick time must be finite and monotonic");
    }
    if (!std::isfinite(rain_rate) || rain_rate < 0.0f || rain_rate > 60.0f) {
        throw std::invalid_argument("rain rate must be in [0, 60]");
    }
    if (!std::isfinite(strength) || strength < 0.0f || strength > 0.02f * config_.depth) {
        throw std::invalid_argument("impact strength must be in [0, 0.02 * depth]");
    }
    const float rate = rain_enabled ? rain_rate : 0.0f;
    if (rate != rain_rate_) {
        next_rain_ = std::numeric_limits<double>::infinity();
        if (rate > 0.0f) {
            next_rain_ = time_ + std::exponential_distribution<double>(rate)(random_);
        }
        rain_rate_ = rate;
    }

    std::vector<Impact> events;
    int spawned = 0;
    while (next_rain_ <= end_time) {
        collect_arrivals(next_rain_, strength, events);
        if (active_.size() == 256 || spawned == 256) {
            // Discard excess rain rather than queueing work beyond this tick.
            next_rain_ = end_time + std::exponential_distribution<double>(rate)(random_);
            break;
        }
        const float x = std::uniform_real_distribution<float>(0.0f, config_.grid.lx)(random_);
        const float z = std::uniform_real_distribution<float>(0.0f, config_.grid.lz)(random_);
        spawn(x, z, next_rain_);
        ++spawned;
        next_rain_ += std::exponential_distribution<double>(rate)(random_);
    }
    collect_arrivals(end_time, strength, events);
    time_ = end_time;
    return events;
}

void DropScheduler::reset() {
    active_.clear();
    random_.seed(config_.seed);
    time_ = 0.0;
    rain_rate_ = 0.0f;
    next_rain_ = std::numeric_limits<double>::infinity();
}

std::optional<Vector2> pick_water(Ray ray, const Grid& grid) {
    if (!std::isfinite(ray.position.x) || !std::isfinite(ray.position.y) ||
        !std::isfinite(ray.position.z) || !std::isfinite(ray.direction.x) ||
        !std::isfinite(ray.direction.y) || !std::isfinite(ray.direction.z) ||
        ray.direction.y >= 0.0f) return std::nullopt;
    const double distance = -static_cast<double>(ray.position.y) / ray.direction.y;
    if (distance < 0.0) return std::nullopt;
    const double x = ray.position.x + distance * ray.direction.x + grid.lx / 2.0;
    const double z = ray.position.z + distance * ray.direction.z + grid.lz / 2.0;
    if (x < 0.0 || x > grid.lx || z < 0.0 || z > grid.lz) return std::nullopt;
    return Vector2{static_cast<float>(x), static_cast<float>(z)};
}

} // namespace water
