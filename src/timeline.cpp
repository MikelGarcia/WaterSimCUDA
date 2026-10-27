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

double drop_spawn_height(float x, float z, const Grid& grid, const SpawnView& view) {
    const auto& camera = view.camera;
    const auto finite = [](Vector3 v) {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    };
    if (!finite(camera.position) || !finite(camera.target) || !finite(camera.up) ||
        !std::isfinite(view.aspect) || view.aspect <= 0 ||
        !std::isfinite(camera.fovy) || camera.fovy <= 0 || camera.fovy >= 180 ||
        camera.projection != CAMERA_PERSPECTIVE) {
        throw std::invalid_argument("spawn view must be a finite perspective camera");
    }

    struct Vec { double x, y, z; };
    const auto normalize = [](Vec v) {
        const double length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (length <= 1e-12) throw std::invalid_argument("degenerate spawn camera basis");
        return Vec{v.x / length, v.y / length, v.z / length};
    };
    const Vec forward = normalize({static_cast<double>(camera.target.x) - camera.position.x,
                                   static_cast<double>(camera.target.y) - camera.position.y,
                                   static_cast<double>(camera.target.z) - camera.position.z});
    const Vec given_up = normalize({camera.up.x, camera.up.y, camera.up.z});
    const double along = given_up.x * forward.x + given_up.y * forward.y + given_up.z * forward.z;
    const Vec up = normalize({given_up.x - along * forward.x, given_up.y - along * forward.y,
                              given_up.z - along * forward.z});
    const double tangent = std::tan(camera.fovy * 3.14159265358979323846 / 360.0);
    const Vec upper = normalize({up.x - forward.x * tangent, up.y - forward.y * tangent,
                                 up.z - forward.z * tangent});
    const Vec behind{-forward.x, -forward.y, -forward.z};
    const double relative_x = x - grid.lx / 2.0 - camera.position.x;
    const double relative_z = z - grid.lz / 2.0 - camera.position.z;
    double height = std::numeric_limits<double>::infinity();
    for (const Vec plane : {upper, behind}) {
        if (plane.y <= 1e-8) continue;
        // Local mesh y is nonnegative, so its lowest corners bound both outward planes.
        const double candidate = camera.position.y +
            (drop_spawn_margin + drop_radius * (std::abs(plane.x) + std::abs(plane.z)) -
             plane.x * relative_x - plane.z * relative_z) / plane.y;
        if (std::isfinite(candidate)) height = std::min(height, candidate);
    }
    if (!std::isfinite(height)) throw std::invalid_argument("spawn camera has no upward excluding plane");
    height = std::max(height, static_cast<double>(camera.position.y) + drop_spawn_margin);
    return height + 1e-6 * std::max(1.0, std::abs(height));
}

double drop_y(const Drop& drop, double sim_time) {
    if (sim_time >= drop.arrival) return 0.0;
    const double elapsed = std::max(0.0, sim_time - drop.start);
    return std::max(0.0, drop.spawn_height - 0.5 * drop_acceleration * elapsed * elapsed);
}

bool DropScheduler::spawn(float x, float z, double sim_time, const SpawnView& view) {
    if (!std::isfinite(sim_time) || sim_time < time_) {
        throw std::invalid_argument("drop time must be finite and monotonic");
    }
    if (!std::isfinite(x) || !std::isfinite(z) ||
        x < 0.0f || x > config_.grid.lx || z < 0.0f || z > config_.grid.lz ||
        active_.size() == 256) return false;
    const double height = drop_spawn_height(x, z, config_.grid, view);
    active_.push_back({x, z, height, sim_time,
                       sim_time + std::sqrt(2 * height / drop_acceleration)});
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
                                      float strength, const SpawnView& view) {
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
        spawn(x, z, next_rain_, view);
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
