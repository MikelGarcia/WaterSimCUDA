#include "config.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace water {
namespace {

template <typename T>
T parse_number(std::string_view text, std::string_view flag) {
    T value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string(flag) + ": invalid value '" +
                                    std::string(text) + "'");
    }
    return value;
}

} // namespace

Config parse_config(int argc, const char* const* argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag != "--nx" && flag != "--ny" && flag != "--lx" && flag != "--lz" &&
            flag != "--depth" && flag != "--damping" && flag != "--radius" &&
            flag != "--strength" && flag != "--rain-rate" && flag != "--seed" &&
            flag != "--reflection") {
            throw std::invalid_argument("unknown option: " + std::string(flag));
        }
        if (++i == argc) {
            throw std::invalid_argument(std::string(flag) + ": missing value");
        }
        const std::string_view value(argv[i]);
        if (flag == "--nx") config.grid.nx = parse_number<int>(value, flag);
        else if (flag == "--ny") config.grid.ny = parse_number<int>(value, flag);
        else if (flag == "--lx") config.grid.lx = parse_number<float>(value, flag);
        else if (flag == "--lz") config.grid.lz = parse_number<float>(value, flag);
        else if (flag == "--depth") config.depth = parse_number<float>(value, flag);
        else if (flag == "--damping") config.damping = parse_number<float>(value, flag);
        else if (flag == "--radius") config.impact_radius = parse_number<float>(value, flag);
        else if (flag == "--strength") config.impact_strength = parse_number<float>(value, flag);
        else if (flag == "--rain-rate") config.rain_rate = parse_number<float>(value, flag);
        else if (flag == "--seed") config.seed = parse_number<unsigned>(value, flag);
        else if (flag == "--reflection") config.reflection = parse_number<float>(value, flag);
    }
    validate(config);
    return config;
}

void validate(const Config& config) {
    const auto& grid = config.grid;
    if (!std::isfinite(config.reflection) || config.reflection < 0 || config.reflection > 1) {
        throw std::invalid_argument("--reflection must be finite and in [0, 1]");
    }
    if (grid.nx < 8 || grid.nx > 2048) {
        throw std::invalid_argument("--nx must be in [8, 2048]");
    }
    if (grid.ny < 8 || grid.ny > 2048) {
        throw std::invalid_argument("--ny must be in [8, 2048]");
    }
    if (!std::isfinite(grid.lx) || grid.lx <= 0.0f) {
        throw std::invalid_argument("--lx must be finite and positive");
    }
    if (!std::isfinite(grid.lz) || grid.lz <= 0.0f) {
        throw std::invalid_argument("--lz must be finite and positive");
    }
    if (!std::isfinite(config.depth) || config.depth <= 0.0f) {
        throw std::invalid_argument("--depth must be finite and positive");
    }
    if (!std::isfinite(config.gravity) || config.gravity <= 0.0f) {
        throw std::invalid_argument("gravity must be finite and positive");
    }
    if (!std::isfinite(config.damping) || config.damping < 0.0f) {
        throw std::invalid_argument("--damping must be finite and nonnegative");
    }
    if (!std::isfinite(config.rain_rate) || config.rain_rate < 0.0f || config.rain_rate > 60.0f) {
        throw std::invalid_argument("--rain-rate must be finite and in [0, 60]");
    }
    if (!std::isfinite(config.impact_strength) || config.impact_strength < 0.0f ||
        config.impact_strength > 0.02f * config.depth) {
        throw std::invalid_argument("--strength must be finite and in [0, 0.02 * depth]");
    }
    const double dx = static_cast<double>(grid.lx) / grid.nx;
    const double dz = static_cast<double>(grid.lz) / grid.ny;
    if (!std::isfinite(config.impact_radius) || config.impact_radius <= 0.0f ||
        config.impact_radius < 2.0 * std::max(dx, dz) ||
        config.impact_radius > 0.25 * std::min(grid.lx, grid.lz)) {
        throw std::invalid_argument("--radius must be finite, positive, at least two cell widths, "
                                    "and at most one quarter of the shorter basin dimension");
    }
}

float adjust_depth(float current, int steps) {
    return static_cast<float>(std::max(0.25, static_cast<double>(current) + steps * 0.25));
}

float adjust_reflection(float current, int steps) {
    return static_cast<float>(std::clamp(static_cast<double>(current) + steps * 0.05, 0.0, 1.0));
}

double stable_dt(const Config& config) {
    validate(config);
    const double dx = static_cast<double>(config.grid.lx) / config.grid.nx;
    const double dz = static_cast<double>(config.grid.lz) / config.grid.ny;
    const double wave_speed = std::sqrt(static_cast<double>(config.gravity) * config.depth);
    return 0.8 / (wave_speed * std::sqrt(1.0 / (dx * dx) + 1.0 / (dz * dz)));
}

int substeps_per_quantum(const Config& config) {
    const double max_dt = stable_dt(config);
    const double required = std::ceil((1.0 / 120.0) / max_dt);
    if (required > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("configuration requires too many substeps per quantum");
    }
    int substeps = std::max(1, static_cast<int>(required));
    // The controller's float timestep can round above the double CFL bound.
    while (static_cast<float>((1.0 / 120.0) / substeps) > max_dt) {
        if (substeps == std::numeric_limits<int>::max()) {
            throw std::invalid_argument("configuration requires too many substeps per quantum");
        }
        ++substeps;
    }
    return substeps;
}

} // namespace water
