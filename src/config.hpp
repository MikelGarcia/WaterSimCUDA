#pragma once

namespace water {

struct Grid {
    int nx = 512;
    int ny = 512;
    float lx = 20.0f;
    float lz = 12.0f;
};

struct Config {
    Grid grid;
    float depth = 1.0f;
    float gravity = 9.81f;
    float damping = 0.15f;
    float impact_radius = 0.35f;
    float impact_strength = 0.01f;
    float rain_rate = 8.0f;
    unsigned seed = 1;
};

Config parse_config(int argc, const char* const* argv);
void validate(const Config& config);
double stable_dt(const Config& config);
int substeps_per_quantum(const Config& config);

} // namespace water
