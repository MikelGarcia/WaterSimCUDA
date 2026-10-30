#include "validation.hpp"
#include "dispersion_reference.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
constexpr double pi = 3.14159265358979323846;
double sample(const water::Snapshot &state, const water::Grid &g, double x, double z) {
    x = x * g.nx / g.lx - .5;
    z = z * g.ny / g.lz - .5;
    const int ix = int(std::floor(x)), iz = int(std::floor(z));
    CHECK(ix >= 0 && ix + 1 < g.nx && iz >= 0 && iz + 1 < g.ny);
    const double fx = x - ix, fz = z - iz;
    return (1 - fz) * ((1 - fx) * state.eta[iz * g.nx + ix] + fx * state.eta[iz * g.nx + ix + 1]) +
           fz * ((1 - fx) * state.eta[(iz + 1) * g.nx + ix] + fx * state.eta[(iz + 1) * g.nx + ix + 1]);
}
std::array<double, 3> spectra(const water::Snapshot &state, const water::Grid &g, water::BoatState boat,
                              std::ofstream &csv, std::ofstream &radial_csv, const char *dataset, float speed,
                              double time) {
    std::array<double, 3> peaks{};
    for (int shell = 0; shell < 3; ++shell) {
        double maximum = -1;
        std::array<double, 281> strengths{};
        for (int bin = 0; bin <= 280; ++bin) {
            const double angle = bin * .25, a = angle * pi / 180;
            double square[2]{};
            for (int radial = 0; radial < 80; ++radial) {
                const double r = 1.2 + .6 * shell + .6 * (radial + .5) / 80;
                double heights[2]{};
                for (int side = 0; side < 2; ++side) {
                    const double height = sample(state, g, boat.x - r * std::cos(a),
                                                 boat.z + (side ? 1 : -1) * r * std::sin(a));
                    square[side] += height * height / 80;
                    heights[side] = height;
                }
                radial_csv << dataset << ',' << speed << ',' << time << ',' << r << ',' << angle << ','
                           << heights[0] << ',' << heights[1] << '\n';
            }
            const double rms = std::sqrt(.5 * (square[0] + square[1]));
            strengths[bin] = rms;
            csv << dataset << ',' << speed << ',' << time << ',' << shell << ',' << angle << ','
                << std::sqrt(square[0]) << ',' << std::sqrt(square[1]) << '\n';
            if (rms > maximum) {
                maximum = rms;
                peaks[shell] = angle;
            }
        }
        std::vector<std::pair<double, double>> candidates;
        for (int bin = 1; bin < 280; ++bin) {
            if (strengths[bin] >= strengths[bin - 1] && strengths[bin] > strengths[bin + 1])
                candidates.push_back({strengths[bin], bin * .25});
        }
        std::sort(candidates.rbegin(), candidates.rend());
        std::cout << "COMPETING dataset=" << dataset << " speed=" << speed << " time=" << time
                  << " shell=" << shell;
        for (size_t i = 0; i < std::min(size_t{3}, candidates.size()); ++i)
            std::cout << " angle/rms=" << candidates[i].second << '/' << candidates[i].first;
        std::cout << std::endl;
        std::cout << "RIDGE dataset=" << dataset << " U=" << speed << " t=" << time << " shell=" << shell
                  << " angle=" << peaks[shell] << " rms=" << maximum << std::endl;
    }
    return peaks;
}
} // namespace

TEST_CASE("wake_physics", dominant_ridge_narrows_at_resolved_high_speed) {
    std::filesystem::create_directories("wake-physics");
    std::ofstream csv("wake-physics/angular.csv");
    std::ofstream radial_csv("wake-physics/radial.csv");
    radial_csv << "dataset,speed,time,radius,angle,left_eta,right_eta\n";
    radial_csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << "dataset,speed,time,shell,angle,left_rms,right_rms\n";
    std::ofstream trajectory("wake-physics/trajectory.csv");
    trajectory << "dataset,speed,time,x,z,vx,vz\n";
    constexpr float dt = 1.f / 120;
    std::array<std::array<std::vector<double>, 4>, 4> angles;
    const float speeds[]{.75f, 1.25f, 1.5f, 1.875f};
    const char *names[]{"base", "walls", "refined", "drag"};
    for (int dataset = 0; dataset < 4; ++dataset)
        for (int run = 0; run < 4; ++run) {
            if (dataset && run != 0 && run != 3)
                continue;
            water::Config c;
            c.grid = {1024, 512, 60, 30};
            c.impact_radius = .15f;
            c.rain_rate = 0;
            if (dataset == 1)
                c.grid = {2048, 1024, 120, 60};
            if (dataset == 2)
                c.grid = {2048, 2048, 60, 60};
            const auto started = std::chrono::steady_clock::now();
            std::size_t free_before = 0, total = 0, free_after = 0;
            CHECK(cudaMemGetInfo(&free_before, &total) == cudaSuccess);
            water::Simulation simulation(c);
            CHECK(cudaMemGetInfo(&free_after, &total) == cudaSuccess);
            std::cout << "MEMORY dataset=" << names[dataset]
                      << " simulation_allocation_bytes=" << free_before - free_after << std::endl;
            const double start = c.grid.lx / 2;
            for (int step = 0; step < 960; ++step) {
                const double t = step * double(dt);
                const float u = speeds[run] * float(std::min(1.0, (t + .5 * dt) / .5));
                const double distance = speeds[run] * (t < .5 ? t * t : t - .25);
                water::upload_boat(simulation, {float(start + distance), c.grid.lz / 2, 0, u, 0, 0});
                simulation.step(dt, dataset == 3 ? .15f : 0);
                const auto b = water::download_boat(simulation);
                trajectory << names[dataset] << ',' << speeds[run] << ',' << (step + 1) * dt << ',' << b.x
                           << ',' << b.z << ',' << b.vx << ',' << b.vz << '\n';
                if ((step + 1) % 120 == 0 && step >= 719) {
                    const auto peaks = spectra(water::download(simulation), c.grid, b, csv, radial_csv,
                                               names[dataset], speeds[run], (step + 1) * dt);
                    angles[dataset][run].insert(angles[dataset][run].end(), peaks.begin(), peaks.end());
                }
            }
            std::cout << "RIDGE dataset=" << names[dataset] << " run_seconds="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                      << std::endl;
        }
    for (int shell = 0; shell < 3; ++shell) {
        double minimum = 100, maximum = -100;
        for (int dataset = 0; dataset < 4; ++dataset)
            for (int time = 0; time < 3; ++time) {
                const double difference =
                    angles[dataset][0][3 * time + shell] - angles[dataset][3][3 * time + shell];
                minimum = std::min(minimum, difference);
                maximum = std::max(maximum, difference);
            }
        std::cout << "NARROWING shell=" << shell << " difference_range=" << minimum << ',' << maximum
                  << " bin_uncertainty=.5deg" << std::endl;
        double refinement = 0, reflection = 0;
        for (int run : {0, 3})
            for (int time = 0; time < 3; ++time) {
                const int i = 3 * time + shell;
                refinement = std::max(refinement, std::abs(angles[1][run][i] - angles[2][run][i]));
                reflection = std::max(reflection, std::abs(angles[0][run][i] - angles[1][run][i]));
            }
        const double uncertainty = .5 + 2 * (refinement + reflection);
        std::cout << "UNCERTAINTY shell=" << shell << " refined_angle_change=" << refinement
                  << " reflected_angle_change=" << reflection << " combined_deg=" << uncertainty << std::endl;
        CHECK(minimum > uncertainty);
    }
}

namespace {
std::array<int, 2> displacement(const water::Snapshot &initial, const water::Snapshot &final,
                                const water::Grid &g, bool envelope) {
    auto value = [&](const water::Snapshot &state, int x, int z) {
        if (!envelope)
            return double(state.eta[z * g.nx + x]);
        double square = 0;
        for (int j = -3; j <= 3; ++j)
            for (int i = -3; i <= 3; ++i) {
                const double h = state.eta[(z + j) * g.nx + x + i];
                square += h * h / 49;
            }
        return std::sqrt(square);
    };
    double best = -1;
    std::array<int, 2> shift{};
    for (int sz = -5; sz <= 5; ++sz)
        for (int sx = -5; sx <= 5; ++sx) {
            double ab = 0, aa = 0, bb = 0, sa = 0, sb = 0;
            int count = 0;
            for (int z = 28; z < 39; ++z)
                for (int x = 40; x < 59; ++x) {
                    const double a = value(initial, x, z), b = value(final, x + sx, z + sz);
                    ab += a * b;
                    aa += a * a;
                    bb += b * b;
                    sa += a;
                    sb += b;
                    ++count;
                }
            const double correlation =
                (ab - sa * sb / count) / std::sqrt((aa - sa * sa / count) * (bb - sb * sb / count));
            if (correlation > best) {
                best = correlation;
                shift = {sx, sz};
            }
        }
    std::cout << "HISTORY envelope=" << envelope << " shift_cells=" << shift[0] << ',' << shift[1]
              << " correlation=" << best << std::endl;
    return shift;
}
} // namespace

TEST_CASE("wake_physics", emitted_history_moves_in_world_coordinates) {
    std::filesystem::create_directories("wake-physics");
    std::ofstream csv("wake-physics/released.csv");
    csv << "controller,history,time,old_window_rms,reference_error_rms,change_rms\n";
    water::Config c;
    c.grid = {96, 64, 12, 8};
    c.impact_radius = .3f;
    c.rain_rate = 0;
    constexpr float dt = .01f;
    for (bool controller : {false, true}) {
        water::Simulation emitted(c);
        water::upload_boat(emitted, {3, 4, 0, 1.5f, 0, 0});
        for (int step = 0; step < 400; ++step) {
            if (!controller)
                water::upload_boat(emitted, {3 + 1.5f * step * dt, 4, 0, 1.5f, 0, 0});
            emitted.step(dt, 0, controller ? water::BoatControl{.7f, 0} : water::BoatControl{});
        }
        const auto initial = water::download(emitted);
        const auto released_boat = water::download_boat(emitted);
        auto expected = initial;
        std::vector<water::Snapshot> reference_frames;
        for (int step = 0; step < 60; ++step) {
            expected = reference::step(c, expected, dt, 0);
            if ((step + 1) % 20 == 0)
                reference_frames.push_back(expected);
        }
        for (int history = 0; history < 5; ++history) {
            water::Simulation simulation(c);
            water::upload(simulation, initial);
            auto boat = released_boat;
            if (history == 0)
                boat.vx = boat.vz = boat.yaw_rate = 0;
            if (history == 4)
                boat = {11.69f, 4, 0, 1.5f, 0, 0};
            water::upload_boat(simulation, boat);
            for (int step = 0; step < 60; ++step) {
                water::BoatControl control{};
                if (history == 2)
                    control = {.7f, 1};
                if (history == 3 || history == 4)
                    control = {-1, -1};
                if (!controller && history == 2) {
                    const float angle = (step + 1) * dt;
                    water::upload_boat(simulation, {released_boat.x + 1.5f * step * dt, 4, angle,
                                                    1.5f * std::cos(angle), 1.5f * std::sin(angle), 0});
                }
                simulation.step(dt, 0, control);
                if ((step + 1) % 20)
                    continue;
                const auto state = water::download(simulation);
                const auto &ref = reference_frames[(step + 1) / 20 - 1];
                double energy = 0, error = 0, change = 0, rigid_error = 0;
                int count = 0;
                const auto actual_boat = water::download_boat(simulation);
                const double yaw = actual_boat.yaw - released_boat.yaw;
                // Fixed world window, x=5..7.375 and z=3.5..4.875; all later hull supports are x>8.
                for (int z = 28; z < 39; ++z)
                    for (int x = 40; x < 59; ++x) {
                        const int i = z * c.grid.nx + x;
                        energy += double(ref.eta[i]) * ref.eta[i];
                        error += std::pow(double(state.eta[i]) - ref.eta[i], 2);
                        change += std::pow(double(state.eta[i]) - initial.eta[i], 2);
                        ++count;
                        if (history > 0 && history < 4) {
                            const double px = (x + .5) * c.grid.lx / c.grid.nx - actual_boat.x;
                            const double pz = (z + .5) * c.grid.lz / c.grid.ny - actual_boat.z;
                            const double rx = released_boat.x + px * std::cos(yaw) + pz * std::sin(yaw);
                            const double rz = released_boat.z - px * std::sin(yaw) + pz * std::cos(yaw);
                            rigid_error += std::pow(double(state.eta[i]) - sample(ref, c.grid, rx, rz), 2);
                        }
                    }
                std::cout << "HISTORY controller=" << controller << " maneuver=" << history
                          << " t=" << (step + 1) * dt << " old_rms=" << std::sqrt(energy / count)
                          << " relative_reference_error=" << std::sqrt(error / energy)
                          << " change_rms=" << std::sqrt(change / count) << std::endl;
                csv << controller << ',' << history << ',' << (step + 1) * dt << ','
                    << std::sqrt(energy / count) << ',' << std::sqrt(error / count) << ','
                    << std::sqrt(change / count) << '\n';
                CHECK(energy / count > 1e-12);
                CHECK(change / energy > .01);
                CHECK(std::sqrt(error / energy) < (history == 0 ? .0001 : .02));
                if (history > 0 && history < 4) {
                    std::cout << "HISTORY rigid_relative_error=" << std::sqrt(rigid_error / energy)
                              << std::endl;
                    CHECK(rigid_error > 100 * error);
                    CHECK(rigid_error / energy > .01);
                }
                if (history == 0) {
                    double maximum_error = 0, scale = 0;
                    for (size_t i = 0; i < state.eta.size(); ++i) {
                        maximum_error = std::max(maximum_error, std::abs(double(state.eta[i]) - ref.eta[i]));
                        scale = std::max(scale, std::abs(double(ref.eta[i])));
                    }
                    CHECK(maximum_error < 2e-7 + 3e-5 * scale);
                }
                if (history == 0 && step == 19) {
                    for (bool envelope : {false, true}) {
                        const auto shift = displacement(initial, state, c.grid, envelope);
                        CHECK(shift == displacement(initial, ref, c.grid, envelope));
                        CHECK(shift[0] != 0 || shift[1] != 0);
                        CHECK(std::abs(shift[0]) < 5 && std::abs(shift[1]) < 5);
                    }
                }
            }
            if (history == 3) {
                for (int step = 0; step < 140; ++step)
                    simulation.step(dt, 0, {-1, 0});
                const auto reverse = water::download_boat(simulation);
                std::cout << "HISTORY reversal_vx=" << reverse.vx << " x=" << reverse.x << std::endl;
                CHECK(reverse.vx < -.1f);
            }
        }
    }
}

TEST_CASE("wake_reverse", released_history_survives_actual_reversal) {
    std::filesystem::create_directories("wake-physics");
    std::ofstream csv("wake-physics/reversal.csv");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << "time,x,z,yaw,vx,vz,longitudinal_speed,min_support_clearance,old_rms,incoming_rms,"
           "reference_error_rms,change_rms,frozen_error_rms,rigid_error_rms\n";
    water::Config c;
    c.grid = {96, 64, 12, 8};
    c.impact_radius = .3f;
    c.rain_rate = 0;
    constexpr float dt = .01f;
    water::Simulation simulation(c), incoming(c);
    water::upload_boat(simulation, {3, 4, 0, 1.5f, 0, 0});
    for (int step = 0; step < 400; ++step) {
        water::upload_boat(simulation, {3 + 1.5f * step * dt, 4, 0, 1.5f, 0, 0});
        simulation.step(dt, 0);
    }
    const auto initial = water::download(simulation);
    const auto released = water::download_boat(simulation);
    auto reference_state = initial;
    double clearance = 100;
    int negative_samples = 0;
    for (int step = 0; step < 240; ++step) {
        const auto before = water::download_boat(simulation);
        // Identical boat inputs isolate the linear response to later pressure forcing.
        water::upload_boat(incoming, before);
        simulation.step(dt, 0, {-1, 0});
        incoming.step(dt, 0, {-1, 0});
        reference_state = reference::step(c, reference_state, dt, 0);
        const auto boat = water::download_boat(simulation);
        // .5 m encloses the rotated pressure ellipse plus its face-gradient halo.
        // The complete future support union is checked at every substep against the fixed window.
        clearance = std::min(clearance, std::min(double(before.x), double(boat.x)) - .5 - 7.375);
        CHECK(clearance > 0);
        if ((step + 1) % 20)
            continue;
        const auto state = water::download(simulation);
        const auto forcing = water::download(incoming);
        auto old = state;
        for (size_t i = 0; i < old.eta.size(); ++i)
            old.eta[i] -= forcing.eta[i];
        const double longitudinal = boat.vx * std::cos(boat.yaw) + boat.vz * std::sin(boat.yaw);
        double energy = 0, contamination = 0, error = 0, change = 0, frozen = 0, rigid = 0;
        int count = 0;
        for (int z = 28; z < 39; ++z)
            for (int x = 40; x < 59; ++x) {
                const int i = z * c.grid.nx + x;
                energy += double(old.eta[i]) * old.eta[i];
                contamination += double(forcing.eta[i]) * forcing.eta[i];
                error += std::pow(double(old.eta[i]) - reference_state.eta[i], 2);
                change += std::pow(double(old.eta[i]) - initial.eta[i], 2);
                frozen += std::pow(double(reference_state.eta[i]) - initial.eta[i], 2);
                const double rx = (x + .5) * c.grid.lx / c.grid.nx - boat.x + released.x;
                const double rz = (z + .5) * c.grid.lz / c.grid.ny - boat.z + released.z;
                rigid += std::pow(double(old.eta[i]) - sample(reference_state, c.grid, rx, rz), 2);
                ++count;
            }
        csv << (step + 1) * dt << ',' << boat.x << ',' << boat.z << ',' << boat.yaw << ','
            << boat.vx << ',' << boat.vz << ',' << longitudinal << ',' << clearance << ','
            << std::sqrt(energy / count) << ',' << std::sqrt(contamination / count) << ','
            << std::sqrt(error / count) << ',' << std::sqrt(change / count) << ','
            << std::sqrt(frozen / count) << ',' << std::sqrt(rigid / count) << '\n';
        std::cout << "REVERSAL t=" << (step + 1) * dt << " longitudinal=" << longitudinal
                  << " old_rms=" << std::sqrt(energy / count)
                  << " incoming_relative=" << std::sqrt(contamination / energy)
                  << " reference_relative=" << std::sqrt(error / energy)
                  << " frozen_relative=" << std::sqrt(frozen / energy)
                  << " rigid_relative=" << std::sqrt(rigid / energy) << std::endl;
        if (longitudinal >= -.1)
            continue;
        ++negative_samples;
        CHECK(energy / count > 1e-12);
        CHECK(change / energy > .01);
        CHECK(std::sqrt(error / energy) < .0001);
        CHECK(frozen > 100 * error);
        CHECK(frozen / energy > .01);
        CHECK(rigid > 100 * error);
        CHECK(rigid / energy > .01);
    }
    CHECK(negative_samples >= 3);
}
