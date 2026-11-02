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
    const double c = std::cos(boat.yaw), s = std::sin(boat.yaw);
    const double velocity = std::hypot(boat.vx, boat.vz);
    // Keep shell radii relative to the pressure emitter at hull positions 8/2.
    const double offset = velocity > 0 ? .3 * water::boat_geometry.length *
        (boat.vx * c + boat.vz * s) / velocity : 0;
    const double origin_x = boat.x + offset * c, origin_z = boat.z + offset * s;
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
                    const double height = sample(state, g, origin_x - r * std::cos(a),
                                                 origin_z + (side ? 1 : -1) * r * std::sin(a));
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
        // .68 m encloses the shifted pressure ellipse plus its face-gradient halo.
        // The complete future support union is checked at every substep against the fixed window.
        clearance = std::min(clearance, std::min(double(before.x), double(boat.x)) -
                             (.5 + .3 * water::boat_geometry.length) - 7.375);
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

#ifdef WATER_TIMING_DESKTOP
#include "timeline.hpp"
#include "app_controller.hpp"
#include "water_renderer.hpp"
#include "boat_renderer.hpp"
#include "scene_renderer.hpp"

TEST_CASE("wake_physics", continuity_timing_and_memory_sample) {
    using Clock = std::chrono::steady_clock;
    std::filesystem::create_directories("wake-physics");
    std::ofstream csv("wake-physics/timing-memory.csv");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << "nx,rain,substep,simulation_ms,fft_ms\n";
    SetTraceLogLevel(LOG_WARNING);
    InitWindow(1280, 800, "Water validation timing");
    CHECK(IsWindowReady());
    struct WindowCleanup { ~WindowCleanup() { CloseWindow(); } } window_cleanup;
    water::select_gl_cuda_device();
    // Warm driver/JIT bookkeeping before taking free-memory deltas.
    { water::Simulation warmup(water::Config{}); warmup.step(1.f / 120, 2.5f);
      CHECK(cudaStreamSynchronize(warmup.stream()) == cudaSuccess); }
    for (int nx : {512, 2048}) {
        for (float rate : {30.f, 120.f}) {
            if (nx == 2048 && rate == 120) continue;
            water::Config c;
            c.grid.nx = c.grid.ny = nx;
            c.rain_rate = rate;
            std::size_t free_before = 0, total = 0, free_sim = 0, free_render = 0;
            CHECK(cudaMemGetInfo(&free_before, &total) == cudaSuccess);
            const auto started = Clock::now();
            water::Simulation simulation(c);
            CHECK(cudaStreamSynchronize(simulation.stream()) == cudaSuccess);
            const double startup_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            CHECK(cudaMemGetInfo(&free_sim, &total) == cudaSuccess);
            const auto resources = water::fft_resources(simulation);
            water::WaterRenderer renderer(c.grid);
            water::BoatRenderer boat(c.grid);
            water::SceneRenderer scene(c);
            CHECK(scene.resize(1280, 800));
            renderer.update(simulation);
            boat.update(simulation);
            const float distance = 1.65f * std::max(c.grid.lx, c.grid.lz);
            Camera3D camera{};
            camera.position = {distance * std::cos(.72f) * std::sin(.65f), distance * std::sin(.72f),
                               distance * std::cos(.72f) * std::cos(.65f)};
            camera.up = {0, 1, 0}; camera.fovy = 42; camera.projection = CAMERA_PERSPECTIVE;
            const water::SpawnView view{camera, 1280.f / 800};
            water::DropScheduler scheduler(c);
            BeginDrawing(); scene.draw(camera, renderer, boat, scheduler.active(), 0, c.reflection); EndDrawing();
            CHECK(cudaStreamSynchronize(simulation.stream()) == cudaSuccess);
            CHECK(cudaMemGetInfo(&free_render, &total) == cudaSuccess);
            cudaEvent_t begin = nullptr, end = nullptr, fft_begin = nullptr, fft_end = nullptr;
            CHECK(cudaEventCreate(&begin) == cudaSuccess);
            CHECK(cudaEventCreate(&end) == cudaSuccess);
            CHECK(cudaEventCreate(&fft_begin) == cudaSuccess);
            CHECK(cudaEventCreate(&fft_end) == cudaSuccess);
            struct Events {
                water::Simulation& simulation;
                cudaEvent_t begin, end, fft_begin, fft_end;
                ~Events() {
                    water::set_fft_timing_events(simulation, nullptr, nullptr);
                    cudaEventDestroy(begin); cudaEventDestroy(end);
                    cudaEventDestroy(fft_begin); cudaEventDestroy(fft_end);
                }
            } events{simulation, begin, end, fft_begin, fft_end};
            water::set_fft_timing_events(simulation, fft_begin, fft_end);
            const int subdivisions = water::substeps_per_quantum(c);
            const float dt = float(water::FixedClock::quantum / subdivisions);
            const int warmup_ticks = nx == 512 ? 1200 : 120;
            const int sample_ticks = nx == 512 ? 7200 : 600;
            std::vector<double> gpu, fft;
            double peak = 0, mean_max = 0;
            const auto timed_started = Clock::now();
            for (int tick = 0; tick < warmup_ticks + sample_ticks; ++tick) {
                const auto phase = tick % (32 * 120);
                water::BoatControl control;
                if (phase < 12 * 120) control = {1, 0};
                else if (phase < 20 * 120) control = {1, .55f};
                else if (phase < 28 * 120) control = {-1, -.65f};
                else control = {1, -1};
                const auto impacts = scheduler.tick((tick + 1) * water::FixedClock::quantum, true,
                                                    rate, c.impact_strength, view);
                for (std::size_t first = 0; first < impacts.size(); first += 64)
                    simulation.apply_impacts(impacts.data() + first, std::min<std::size_t>(64, impacts.size() - first));
                for (int substep = 0; substep < subdivisions; ++substep) {
                    CHECK(cudaEventRecord(begin, simulation.stream()) == cudaSuccess);
                    simulation.step(dt, c.damping, control);
                    CHECK(cudaEventRecord(end, simulation.stream()) == cudaSuccess);
                    // Collect this substep before the borrowed FFT events can be recorded again.
                    CHECK(cudaEventSynchronize(end) == cudaSuccess);
                    float gpu_ms = 0, fft_ms = 0;
                    CHECK(cudaEventElapsedTime(&gpu_ms, begin, end) == cudaSuccess);
                    CHECK(cudaEventElapsedTime(&fft_ms, fft_begin, fft_end) == cudaSuccess);
                    if (tick >= warmup_ticks) {
                        gpu.push_back(gpu_ms); fft.push_back(fft_ms);
                        csv << nx << ',' << rate << ',' << (tick - warmup_ticks) * subdivisions + substep
                            << ',' << gpu_ms << ',' << fft_ms << '\n';
                    }
                }
                if ((tick + 1) % 120 == 0) {
                    const auto state = water::download(simulation);
                    double mean = 0;
                    for (float value : state.eta) {
                        CHECK(std::isfinite(value));
                        peak = std::max(peak, std::abs(double(value))); mean += value;
                    }
                    for (const auto* field : {&state.u, &state.v})
                        for (float value : *field) CHECK(std::isfinite(value));
                    mean_max = std::max(mean_max, std::abs(mean / state.eta.size()));
                }
            }
            const double elapsed_s = std::chrono::duration<double>(Clock::now() - timed_started).count();
            {
                water::set_fft_timing_events(simulation, nullptr, nullptr);
                water::AppController app(simulation, c);
                app.reset();
                SetTargetFPS(60);
                std::vector<double> frames;
                const auto realtime_start = Clock::now();
                auto previous = realtime_start;
                double offered = 0;
                int capped_frames = 0, maximum_quanta = 0;
                while (std::chrono::duration<double>(Clock::now() - realtime_start).count() < 6) {
                    const auto now = Clock::now();
                    const double seconds = std::chrono::duration<double>(now - previous).count();
                    previous = now;
                    offered += seconds;
                    if (seconds >= 1. / 120 * 9) ++capped_frames;
                    const double old_time = app.sim_time();
                    app.frame(seconds, false, true, rate, c.impact_strength, c.damping, view, {1, 0});
                    maximum_quanta = std::max(maximum_quanta, int(std::llround((app.sim_time() - old_time) * 120)));
                    renderer.update(simulation); boat.update(simulation);
                    BeginDrawing(); scene.draw(camera, renderer, boat, app.drops(), app.sim_time(), c.reflection); EndDrawing();
                    if (std::chrono::duration<double>(now - realtime_start).count() >= 1)
                        frames.push_back(seconds * 1000);
                }
                const double frame_mean = std::accumulate(frames.begin(), frames.end(), 0.) / frames.size();
                std::sort(frames.begin(), frames.end());
                std::cout << "PRACTICAL nx=" << nx << " rain=" << rate << " frame_mean_ms=" << frame_mean
                          << " frame_p95_ms=" << frames[std::size_t(.95 * (frames.size() - 1))]
                          << " samples=" << frames.size() << " offered_wall_s=" << offered
                          << " simulated_s=" << app.sim_time() << " discarded_plus_fraction_s=" << offered - app.sim_time()
                          << " frames_over_9_quanta=" << capped_frames << " maximum_quanta=" << maximum_quanta << std::endl;
            }
            CHECK(peak < .1 * c.depth);
            CHECK(mean_max < 1e-5);
            const double gpu_mean = std::accumulate(gpu.begin(), gpu.end(), 0.) / gpu.size();
            const double fft_mean = std::accumulate(fft.begin(), fft.end(), 0.) / fft.size();
            std::sort(gpu.begin(), gpu.end()); std::sort(fft.begin(), fft.end());
            std::size_t free_after = 0;
            CHECK(cudaMemGetInfo(&free_after, &total) == cudaSuccess);
            const std::size_t cells = std::size_t(nx) * nx;
            const std::size_t state_bytes = (cells + 2 * std::size_t(nx + 1) * nx) * sizeof(float);
            std::cout << "TIMING nx=" << nx << " rain=" << rate << " substeps=" << subdivisions
                      << " samples=" << gpu.size() << " gpu_substep_mean_ms=" << gpu_mean
                      << " gpu_substep_p95_ms=" << gpu[std::size_t(.95 * (gpu.size() - 1))]
                      << " fft_mean_ms=" << fft_mean << " fft_p95_ms=" << fft[std::size_t(.95 * (fft.size() - 1))]
                      << " fft_fraction=" << fft_mean / gpu_mean << " startup_ms=" << startup_ms
                      << " sim_delta_bytes=" << free_before - free_sim
                      << " render_delta_bytes=" << free_sim - free_render
                      << " peak_observed_delta_bytes=" << free_before - std::min(free_render, free_after)
                      << " headroom_bytes=" << std::min(free_render, free_after)
                      << " workspace_bytes=" << resources.workspace_bytes << " mirrored_real_bytes=" << resources.real_bytes
                      << " spectrum_bytes=" << resources.spectral_bytes << " two_weight_banks_bytes=" << resources.weights_bytes
                      << " state_bytes=" << state_bytes << " water_vbo_bytes=" << 12 * cells * sizeof(water::Vertex)
                      << " wall_sample_seconds=" << elapsed_s << " simulated_seconds=" << (warmup_ticks + sample_ticks) / 120.
                      << " peak_eta=" << peak << " max_abs_mean=" << mean_max << std::endl;
        }
    }
}
#endif
