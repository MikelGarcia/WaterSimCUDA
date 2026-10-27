#include "app_controller.hpp"
#include "timing_average.hpp"
#include "water_renderer.hpp"
#include <rlgl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

class GpuTimer {
public:
    GpuTimer() {
        try {
            for (auto& sample : samples_) {
                check(cudaEventCreate(&sample.begin), "create timing event");
                check(cudaEventCreate(&sample.end), "create timing event");
            }
        } catch (...) { release(); throw; }
    }
    ~GpuTimer() { release(); }
    void begin(cudaStream_t stream) {
        const double now = GetTime();
        for (auto& sample : samples_) {
            if (!sample.pending) continue;
            const auto result = cudaEventQuery(sample.end);
            if (result == cudaErrorNotReady) continue;
            check(result, "query simulation timing");
            check(cudaEventElapsedTime(&milliseconds_, sample.begin, sample.end),
                  "read simulation timing");
            average_.record(now, milliseconds_);
            sample.pending = false;
        }
        average_.update(now);
        current_ = nullptr;
        for (auto& sample : samples_) {
            if (sample.pending) continue;
            current_ = &sample;
            check(cudaEventRecord(sample.begin, stream), "start simulation timing");
            break;
        }
    }
    void end(cudaStream_t stream, bool advanced) {
        if (!current_) return;
        check(cudaEventRecord(current_->end, stream), "end simulation timing");
        current_->pending = advanced;
    }
    float milliseconds() const { return milliseconds_; }
    float average_milliseconds() const { return average_.milliseconds(); }

private:
    void release() noexcept {
        for (auto& sample : samples_) {
            for (auto event : {sample.begin, sample.end}) {
                if (!event) continue;
                const auto result = cudaEventDestroy(event);
                if (result != cudaSuccess)
                    std::fprintf(stderr, "destroy timing event: %s\n", cudaGetErrorString(result));
            }
        }
    }
    struct Sample { cudaEvent_t begin = nullptr, end = nullptr; bool pending = false; };
    std::array<Sample, 8> samples_{};
    Sample* current_ = nullptr;
    float milliseconds_ = 0;
    water::TimingAverage average_{GetTime()};
};

struct Window {
    Window() {
        SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
        InitWindow(1280, 800, "Water | CUDA shallow water");
        if (!IsWindowReady()) throw std::runtime_error("could not create OpenGL window");
        SetWindowMinSize(840, 560);
    }
    ~Window() { CloseWindow(); }
};

struct WaterShader {
    Shader value{};
    WaterShader() {
        const auto assets = std::filesystem::path(GetApplicationDirectory()) / "assets";
        const auto vertex = (assets / "water.vs").string();
        const auto fragment = (assets / "water.fs").string();
        value = LoadShader(vertex.c_str(), fragment.c_str());
        if (!IsShaderValid(value) || value.id == rlGetShaderIdDefault())
            throw std::runtime_error("could not load water shaders from " + assets.string());
    }
    ~WaterShader() { UnloadShader(value); }
};

void draw_basin(const water::Config& config) {
    const auto& g = config.grid;
    const Color stone{122, 132, 138, 255};
    const float thickness = 0.18f;
    const float wall_height = config.depth + 0.12f;
    const float wall_y = (0.12f - config.depth) / 2;
    DrawCube({0, -config.depth - thickness / 2, 0}, g.lx + 2 * thickness,
             thickness, g.lz + 2 * thickness, Color{91, 103, 112, 255});
    for (float sign : {-1.0f, 1.0f}) {
        DrawCube({sign * (g.lx + thickness) / 2, wall_y, 0}, thickness,
                 wall_height, g.lz + 2 * thickness, stone);
        DrawCube({0, wall_y, sign * (g.lz + thickness) / 2}, g.lx,
                 wall_height, thickness, stone);
    }
}

double validation_seconds(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    char* end = nullptr;
    const double seconds = std::strtod(value, &end);
    if (!*value || *end || !std::isfinite(seconds) || seconds < 0 || seconds > 3600)
        throw std::invalid_argument(std::string(name) + " must be seconds in [0, 3600]");
    return seconds;
}

void demo(const water::Config& config) {
    const double duration = validation_seconds("WATER_VALIDATE_SECONDS", 0);
    const double warmup = validation_seconds("WATER_VALIDATE_WARMUP", 10);
    const char* capture = std::getenv("WATER_VALIDATE_CAPTURE");
    Window window;
    water::select_gl_cuda_device();
    water::Simulation simulation(config);
    water::WaterRenderer renderer(config.grid);
    water::AppController app(simulation, config);
    WaterShader shader;
    GpuTimer timer;
    const int eye_location = GetShaderLocation(shader.value, "eyePosition");
    if (eye_location < 0) throw std::runtime_error("water shader is missing eyePosition");
    const Matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    Camera3D camera{};
    camera.target = {0, 0, 0};
    camera.up = {0, 1, 0};
    camera.fovy = 42;
    camera.projection = CAMERA_PERSPECTIVE;
    float yaw = 0.65f, pitch = 0.72f;
    const float basin_size = std::max(config.grid.lx, config.grid.lz);
    float distance = 1.65f * basin_size;
    bool paused = false, rain = true;
    float rate = config.rain_rate, strength = config.impact_strength;
    float damping = config.damping;
    SetTargetFPS(60);
    const double started = GetTime();
    double previous = started;
    std::vector<double> frame_ms;
    while (!WindowShouldClose()) {
        const double now = GetTime();
        const double elapsed = now - previous;
        previous = now;
        if (duration > 0 && now - started >= warmup + duration) break;
        if (duration > 0 && now - started >= warmup) frame_ms.push_back(elapsed * 1000);

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_R)) app.reset();
        if (IsKeyPressed(KEY_T)) rain = !rain;
        if (IsKeyPressed(KEY_UP)) rate = std::min(60.0f, rate + 1);
        if (IsKeyPressed(KEY_DOWN)) rate = std::max(0.0f, rate - 1);
        if (IsKeyPressed(KEY_RIGHT)) strength = std::min(0.02f * config.depth, strength + 0.001f);
        if (IsKeyPressed(KEY_LEFT)) strength = std::max(0.0f, strength - 0.001f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) damping = std::clamp(damping + 0.05f, 0.0f, 5.0f);
        if (IsKeyPressed(KEY_LEFT_BRACKET)) damping = std::clamp(damping - 0.05f, 0.0f, 5.0f);
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            const auto delta = GetMouseDelta();
            yaw -= delta.x * 0.006f;
            pitch = std::clamp(pitch + delta.y * 0.006f, 0.15f, 1.45f);
        }
        distance = std::clamp(distance * std::exp(-GetMouseWheelMove() * 0.1f),
                              0.4f * basin_size, 4.0f * basin_size);
        camera.position = {distance * std::cos(pitch) * std::sin(yaw),
                           distance * std::sin(pitch),
                           distance * std::cos(pitch) * std::cos(yaw)};
        const water::SpawnView spawn_view{camera,
            static_cast<float>(GetScreenWidth()) / std::max(1, GetScreenHeight())};
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && !IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            const auto hit = water::pick_water(GetScreenToWorldRay(GetMousePosition(), camera), config.grid);
            if (hit) app.spawn(hit->x, hit->y, spawn_view);
        }
        timer.begin(simulation.stream());
        const double old_time = app.sim_time();
        app.frame(elapsed, paused, rain, rate, strength, damping, spawn_view);
        timer.end(simulation.stream(), app.sim_time() != old_time);
        renderer.update(simulation);
        SetShaderValue(shader.value, eye_location, &camera.position, SHADER_UNIFORM_VEC3);

        BeginDrawing();
        ClearBackground(Color{225, 231, 234, 255});
        BeginMode3D(camera);
        draw_basin(config);
        renderer.draw(shader.value, identity);
        for (const auto& drop : app.drops()) {
            const float height = static_cast<float>(water::drop_y(drop, app.sim_time())) +
                                 water::drop_radius;
            DrawSphereEx({drop.x - config.grid.lx / 2, height, drop.z - config.grid.lz / 2},
                         water::drop_radius, 6, 8, Color{34, 134, 205, 255});
        }
        EndMode3D();
        DrawRectangle(20, 20, 780, 102, Fade(Color{246, 249, 250, 255}, 0.94f));
        DrawText(paused ? "WATER  /  PAUSED" : "WATER", 36, 32, 22, Color{27, 52, 67, 255});
        const Color status_color{45, 65, 76, 255};
        DrawText(TextFormat("%d FPS", GetFPS()), 36, 62, 16, status_color);
        DrawText(TextFormat("GPU %.2f ms", timer.average_milliseconds()),
                 126, 62, 16, status_color);
        DrawText(TextFormat("Rain %s %.0f/s", rain ? "on" : "off", rate),
                 286, 62, 16, status_color);
        DrawText(TextFormat("Strength %.3f m", strength),
                 456, 62, 16, status_color);
        DrawText(TextFormat("Damping %.2f/s", damping),
                 650, 62, 16, status_color);
        DrawText("Click: drop   Right drag: orbit   Wheel: zoom   Space: pause   R: reset   T: rain",
                 36, 88, 16, Color{65, 83, 93, 255});
        DrawText("Up / Down: rain    Left / Right: strength    [ / ]: damping    Esc: close",
                 28, GetScreenHeight() - 32, 16, Color{45, 65, 76, 255});
        EndDrawing();
    }
    if (capture && *capture) {
        const Image image = LoadImageFromScreen();
        const bool saved = ExportImage(image, capture);
        UnloadImage(image);
        if (!saved) throw std::runtime_error(std::string("could not save validation capture: ") + capture);
    }
    if (!frame_ms.empty()) {
        const double mean = std::accumulate(frame_ms.begin(), frame_ms.end(), 0.0) / frame_ms.size();
        std::sort(frame_ms.begin(), frame_ms.end());
        const double p95 = frame_ms[static_cast<std::size_t>(0.95 * (frame_ms.size() - 1))];
        std::printf("VALIDATION %dx%d warmup=%.1fs samples=%zu mean=%.3fms p95=%.3fms fps=%.2f gpu_last=%.3fms\n",
                    config.grid.nx, config.grid.ny, warmup, frame_ms.size(), mean, p95, 1000 / mean,
                    timer.milliseconds());
    }
}

void help() {
    std::puts("WaterSimCUDA: GPU shallow water basin\n"
              "Usage: water_demo [options]\n"
              "  --nx 512 --ny 512        grid cells per axis [8, 2048]\n"
              "  --lx 20 --lz 12          basin dimensions (m)\n"
              "  --depth 1               mean depth (m)\n"
              "  --damping 0.15          drag (/s); interactive range [0, 5]\n"
              "  --radius 0.35           impact radius (m), >=2 cell widths\n"
              "  --strength 0.01         impact displacement (m), [0, 0.02 * depth]\n"
              "  --rain-rate 8           random drops per second [0, 60]\n"
              "  --seed 1                random seed (unsigned integer)\n"
              "Space pause | R reset | T rain | arrows up/down rain, left/right strength\n"
              "[ / ] damping | left click drop | right drag orbit | wheel zoom | Esc close");
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") { help(); return 0; }
        demo(water::parse_config(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "water_demo: %s\n", error.what());
        return 1;
    }
}
