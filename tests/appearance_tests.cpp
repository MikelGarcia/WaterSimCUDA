#include "test_support.hpp"
#include "appearance_validation.hpp"
#include <memory>
#include "boat_renderer.hpp"
#include "water_renderer.hpp"
#include "timeline.hpp"
#include "app_controller.hpp"
#include "validation.hpp"
#include <external/glad.h>
#include <rlgl.h>
#include <array>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace {

struct Context {
    std::unique_ptr<water::Simulation> boat_simulation;
    std::unique_ptr<water::BoatRenderer> boat_renderer;
    const water::BoatRenderer& boat() {
        if (!boat_renderer) {
            water::Config config; config.grid={8,12,4,6}; config.impact_radius=1;
            boat_simulation=std::make_unique<water::Simulation>(config);
            // Keep the legacy basin/grass landmarks clear of the added boat.
            water::upload_boat(*boat_simulation,{.4f,.4f,0,0,0,0});
            boat_renderer=std::make_unique<water::BoatRenderer>(config.grid);
            boat_renderer->update(*boat_simulation);
        }
        return *boat_renderer;
    }
    Context(int width = 840, int height = 560) {
        SetTraceLogLevel(LOG_WARNING);
        SetConfigFlags(FLAG_WINDOW_HIDDEN);
        InitWindow(width, height, "Water appearance validation");
        if (!IsWindowReady()) throw std::runtime_error("hidden OpenGL context could not be created");
        water::select_gl_cuda_device();
        std::cout << "GL renderer: " << glGetString(GL_RENDERER) << '\n';
    }
    ~Context() { boat_renderer.reset(); boat_simulation.reset(); water::appearance_fault = water::AppearanceFault::none; CloseWindow(); }
};

water::Config small_config() {
    water::Config config;
    config.grid = {8, 12, 4, 6};
    config.impact_radius = 1;
    config.depth = 1;
    return config;
}

const Camera3D camera{{0, 5, 8}, {0, -0.4f, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};

struct Sample { std::array<unsigned char, 4> color{}; float depth = 1; };

Sample sample(RenderTexture2D target, int x, int y) {
    glBindFramebuffer(GL_FRAMEBUFFER, target.id);
    Sample result;
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, result.color.data());
    glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &result.depth);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    CHECK(glGetError() == GL_NO_ERROR);
    return result;
}

Sample sample_world(RenderTexture2D target, Vector3 point, const Camera3D& view = camera) {
    const auto pixel = GetWorldToScreenEx(point, view, GetScreenWidth(), GetScreenHeight());
    return sample(target, static_cast<int>(pixel.x * target.texture.width / GetScreenWidth()),
                  target.texture.height - 1 - static_cast<int>(pixel.y * target.texture.height / GetScreenHeight()));
}

float expected_depth(Vector3 point, const Camera3D& view = camera) {
    const Vector3 forward{view.target.x - view.position.x, view.target.y - view.position.y,
                          view.target.z - view.position.z};
    const float length = std::sqrt(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
    const float distance = ((point.x - view.position.x) * forward.x +
                            (point.y - view.position.y) * forward.y +
                            (point.z - view.position.z) * forward.z) / length;
    const double near = rlGetCullDistanceNear(), far = rlGetCullDistanceFar();
    return static_cast<float>(far / (far - near) - far * near / ((far - near) * distance));
}

void check_deleted(RenderTexture2D target) {
    CHECK(!glIsFramebuffer(target.id));
    CHECK(!glIsTexture(target.texture.id));
    CHECK(!glIsTexture(target.depth.id));
}

void check_target(RenderTexture2D target, int width, int height) {
    CHECK(target.texture.width == width && target.texture.height == height);
    CHECK(target.depth.width == width && target.depth.height == height);
    CHECK(glIsTexture(target.texture.id) && glIsTexture(target.depth.id));
    glBindFramebuffer(GL_FRAMEBUFFER, target.id);
    CHECK(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    GLint type = 0, name = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &name);
    CHECK(type == GL_TEXTURE && name == target.depth.id);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    for (const auto texture : {target.texture, target.depth}) {
        glBindTexture(GL_TEXTURE_2D, texture.id);
        GLint value = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &value);
        CHECK(value == width);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &value);
        CHECK(value == height);
        if (texture.id == target.texture.id) {
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &value);
            CHECK(value == GL_RGBA8);
        }
        for (const auto parameter : {GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER}) {
            glGetTexParameteriv(GL_TEXTURE_2D, parameter, &value);
            CHECK(value == GL_NEAREST);
        }
        for (const auto parameter : {GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T}) {
            glGetTexParameteriv(GL_TEXTURE_2D, parameter, &value);
            CHECK(value == GL_CLAMP_TO_EDGE);
        }
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    CHECK(glGetError() == GL_NO_ERROR);
}

struct SkyProbe {
    Shader shader{};
    SkyProbe() {
        auto source = water::SceneValidation::fragment_source("sky.fs");
        source.insert(source.find('\n') + 1, "#define main backgroundMain\n");
        source += "\n#undef main\nuniform vec3 probeDirection; uniform int visibilityOnly;\n"
                  "void main() { finalColor = vec4(visibilityOnly == 1 ? vec3(sunVisibility(simTime)) : "
                  "skyRadiance(probeDirection, simTime), 1); }\n";
        shader = LoadShaderFromMemory(nullptr, source.c_str());
        CHECK(shader.id && shader.id != rlGetShaderIdDefault());
    }
    ~SkyProbe() { UnloadShader(shader); }
    Sample read(Vector3 direction, float time, bool visibility = false) {
        const int mode = visibility ? 1 : 0;
        SetShaderValue(shader, GetShaderLocation(shader, "probeDirection"), &direction, SHADER_UNIFORM_VEC3);
        SetShaderValue(shader, GetShaderLocation(shader, "simTime"), &time, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shader, GetShaderLocation(shader, "visibilityOnly"), &mode, SHADER_UNIFORM_INT);
        BeginShaderMode(shader);
        DrawRectangle(0, 0, 840, 560, WHITE);
        EndShaderMode();
        rlDrawRenderBatchActive();
        return sample({}, 420, 280);
    }
};

struct WaterProbe {
    Shader shader{};
    explicit WaterProbe(const water::SceneRenderer& scene, const Camera3D& view) {
        const char* vertex = "#version 330\nin vec3 vertexPosition; uniform mat4 mvp;"
            "uniform vec3 probePosition; uniform vec3 probeNormal; out vec3 worldPosition; out vec3 normal;"
            "void main(){ worldPosition=probePosition; normal=probeNormal; gl_Position=mvp*vec4(vertexPosition,1); }";
        const auto source = water::SceneValidation::fragment_source("water.fs");
        shader = LoadShaderFromMemory(vertex, source.c_str());
        CHECK(shader.id && shader.id != rlGetShaderIdDefault());
        SetShaderValueMatrix(shader, GetShaderLocation(shader, "captureView"), water::SceneValidation::capture_view(scene));
        SetShaderValueMatrix(shader, GetShaderLocation(shader, "captureProjection"), water::SceneValidation::capture_projection(scene));
        SetShaderValue(shader, GetShaderLocation(shader, "eyePosition"), &view.position, SHADER_UNIFORM_VEC3);
        const Vector2 size{840, 560}, planes{float(rlGetCullDistanceNear()), float(rlGetCullDistanceFar())};
        SetShaderValue(shader, GetShaderLocation(shader, "drawableSize"), &size, SHADER_UNIFORM_VEC2);
        SetShaderValue(shader, GetShaderLocation(shader, "clipPlanes"), &planes, SHADER_UNIFORM_VEC2);
        const float zero = 0;
        SetShaderValue(shader, GetShaderLocation(shader, "simTime"), &zero, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shader, GetShaderLocation(shader, "reflectionWeight"), &zero, SHADER_UNIFORM_FLOAT);
        const int color = 1, depth = 2;
        SetShaderValue(shader, GetShaderLocation(shader, "captureColor"), &color, SHADER_UNIFORM_INT);
        SetShaderValue(shader, GetShaderLocation(shader, "captureDepth"), &depth, SHADER_UNIFORM_INT);
    }
    ~WaterProbe() { UnloadShader(shader); }
    Sample read(RenderTexture2D target, Vector3 point, Vector3 normal) {
        SetShaderValue(shader, GetShaderLocation(shader, "probePosition"), &point, SHADER_UNIFORM_VEC3);
        SetShaderValue(shader, GetShaderLocation(shader, "probeNormal"), &normal, SHADER_UNIFORM_VEC3);
        rlActiveTextureSlot(1); rlEnableTexture(target.texture.id);
        rlActiveTextureSlot(2); rlEnableTexture(target.depth.id);
        rlActiveTextureSlot(0);
        BeginShaderMode(shader);
        DrawRectangle(0, 0, 840, 560, WHITE);
        EndShaderMode();
        rlDrawRenderBatchActive();
        rlActiveTextureSlot(2); rlDisableTexture();
        rlActiveTextureSlot(1); rlDisableTexture();
        rlActiveTextureSlot(0);
        return sample({}, 420, 280);
    }
};

struct AssetOverride {
    std::filesystem::path path;
    std::string original;
    AssetOverride(const char* name, const std::string& replacement)
        : path(std::filesystem::path(GetApplicationDirectory()) / "assets" / name) {
        std::ifstream input(path, std::ios::binary);
        original = {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        CHECK(input.is_open());
        input.close();
        std::ofstream(path, std::ios::binary) << replacement;
    }
    ~AssetOverride() { std::ofstream(path, std::ios::binary) << original; }
};

} // namespace

TEST_CASE("appearance", drop_mesh_obeys_spawn_bounds) {
    Context context;
    water::SceneRenderer scene(small_config());
    const auto& mesh = water::SceneValidation::drop_mesh(scene);
    CHECK(mesh.vertexCount > 100 && mesh.vertices && mesh.normals && mesh.vaoId);
    float bottom = 1, top = 0, lower_radius = 0, upper_radius = 0;
    for (int i = 0; i < mesh.vertexCount; ++i) {
        const float x = mesh.vertices[3*i], y = mesh.vertices[3*i+1], z = mesh.vertices[3*i+2];
        const float radius = std::sqrt(x*x + z*z);
        CHECK(radius <= 0.065001f && y >= 0 && y <= 0.280001f);
        bottom = std::min(bottom, y); top = std::max(top, y);
        if (y < 0.14f) lower_radius = std::max(lower_radius, radius);
        if (y > 0.23f) upper_radius = std::max(upper_radius, radius);
        const float nx = mesh.normals[3*i], ny = mesh.normals[3*i+1], nz = mesh.normals[3*i+2];
        CHECK_NEAR(nx*nx + ny*ny + nz*nz, 1.0f, 0.001f);
    }
    CHECK_NEAR(bottom, 0.0f, 0.00001f); CHECK_NEAR(top, 0.28f, 0.00001f);
    CHECK(lower_radius > 0.06f && upper_radius < 0.025f);
    const water::Grid grid{96, 64, 20, 12};
    for (float pitch : {0.15f, 0.72f, 1.45f})
    for (float yaw : {0.0f, 0.65f, 1.8f, 3.4f, 4.9f})
    for (float scale : {0.4f, 1.65f, 4.0f})
    for (float aspect : {1.5f, 1.6f, 2.0f}) {
        const float distance = scale * 20;
        const Camera3D view{{distance*std::cos(pitch)*std::sin(yaw), distance*std::sin(pitch),
                             distance*std::cos(pitch)*std::cos(yaw)}, {0,0,0}, {0,1,0}, 42, CAMERA_PERSPECTIVE};
        const auto eye = view.position;
        const Vector3 forward{-eye.x/distance, -eye.y/distance, -eye.z/distance};
        const Vector3 up{-std::sin(pitch)*std::sin(yaw), std::cos(pitch), -std::sin(pitch)*std::cos(yaw)};
        for (const Vector2 point : {Vector2{std::min(0.f,eye.x+10)-10,std::min(0.f,eye.z+6)-10},
                                   {std::max(20.f,eye.x+10)+10,std::min(0.f,eye.z+6)-10},
                                   {std::min(0.f,eye.x+10)-10,std::max(12.f,eye.z+6)+10},
                                   {std::max(20.f,eye.x+10)+10,std::max(12.f,eye.z+6)+10}, {10,6}}) {
            const double height = water::drop_spawn_height(point.x, point.y, grid, {view, aspect});
            bool above = true, behind = true;
            for (int i = 0; i < mesh.vertexCount; ++i) {
                const double x = point.x-10 + mesh.vertices[3*i]-eye.x;
                const double y = height + mesh.vertices[3*i+1]-eye.y;
                const double z = point.y-6 + mesh.vertices[3*i+2]-eye.z;
                const double depth = x*forward.x+y*forward.y+z*forward.z;
                const double vertical = x*up.x+y*up.y+z*up.z;
                above &= vertical*std::cos(21*DEG2RAD)-depth*std::sin(21*DEG2RAD) >= 0.11999;
                behind &= -depth >= 0.11999;
            }
            CHECK(above || behind);
        }
    }
}

TEST_CASE("appearance", drop_mesh_faces_point_outward) {
    Context context;
    water::SceneRenderer scene(small_config());
    const auto& mesh = water::SceneValidation::drop_mesh(scene);
    CHECK(mesh.vertices && mesh.normals && mesh.triangleCount > 100);
    CHECK(mesh.vertexCount == 3 * mesh.triangleCount);
    int checked = 0;
    for (int triangle = 0; triangle < mesh.triangleCount; ++triangle) {
        const float* a = mesh.vertices + 9 * triangle;
        const Vector3 ab{a[3] - a[0], a[4] - a[1], a[5] - a[2]};
        const Vector3 ac{a[6] - a[0], a[7] - a[1], a[8] - a[2]};
        const Vector3 face{ab.y * ac.z - ab.z * ac.y,
                           ab.z * ac.x - ab.x * ac.z,
                           ab.x * ac.y - ab.y * ac.x};
        if (face.x * face.x + face.y * face.y + face.z * face.z <= 1e-20f) continue;
        ++checked;
        for (int vertex = 0; vertex < 3; ++vertex) {
            const float* normal = mesh.normals + 9 * triangle + 3 * vertex;
            CHECK(face.x * normal[0] + face.y * normal[1] + face.z * normal[2] > 0);
        }
    }
    CHECK(checked == mesh.triangleCount);
}

TEST_CASE("appearance", drop_rendering_uses_stored_flight) {
    Context context;
    auto config = small_config();
    water::Simulation simulation(config);
    water::AppController app(simulation, config);
    water::WaterRenderer water(config.grid);
    water::SceneRenderer scene(config);
    CHECK(scene.resize(840, 560));
    const water::SpawnView spawn_view{camera, 1.5f};
    CHECK(app.spawn(2, 3, spawn_view));
    const auto born = app.drops().front();
    const double visible_time = std::sqrt(2*(born.spawn_height - 1.4)/9.81);
    while (app.sim_time() + water::FixedClock::quantum <= visible_time)
        app.frame(water::FixedClock::quantum, false, false, 0, 0.01f, 0.15f, spawn_view);
    CHECK(app.drops().size() == 1);
    CHECK(app.drops()[0].arrival == born.arrival && app.drops()[0].spawn_height == born.spawn_height);
    const double time = app.sim_time();
    const float y = static_cast<float>(born.spawn_height - 0.5*9.81*time*time);
    const auto projected = GetWorldToScreenEx({0, y+0.075f, 0}, camera, 840, 560);
    const int px = int(projected.x), py = 559-int(projected.y);
    CHECK(px > 20 && px < 820 && py > 20 && py < 540);
    water.update(simulation);
    BeginDrawing();
    scene.draw(camera, water, context.boat(), {}, time, 0.5f);
    std::array<unsigned char, 20*24*4> empty{}, visible{}, paused{};
    glReadPixels(px-10, py-12, 20, 24, GL_RGBA, GL_UNSIGNED_BYTE, empty.data());
    scene.draw(camera, water, context.boat(), app.drops(), time, 0.5f);
    glReadPixels(px-10, py-12, 20, 24, GL_RGBA, GL_UNSIGNED_BYTE, visible.data());
    int changed = 0;
    for (std::size_t i = 0; i < empty.size(); i += 4)
        if (std::abs(int(empty[i])-int(visible[i])) + std::abs(int(empty[i+2])-int(visible[i+2])) > 15) ++changed;
    CHECK(changed > 12);
    const auto image = LoadImageFromScreen();
    CHECK(ExportImage(image, "appearance-drop-midflight.png")); UnloadImage(image);
    app.frame(10, true, false, 0, 0.01f, 0.15f, spawn_view);
    CHECK(app.sim_time() == time);
    scene.draw(camera, water, context.boat(), app.drops(), app.sim_time(), 0.5f);
    glReadPixels(px-10, py-12, 20, 24, GL_RGBA, GL_UNSIGNED_BYTE, paused.data());
    CHECK(visible == paused);
    const Camera3D closeup{{0, 2.0f, 1.8f}, {0, y + 0.11f, 0}, {0, 1, 0}, 32, CAMERA_PERSPECTIVE};
    app.frame(10, true, false, 0, 0.01f, 0.15f, {closeup, 1.5f});
    CHECK(app.sim_time() == time && app.drops()[0].spawn_height == born.spawn_height &&
          app.drops()[0].arrival == born.arrival);
    scene.draw(closeup, water, context.boat(), app.drops(), time, 0.5f);
    const auto detail = LoadImageFromScreen();
    CHECK(ExportImage(detail, "appearance-drop-closeup.png")); UnloadImage(detail);
    EndDrawing();
    while (app.sim_time() + water::FixedClock::quantum < born.arrival)
        app.frame(water::FixedClock::quantum, false, false, 0, 0.01f, 0.15f, spawn_view);
    const auto before = water::download(simulation);
    for (float eta : before.eta) CHECK(eta == 0);
    CHECK(app.drops().size() == 1);
    app.frame(water::FixedClock::quantum, false, false, 0, 0.01f, 0.15f, spawn_view);
    CHECK(app.sim_time() >= born.arrival && app.drops().empty());
    const auto after = water::download(simulation);
    float amplitude = 0;
    for (float eta : after.eta) amplitude = std::max(amplitude, std::abs(eta));
    CHECK(amplitude > 0.0001f);
    water.update(simulation);
    BeginDrawing();
    scene.draw(camera, water, context.boat(), app.drops(), app.sim_time(), 0.5f);
    glReadPixels(px-10, py-12, 20, 24, GL_RGBA, GL_UNSIGNED_BYTE, visible.data());
    scene.draw(camera, water, context.boat(), {}, app.sim_time(), 0.5f);
    glReadPixels(px-10, py-12, 20, 24, GL_RGBA, GL_UNSIGNED_BYTE, empty.data());
    CHECK(visible == empty);
    EndDrawing();
    CHECK(MeasureText("Up/Down: rain   Left/Right: strength   [/]: damping   +/-: reflection   Esc: close", 14) <= 784);
    app.reset();
    CHECK(app.sim_time() == 0 && app.drops().empty());
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", mix_endpoints_are_exact_color_weights) {
    Context context;
    auto config = small_config();
    water::Simulation simulation(config);
    water::WaterRenderer water(config.grid);
    water::SceneRenderer scene(config);
    CHECK(scene.resize(840, 560));
    for (bool impacted : {false, true}) {
        if (impacted) {
            const water::Impact impact{2, 3, 1, 0.02f};
            simulation.apply_impacts(&impact, 1);
        }
        water.update(simulation);
        for (const Camera3D view : {camera,
                Camera3D{{0, 0.36f, 2.373f}, {0, 0, 0}, {0, 1, 0}, 42, CAMERA_PERSPECTIVE},
                Camera3D{{0, 23.825f, 2.892f}, {0, 0, 0}, {0, 1, 0}, 42, CAMERA_PERSPECTIVE}}) {
            std::array<std::vector<unsigned char>, 3> images;
            for (int i = 0; i < 3; ++i) {
                BeginDrawing();
                scene.draw(view, water, context.boat(), {}, 12, i * 0.5f);
                rlDrawRenderBatchActive();
                images[i].resize(80 * 40 * 4);
                glReadPixels(380, 260, 80, 40, GL_RGBA, GL_UNSIGNED_BYTE, images[i].data());
                EndDrawing();
            }
            int difference = 0;
            for (std::size_t pixel = 0; pixel < images[0].size(); pixel += 4) {
                for (int c = 0; c < 3; ++c) {
                    CHECK_NEAR(float(images[1][pixel + c]),
                               0.5f * (images[0][pixel + c] + images[2][pixel + c]), 2.0f);
                    difference += std::abs(int(images[0][pixel + c]) - int(images[2][pixel + c]));
                }
                CHECK(images[1][pixel + 3] == 255);
            }
            CHECK(difference > 80 * 40 * 20);
        }
    }
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", normal_driven_refraction_moves_stone) {
    Context context;
    auto config = small_config();
    config.grid.nx = 128; config.grid.ny = 192;
    config.impact_radius = 0.4f;
    water::Simulation simulation(config);
    water::WaterRenderer water(config.grid);
    water::SceneRenderer scene(config);
    CHECK(scene.resize(840, 560));
    const Camera3D overhead{{0, 8, 0}, {0, 0, 0}, {0, 0, -1}, 45, CAMERA_PERSPECTIVE};
    std::array<std::vector<unsigned char>, 2> images;
    for (int i = 0; i < 2; ++i) {
        if (i) {
            const water::Impact impact{2, 3, 0.4f, 0.02f};
            simulation.apply_impacts(&impact, 1);
        }
        water.update(simulation);
        BeginDrawing();
        scene.draw(overhead, water, context.boat(), {}, 12, 0);
        images[i].resize(160 * 160 * 4);
        glReadPixels(340, 200, 160, 160, GL_RGBA, GL_UNSIGNED_BYTE, images[i].data());
        if (!i) {
            const auto stone = sample_world(water::SceneValidation::target(scene), {0, -1, 0}, overhead);
            RenderTexture2D screen{}; screen.texture = {0, 840, 560, 1, 0};
            const auto transmitted = sample_world(screen, {0, 0, 0}, overhead);
            CHECK(std::abs(int(transmitted.color[0]) - int(stone.color[0])) < 45);
            CHECK(std::abs(int(transmitted.color[2]) - int(stone.color[2])) < 45);
        }
        EndDrawing();
    }
    int changed = 0, unchanged = 0;
    for (int y = 0; y < 160; ++y) for (int x = 0; x < 160; ++x) {
        int difference = 0;
        for (int c = 0; c < 3; ++c)
            difference += std::abs(int(images[0][(y * 160 + x) * 4 + c]) - int(images[1][(y * 160 + x) * 4 + c]));
        if (x > 45 && x < 115 && y > 45 && y < 115 && difference > 2) ++changed;
        if ((x < 20 || x > 140) && difference <= 2) ++unchanged;
    }
    CHECK(changed > 40);
    CHECK(unchanged > 5000);
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", sky_direction_time_and_pause_are_shared) {
    Context context;
    const auto config = small_config();
    water::Simulation simulation(config);
    water::WaterRenderer water(config.grid);
    water.update(simulation);
    water::SceneRenderer scene(config);
    CHECK(scene.resize(840, 560));
    const Camera3D sky_camera{{0, 8, 0}, {0, 9, -1}, {0, 1, 0}, 60, CAMERA_PERSPECTIVE};
    std::array<std::vector<unsigned char>, 4> images;
    for (int i = 0; i < 4; ++i) {
        BeginDrawing();
        scene.draw(sky_camera, water, context.boat(), {}, i == 2 ? 90 : 0, 1);
        images[i].resize(840 * 560 * 4);
        glReadPixels(0, 0, 840, 560, GL_RGBA, GL_UNSIGNED_BYTE, images[i].data());
        EndDrawing();
    }
    CHECK(images[0] == images[1]);
    CHECK(images[0] == images[3]);
    int changed = 0;
    for (std::size_t i = 0; i < images[0].size(); i += 4)
        if (std::abs(int(images[0][i]) - int(images[2][i])) > 8) ++changed;
    CHECK(changed > 10000);

    SkyProbe probe;
    for (const float time : {0.0f, 90.0f}) {
        BeginDrawing();
        scene.draw(sky_camera, water, context.boat(), {}, time, 1);
        const auto background = sample({}, 420, 280);
        const Ray ray = GetScreenToWorldRay({420.5f, 279.5f}, sky_camera);
        const auto expected = probe.read(ray.direction, time);
        for (int c = 0; c < 3; ++c) CHECK_NEAR(int(background.color[c]), int(expected.color[c]), 2);
        auto moved = sky_camera;
        moved.position.x += 50; moved.target.x += 50;
        scene.draw(moved, water, context.boat(), {}, time, 1);
        const auto paused = sample({}, 420, 280);
        for (int c = 0; c < 3; ++c) CHECK_NEAR(int(paused.color[c]), int(expected.color[c]), 2);
        const Camera3D reflected{{-8, 4, 0}, {0, 0, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};
        scene.draw(reflected, water, context.boat(), {}, time, 1);
        const auto reflection = sample({}, 420, 280);
        const Ray incident = GetScreenToWorldRay({420.5f, 279.5f}, reflected);
        const auto sky = probe.read({incident.direction.x, -incident.direction.y, incident.direction.z}, time);
        for (int c = 0; c < 3; ++c) CHECK_NEAR(int(reflection.color[c]), int(sky.color[c]), 2);
        EndDrawing();
    }
    std::array<Sample, 2> sunlight;
    int sun_index = 0;
    for (const float time : {389.0f, 0.0f}) {
        BeginDrawing();
        const auto visibility = probe.read({0, 1, 0}, time, true);
        CHECK(time > 0 ? visibility.color[0] < 25 : visibility.color[0] > 230);
        const Camera3D sun_camera{{0, 8, 0}, {-0.35f, 8.85f, -0.4f}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};
        scene.draw(sun_camera, water, context.boat(), {}, time, 1);
        sunlight[sun_index++] = sample({}, 420, 280);
        const auto sun_ray = GetScreenToWorldRay({420.5f, 279.5f}, sun_camera);
        const auto expected = probe.read(sun_ray.direction, time);
        for (int c = 0; c < 3; ++c) CHECK_NEAR(int(sunlight[sun_index-1].color[c]), int(expected.color[c]), 2);
        const Camera3D reflected{{2.8f, 6.8f, 3.2f}, {0, 0, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};
        scene.draw(reflected, water, context.boat(), {}, time, 1);
        const auto highlight = sample({}, 420, 280);
        const auto incident = GetScreenToWorldRay({420.5f, 279.5f}, reflected);
        const auto sky = probe.read({incident.direction.x, -incident.direction.y, incident.direction.z}, time);
        const int added_blue = int(highlight.color[2]) - sky.color[2];
        CHECK(time > 0 ? added_blue >= 0 && added_blue < 8 : added_blue > 35);
        EndDrawing();
    }
    CHECK(sunlight[1].color[0] > sunlight[0].color[0] + 90);
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", refraction_rejects_rim_clear_and_screen_edges) {
    Context context;
    water::SceneRenderer scene(small_config());
    CHECK(scene.resize(840, 560));
    BeginDrawing();
    for (const Camera3D view : {camera,
            Camera3D{{0, 0.7f, 7}, {0, -0.2f, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE},
            Camera3D{{0, 3, 1}, {0, 0, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE}}) {
        water::SceneValidation::capture(scene, view, context.boat());
        const auto target = water::SceneValidation::target(scene);
        WaterProbe probe(scene, view);
        for (const Vector3 point : {Vector3{0, 0.12f, 3.09f}, Vector3{100, 0, 0}}) {
            const auto fallback = probe.read(target, point, {0.8f, 0.6f, 0});
            CHECK_NEAR(int(fallback.color[0]), 67, 2);
            CHECK_NEAR(int(fallback.color[1]), 83, 2);
            CHECK_NEAR(int(fallback.color[2]), 90, 2);
            CHECK(fallback.color[3] == 255);
        }
    }
    // A narrow invalid texel on the ray segment must reject an otherwise valid destination.
    const Camera3D overhead{{0, 8, 0}, {0, 0, 0}, {0, 0, -1}, 45, CAMERA_PERSPECTIVE};
    water::SceneValidation::capture(scene, overhead, context.boat());
    const auto target = water::SceneValidation::target(scene);
    WaterProbe probe(scene, overhead);
    std::vector<unsigned char> colors(840 * 560 * 4, 255);
    std::vector<float> depths(840 * 560, expected_depth({0, -1, 0}, overhead));
    const auto upload = [&] {
        glBindTexture(GL_TEXTURE_2D, target.texture.id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 840, 560, GL_RGBA, GL_UNSIGNED_BYTE, colors.data());
        glBindTexture(GL_TEXTURE_2D, target.depth.id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 840, 560, GL_DEPTH_COMPONENT, GL_FLOAT, depths.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    };
    // At this camera a tilted normal reaches the +0.02 UV cap: base x=420, destination x=436.
    for (int y = 0; y < 560; ++y) for (int x = 0; x < 840; ++x) {
        const int p = (y*840+x)*4;
        colors[p] = x >= 435 ? 255 : 100;
        colors[p+1] = x >= 435 ? 0 : 110;
        colors[p+2] = x >= 435 ? 255 : 120;
    }
    upload();
    const auto edge_ray = GetScreenToWorldRay({0.5f, 279.5f}, overhead);
    const float distance = -overhead.position.y / edge_ray.direction.y;
    const auto edge = probe.read(target,
        {edge_ray.direction.x * distance, 0, edge_ray.direction.z * distance}, {0, 1, 0});
    CHECK_NEAR(int(edge.color[0]), 67, 2);
    CHECK_NEAR(int(edge.color[1]), 83, 2);
    CHECK_NEAR(int(edge.color[2]), 90, 2);
    const Vector3 tilt{0.8f, 0.6f, 0};
    const auto accepted = probe.read(target, {0, 0, 0}, tilt);
    const auto undistorted = probe.read(target, {0, 0, 0}, {0, 1, 0});
    CHECK(accepted.color[0] > undistorted.color[0] + 70);
    const float fault_depths[] = {expected_depth({0, -1, 0}, overhead), 1,
                                 expected_depth({0, 0.1f, 0}, overhead), expected_depth({0, -3, 0}, overhead)};
    for (int fault = 0; fault < 4; ++fault) {
        for (int y = 0; y < 560; ++y) {
            const int pixel = y*840 + 428;
            colors[pixel*4+3] = fault == 0 ? 0 : 255;
            depths[pixel] = fault_depths[fault];
        }
        upload();
        const auto rejected = probe.read(target, {0, 0, 0}, tilt);
        for (int c = 0; c < 3; ++c) CHECK_NEAR(int(rejected.color[c]), int(undistorted.color[c]), 2);
    }
    for (int y = 0; y < 560; ++y) for (int x = 0; x < 840; ++x) {
        const int pixel = y*840 + x;
        const bool displaced_band = x >= 421 && x <= 423;
        colors[pixel*4] = displaced_band ? 255 : 100;
        colors[pixel*4+1] = displaced_band ? 0 : 110;
        colors[pixel*4+2] = displaced_band ? 255 : 120;
        colors[pixel*4+3] = 255;
        depths[pixel] = expected_depth({0, -1, 0}, overhead);
    }
    upload();
    const auto small_wave = probe.read(target, {0, 0, 0}, {0.02f, 0.9998f, 0});
    CHECK(small_wave.color[0] > undistorted.color[0] + 70);
    EndDrawing();
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", shader_loader_rejects_missing_marker_compile_and_uniform_failures) {
    Context context;
    CHECK_THROWS(water::SceneValidation::fragment_source("missing.fs"), std::runtime_error);
    const std::array<std::pair<std::string, std::string>, 4> failures{{
        {"void main() {}", "#version"},
        {"#version 330\nout vec4 finalColor; void main() { finalColor=vec4(1); }", "SHARED_SKY"},
        {"#version 330\n// SHARED_SKY\n invalid GLSL code", "compile"},
        {"#version 330\n// SHARED_SKY\nout vec4 finalColor; void main() { finalColor=vec4(1); }", "eyePosition"}
    }};
    for (const auto& failure : failures) {
        AssetOverride asset("water.fs", failure.first);
        bool rejected = false;
        try { water::SceneRenderer scene(small_config()); }
        catch (const std::runtime_error& error) {
            rejected = true;
            CHECK(std::string(error.what()).find(failure.second) != std::string::npos);
        }
        CHECK(rejected);
        CHECK(glIsProgram(rlGetShaderIdDefault()));
    }
    water::SceneRenderer recovered(small_config());
    CHECK(recovered.resize(840, 560));
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", capture_contains_only_submerged_stone) {
    Context context;
    water::SceneRenderer scene(small_config());
    CHECK(scene.resize(840, 560));
    BeginDrawing();
    water::SceneValidation::capture(scene, camera, context.boat());
    const auto target = water::SceneValidation::target(scene);
    for (const Vector3 point : {Vector3{0, -1, 0}, Vector3{-1, -1, -1}, Vector3{0, -0.5f, -3}}) {
        const auto pixel = sample_world(target, point);
        CHECK(pixel.color[3] == 255);
        CHECK(pixel.color[0] > 20 && pixel.color[1] > 20 && pixel.color[2] > 20);
        CHECK_NEAR(pixel.depth, expected_depth(point), 0.00002f);
    }
    const auto clear = sample(target, 10, 550);
    CHECK(clear.color[3] == 0 && clear.depth == 1);
    const auto rim = sample_world(target, {0, 0.12f, 3.09f});
    CHECK(rim.color[3] == 0);
    CHECK_NEAR(rim.depth, expected_depth({0, 0.12f, 3.09f}), 0.00002f);
    // This floor point is behind the foreground rim on the same viewing ray.
    const float scale = (-1.0f - camera.position.y) / (0.12f - camera.position.y);
    const Vector3 occluded{0, -1, camera.position.z + scale * (3.09f - camera.position.z)};
    const auto foreground = sample_world(target, occluded);
    CHECK(foreground.color[3] == 0);
    CHECK(foreground.depth < expected_depth(occluded) - 0.00005f);
    const Camera3D overhead{{0, 12, 0}, {0, 0, 0}, {0, 0, -1}, 45, CAMERA_PERSPECTIVE};
    water::SceneValidation::capture(scene, overhead, context.boat());
    for (const Vector3 point : {Vector3{-1.7f, -1, -2.7f}, Vector3{1.7f, -1, 2.7f}}) {
        const auto pixel = sample_world(target, point, overhead);
        CHECK(pixel.color[3] == 255);
        CHECK_NEAR(pixel.depth, expected_depth(point, overhead), 0.00002f);
    }
    const Camera3D grazing{{0, 1, 8}, {0, -0.4f, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};
    water::SceneValidation::capture(scene, grazing, context.boat());
    const auto blocked_floor = sample_world(target, {0, -1, 0}, grazing);
    CHECK(blocked_floor.color[3] == 255);
    CHECK(blocked_floor.depth < expected_depth({0, -1, 0}, grazing) - 0.00005f);
    const auto grazing_rim = sample_world(target, {0, 0.12f, 3.09f}, grazing);
    CHECK(grazing_rim.color[3] == 0);
    CHECK_NEAR(grazing_rim.depth, expected_depth({0, 0.12f, 3.09f}, grazing), 0.00002f);
    const auto distant_wall = sample_world(target, {0, -0.15f, -3}, grazing);
    CHECK(distant_wall.color[3] == 255);
    CHECK_NEAR(distant_wall.depth, expected_depth({0, -0.15f, -3}, grazing), 0.00002f);
    EndDrawing();
}

TEST_CASE("appearance", target_resize_and_failure_release_resources) {
    Context context;
    RenderTexture2D final{};
    {
        water::SceneRenderer scene(small_config());
        CHECK(!scene.resize(0, 480));
        CHECK(water::SceneValidation::target(scene).id == 0);
        for (const auto size : {std::array<int, 2>{640, 480}, {840, 560}, {1260, 840}}) {
            const auto old = water::SceneValidation::target(scene);
            CHECK(scene.resize(size[0], size[1]));
            final = water::SceneValidation::target(scene);
            check_target(final, size[0], size[1]);
            if (old.id) check_deleted(old);
        }
        CHECK(!scene.resize(0, 0));
        CHECK(!scene.resize(-1, 560));
        CHECK(water::SceneValidation::target(scene).id == final.id);
        CHECK(scene.resize(1260, 840));
        CHECK(water::SceneValidation::target(scene).id == final.id);
        for (const auto fault : {water::AppearanceFault::framebuffer, water::AppearanceFault::color_texture,
                                water::AppearanceFault::depth_texture, water::AppearanceFault::color_attachment,
                                water::AppearanceFault::depth_attachment, water::AppearanceFault::complete}) {
            water::appearance_fault = fault;
            CHECK(!scene.resize(700, 500));
            const auto failed = water::appearance_handles;
            CHECK(!glIsFramebuffer(failed.framebuffer));
            CHECK(!glIsTexture(failed.color));
            CHECK(!glIsTexture(failed.depth));
            CHECK(water::SceneValidation::target(scene).id == final.id);
            check_target(final, 1260, 840);
            water::appearance_fault = water::AppearanceFault::none;
            CHECK(scene.resize(700, 500));
            check_deleted(final);
            CHECK(scene.resize(1260, 840));
            final = water::SceneValidation::target(scene);
        }
    }
    check_deleted(final);
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", constructor_failures_release_meshes_and_shaders) {
    Context context;
    for (const auto fault : {water::AppearanceFault::stone_shader, water::AppearanceFault::water_shader,
                            water::AppearanceFault::basin_mesh, water::AppearanceFault::grass_shader,
                            water::AppearanceFault::grass_mesh}) {
        glUseProgram(rlGetShaderIdDefault());
        water::appearance_fault = fault;
        CHECK_THROWS(water::SceneRenderer(small_config()), std::runtime_error);
        GLint program = 0;
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        CHECK(program == rlGetShaderIdDefault());
        const auto handles = water::appearance_handles;
        CHECK(!glIsProgram(handles.stone_shader));
        CHECK(!glIsProgram(handles.water_shader));
        CHECK(!glIsProgram(handles.grass_shader));
        CHECK(!glIsVertexArray(handles.grass_vertex_array));
        CHECK(!glIsBuffer(handles.grass_positions) && !glIsBuffer(handles.grass_normals));
        CHECK(!glIsVertexArray(handles.vertex_array));
        CHECK(!glIsBuffer(handles.positions));
        CHECK(!glIsBuffer(handles.normals));
        water::appearance_fault = water::AppearanceFault::none;
        water::SceneRenderer recovered(small_config());
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        CHECK(program == rlGetShaderIdDefault());
        CHECK(recovered.resize(640, 480));
        CHECK(glGetError() == GL_NO_ERROR);
    }
}

TEST_CASE("appearance", constructor_restores_caller_program) {
    Context context;
    for (const auto caller : {0u, rlGetShaderIdDefault()}) {
        glUseProgram(caller);
        {
            water::SceneRenderer scene(small_config());
            GLint program = 0;
            glGetIntegerv(GL_CURRENT_PROGRAM, &program);
            CHECK(program == caller);
        }
        GLint program = 0;
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        CHECK(program == caller);
    }
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", capture_restores_caller_depth_masks_framebuffer_and_viewport) {
    Context context;
    water::SceneRenderer scene(small_config());
    CHECK(scene.resize(640, 480));
    const auto other = LoadRenderTexture(40, 30);
    BeginDrawing();
    glBindFramebuffer(GL_FRAMEBUFFER, other.id);
    glViewport(3, 4, 25, 20);
    glDepthFunc(GL_GREATER);
    glDepthMask(GL_FALSE);
    glColorMask(GL_TRUE, GL_FALSE, GL_TRUE, GL_FALSE);
    glEnable(GL_DEPTH_TEST);
    water::SceneValidation::capture(scene, camera, context.boat());
    GLint value = 0, viewport[4]{};
    GLboolean masks[4]{}, depth_mask = GL_TRUE;
    glGetIntegerv(GL_DEPTH_FUNC, &value); CHECK(value == GL_GREATER);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &value); CHECK(value == other.id);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &value); CHECK(value == other.id);
    glGetIntegerv(GL_VIEWPORT, viewport);
    CHECK(viewport[0] == 3 && viewport[1] == 4 && viewport[2] == 25 && viewport[3] == 20);
    glGetBooleanv(GL_COLOR_WRITEMASK, masks);
    CHECK(masks[0] && !masks[1] && masks[2] && !masks[3]);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &depth_mask); CHECK(!depth_mask);
    CHECK(glIsEnabled(GL_DEPTH_TEST));
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, GetRenderWidth(), GetRenderHeight());
    glDepthFunc(GL_LEQUAL); glDepthMask(GL_TRUE); glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDisable(GL_DEPTH_TEST);
    water::SceneValidation::capture(scene, camera, context.boat());
    glGetIntegerv(GL_DEPTH_FUNC, &value); CHECK(value == GL_LEQUAL);
    CHECK(!glIsEnabled(GL_DEPTH_TEST));
    EndDrawing();
    UnloadRenderTexture(other);
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", capture_and_main_projection_align_at_drawable_sizes) {
    Context context;
    const auto config = small_config();
    water::Simulation simulation(config);
    water::WaterRenderer water(config.grid);
    water.update(simulation);
    water::SceneRenderer scene(config);
    for (const auto size : {std::array<int, 2>{840, 560}, {1260, 840}, {900, 900}}) {
        CHECK(scene.resize(size[0], size[1]));
        BeginDrawing();
        scene.draw(camera, water, context.boat(), {}, 0, 0.5f);
        const auto a = water::SceneValidation::capture_projection(scene), b = water::SceneValidation::projection(scene);
        CHECK_NEAR(a.m0, b.m0, 1e-6f); CHECK_NEAR(a.m5, b.m5, 1e-6f);
        CHECK_NEAR(a.m0 / a.m5, 560.0f / 840, 1e-6f);
        const auto va = water::SceneValidation::capture_view(scene), vb = water::SceneValidation::view(scene);
        CHECK_NEAR(va.m12, vb.m12, 1e-6f); CHECK_NEAR(va.m13, vb.m13, 1e-6f); CHECK_NEAR(va.m14, vb.m14, 1e-6f);
        const auto pixel = sample_world(water::SceneValidation::target(scene), {0, -1, 0});
        CHECK(pixel.color[3] == 255);
        CHECK_NEAR(pixel.depth, expected_depth({0, -1, 0}), 0.00002f);
        RenderTexture2D screen{};
        screen.texture.width = GetRenderWidth();
        screen.texture.height = GetRenderHeight();
        const auto water_pixel = sample_world(screen, {0, 0, 0});
        CHECK_NEAR(water_pixel.depth, expected_depth({0, 0, 0}), 0.00002f);
        CHECK(water_pixel.color[3] == 255);
        const auto stone_pixel = sample_world(screen, {0, 0.12f, 3.09f});
        CHECK(stone_pixel.color[0] > 35);
        CHECK(std::abs(int(stone_pixel.color[0]) - int(stone_pixel.color[2])) < 20);
        const auto background = sample(screen, 5, GetRenderHeight() - 6);
        const auto ray = GetScreenToWorldRay({5.5f, 5.5f}, camera);
        CHECK(ray.direction.y < 0);
        const float distance = -ray.position.y / ray.direction.y;
        const Vector3 ground{ray.position.x + distance * ray.direction.x, 0,
                              ray.position.z + distance * ray.direction.z};
        CHECK(background.color[1] > background.color[0] * 1.4f);
        CHECK_NEAR(background.depth, expected_depth(ground), .00002f);
        DrawRectangle(0, 0, 20, 20, MAGENTA);
        rlDrawRenderBatchActive();
        unsigned char ui[4]{};
        glReadPixels(5, GetRenderHeight() - 6, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, ui);
        CHECK(ui[0] == 255 && ui[1] == 0 && ui[2] == 255);
        EndDrawing();
        CHECK(glGetError() == GL_NO_ERROR);
    }
}

TEST_CASE("appearance", capture_orientation_and_hidpi_projection_match_known_floor_points) {
    Context context;
    water::SceneRenderer scene(small_config());
    const Camera3D view{{3, 5, 8}, {0, -0.4f, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};
    const Vector3 points[]{{-1.13f, -1, -1.73f}, {0.8f, -1, 0.37f}, {1.05f, -1, 1.43f}};
    // Independent pinhole projection and floor intersection, without raylib's projection helpers.
    const double length = std::sqrt(3.0 * 3.0 + 5.4 * 5.4 + 8.0 * 8.0);
    const double fx = -3.0 / length, fy = -5.4 / length, fz = -8.0 / length;
    const double rx = 8 / std::sqrt(73.0), rz = -3 / std::sqrt(73.0);
    const double ux = -rz * fy, uy = rz * fx - rx * fz, uz = rx * fy;
    const double tangent = std::tan(22.5 * DEG2RAD);
    std::array<Sample, 3> original{};
    for (const auto size : {std::array<int, 2>{840, 560}, {1260, 840}, {900, 600}, {840, 560}}) {
        CHECK(GetScreenWidth() == 840 && GetScreenHeight() == 560);
        CHECK(scene.resize(size[0], size[1]));
        BeginDrawing();
        water::SceneValidation::capture(scene, view, context.boat());
        const auto target = water::SceneValidation::target(scene);
        for (int i = 0; i < 3; ++i) {
            const auto point = points[i];
            const double depth = (point.x - 3) * fx + (point.y - 5) * fy + (point.z - 8) * fz;
            const double horizontal = (point.x - 3) * rx + (point.z - 8) * rz;
            const double vertical = (point.x - 3) * ux + (point.y - 5) * uy + (point.z - 8) * uz;
            const double logical_x = 420 * (1 + horizontal / (depth * tangent * 1.5));
            const double logical_y = 280 * (1 - vertical / (depth * tangent));
            const int x = int(logical_x * size[0] / 840);
            const int y = size[1] - 1 - int(logical_y * size[1] / 560);
            const auto pixel = sample(target, x, y);
            CHECK(pixel.color[3] == 255 && pixel.color[0] > 35);
            // The middle sample is a dark mortar joint; the others lie inside stone tiles.
            CHECK(i == 1 ? pixel.color[0] < 80 : pixel.color[0] > 85);
            const double screen_x = (2.0 * (x + 0.5) / size[0] - 1) * tangent * 1.5;
            const double screen_y = (2.0 * (y + 0.5) / size[1] - 1) * tangent;
            const double ray_x = fx + screen_x * rx + screen_y * ux;
            const double ray_y = fy + screen_y * uy, ray_z = fz + screen_x * rz + screen_y * uz;
            const double distance = -6 / ray_y;
            const Vector3 floor{float(3 + distance * ray_x), -1, float(8 + distance * ray_z)};
            CHECK_NEAR(pixel.depth, expected_depth(floor, view), 0.0000003f);
            if (size[0] == 840) {
                if (original[i].depth == 1) original[i] = pixel;
                else CHECK(original[i].color == pixel.color && original[i].depth == pixel.depth);
            }
        }
        EndDrawing();
        const auto retained = target.id;
        CHECK(!scene.resize(0, 0));
        CHECK(water::SceneValidation::target(scene).id == retained);
    }
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", controller_pause_resume_and_reset_preserve_scene_clock) {
    Context context;
    auto config = small_config();
    water::Simulation simulation(config);
    water::AppController app(simulation, config);
    water::WaterRenderer water(config.grid);
    water::SceneRenderer scene(config);
    CHECK(scene.resize(840, 560));
    const water::SpawnView view{camera, 1.5f};
    std::filesystem::create_directories("appearance-captures");
    const auto capture = [&](const char* path) {
        water.update(simulation);
        BeginDrawing();
        scene.draw(camera, water, context.boat(), app.drops(), app.sim_time(), 0.5f);
        std::vector<unsigned char> pixels(840 * 560 * 4);
        glReadPixels(0, 0, 840, 560, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        const auto image = LoadImageFromScreen();
        CHECK(ExportImage(image, path));
        UnloadImage(image);
        EndDrawing();
        return pixels;
    };
    const auto initial = capture("appearance-captures/clock-initial.png");
    for (int tick = 0; tick < 120; ++tick)
        app.frame(water::FixedClock::quantum, false, true, 8, 0.01f, 0.15f, view);
    CHECK(app.sim_time() == 1 && !app.drops().empty());
    const auto running = capture("appearance-captures/clock-running.png");
    app.frame(30, true, true, 8, 0.01f, 0.15f, view);
    CHECK(app.sim_time() == 1);
    CHECK(capture("appearance-captures/clock-paused.png") == running);
    std::filesystem::create_directories("pool-captures");
    const auto flights = app.drops();
    for (float depth : {.25f, 6.f}) {
        app.set_depth(depth);
        scene.set_depth(depth);
        CHECK(app.sim_time() == 1 && app.drops().size() == flights.size());
        for (std::size_t i = 0; i < flights.size(); ++i) {
            CHECK(app.drops()[i].spawn_height == flights[i].spawn_height);
            CHECK(app.drops()[i].arrival == flights[i].arrival);
            CHECK(app.drops()[i].x == flights[i].x && app.drops()[i].z == flights[i].z);
        }
        const auto path = depth == .25f ? "pool-captures/paused-h025.png" : "pool-captures/paused-h6.png";
        const auto changed = capture(path);
        app.frame(30, true, true, 8, std::min(.01f, .02f * depth), .15f, view);
        CHECK(capture(path) == changed);
    }
    app.set_depth(config.depth);
    scene.set_depth(config.depth);
    app.frame(water::FixedClock::quantum, false, true, 8, 0.01f, 0.15f, view);
    CHECK_NEAR(app.sim_time(), 1 + water::FixedClock::quantum, 1e-12);
    CHECK(capture("appearance-captures/clock-resumed.png") != running);
    app.reset();
    CHECK(app.sim_time() == 0 && app.drops().empty());
    CHECK(capture("appearance-captures/clock-reset.png") == initial);
    for (int tick = 0; tick < 120; ++tick)
        app.frame(water::FixedClock::quantum, false, true, 8, 0.01f, 0.15f, view);
    CHECK(capture("appearance-captures/clock-reseeded.png") == running);
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", rendered_acceptance_matrix) {
    Context context(1280, 800);
    struct Scenario {
        const char* name;
        int nx, ny, width, height;
        float time, mix, pitch, zoom, yaw;
        bool impacted;
        float depth = 3;
    };
    const Scenario scenarios[]{
        {"default-t0-r0", 512,512,1280,800, 0,0,0.72f,1.65f,0.65f,false},
        {"default-t0-r05",512,512,1280,800, 0,.5f,0.72f,1.65f,0.65f,false},
        {"default-t0-r1", 512,512,1280,800, 0,1,0.72f,1.65f,0.65f,false},
        {"default-t30-r0",512,512,1280,800,30,0,0.72f,1.65f,0.65f,true},
        {"default-t30-r05",512,512,1280,800,30,.5f,0.72f,1.65f,0.65f,true},
        {"default-t30-r1",512,512,1280,800,30,1,0.72f,1.65f,0.65f,true},
        {"wide-low-near-h025",96,64,840,560,30,.5f,0.15f,0.4f,1.8f,true,.25f},
        {"wide-high-far-h6",96,64,840,560,0,0,1.45f,4,3.4f,false,6},
        {"tall-high-near-h6",64,96,840,560,30,1,1.45f,0.4f,4.9f,true,6},
        {"tall-low-far-h025",64,96,1280,800,0,.5f,0.15f,4,0,false,.25f},
        {"floor-h6",96,64,1280,800,0,0,1.09f,1.65f,0.719f,false,6},
        {"floor-h025",64,96,840,560,0,0,1.09f,1.65f,0.719f,false,.25f}
    };
    std::filesystem::create_directories("pool-captures");
    for (const auto& scenario : scenarios) {
        SetWindowSize(scenario.width, scenario.height);
        CHECK(GetScreenWidth() == scenario.width && GetScreenHeight() == scenario.height);
        water::Config config;
        config.depth = scenario.depth;
        config.impact_strength = std::min(config.impact_strength, .02f * config.depth);
        config.grid.nx = scenario.nx; config.grid.ny = scenario.ny;
        if (scenario.nx != 512) config.impact_radius = 0.75f;
        water::Simulation simulation(config);
        water::AppController app(simulation, config);
        water::WaterRenderer water(config.grid);
        water::SceneRenderer scene(config);
        CHECK(scene.resize(GetRenderWidth(), GetRenderHeight()));
        const float distance = scenario.zoom * 20;
        const Camera3D view{{distance*std::cos(scenario.pitch)*std::sin(scenario.yaw), distance*std::sin(scenario.pitch),
                            distance*std::cos(scenario.pitch)*std::cos(scenario.yaw)},
                            {0,0,0}, {0,1,0}, 42, CAMERA_PERSPECTIVE};
        if (scenario.impacted) {
            const water::SpawnView spawn_view{view, float(scenario.width)/scenario.height};
            CHECK(app.spawn(10, 6, spawn_view));
            for (int tick = 0; tick < int(scenario.time * 120); ++tick)
                app.frame(water::FixedClock::quantum, false, true, 8, config.impact_strength, 0.15f, spawn_view);
            CHECK_NEAR(app.sim_time(), scenario.time, 1e-10);
        }
        water.update(simulation);
        BeginDrawing();
        scene.draw(view, water, context.boat(), app.drops(), scenario.time, scenario.mix);
        const auto image = LoadImageFromScreen();
        const auto path = std::string("pool-captures/") + scenario.name + ".png";
        CHECK(image.width == scenario.width && image.height == scenario.height);
        CHECK(ExportImage(image, path.c_str()));
        UnloadImage(image);
        EndDrawing();
        CHECK(glGetError() == GL_NO_ERROR);
    }
}

TEST_CASE("appearance", live_depth_replaces_only_basin_and_retains_old_geometry_on_failure) {
    Context context;
    water::SceneRenderer scene(small_config());
    CHECK(scene.resize(840, 560));
    const auto target = water::SceneValidation::target(scene);
    const auto drop = water::SceneValidation::drop_mesh(scene).vaoId;
    auto basin = water::SceneValidation::basin_mesh(scene).vaoId;
    scene.set_depth(1);
    CHECK(water::SceneValidation::basin_mesh(scene).vaoId == basin);
    water::appearance_fault = water::AppearanceFault::basin_mesh;
    CHECK_THROWS(scene.set_depth(3), std::runtime_error);
    CHECK(water::SceneValidation::basin_mesh(scene).vaoId == basin);
    CHECK(glIsVertexArray(basin));
    const auto failed = water::appearance_handles;
    CHECK(!glIsVertexArray(failed.vertex_array));
    CHECK(!glIsBuffer(failed.positions) && !glIsBuffer(failed.normals));
    water::appearance_fault = water::AppearanceFault::none;
    const Camera3D overhead{{0, 12, 0}, {0, 0, 0}, {0, 0, -1}, 45, CAMERA_PERSPECTIVE};
    for (float depth : {3.f, .25f, 1.5178088f}) {
        scene.set_depth(depth);
        CHECK(!glIsVertexArray(basin));
        basin = water::SceneValidation::basin_mesh(scene).vaoId;
        const auto& mesh = water::SceneValidation::basin_mesh(scene);
        float top = -100;
        for (int v = 0; v < mesh.vertexCount; ++v) {
            const float y = mesh.vertices[v * 3 + 1];
            top = std::max(top, y);
            if (v < mesh.vertexCount / 5 && mesh.normals[v * 3 + 1] > .9f)
                CHECK_NEAR(y, -depth, .000001f);
        }
        CHECK_NEAR(top, .12f, .000001f);
        CHECK(water::SceneValidation::drop_mesh(scene).vaoId == drop);
        CHECK(water::SceneValidation::target(scene).id == target.id);
        scene.set_depth(depth);
        CHECK(water::SceneValidation::basin_mesh(scene).vaoId == basin);
        BeginDrawing();
        water::SceneValidation::capture(scene, overhead, context.boat());
        const Vector3 point{0, -depth, 0};
        const auto pixel = sample_world(target, point, overhead);
        CHECK(pixel.color[3] == 255);
        CHECK_NEAR(pixel.depth, expected_depth(point, overhead), .00002f);
        EndDrawing();
    }
    CHECK(glGetError() == GL_NO_ERROR);
}

TEST_CASE("appearance", grass_surrounds_coping_without_covering_pool_or_capture) {
    Context context;
    const auto config = small_config();
    water::Simulation simulation(config);
    water::WaterRenderer renderer(config.grid);
    renderer.update(simulation);
    water::SceneRenderer scene(config);
    CHECK(scene.resize(840, 560));
    BeginDrawing();
    scene.draw(camera, renderer, context.boat(), {}, 0, 0);
    RenderTexture2D screen{};
    screen.texture.width = 840;
    screen.texture.height = 560;
    const Vector3 point{3, 0, 0};
    const auto grass = sample_world(screen, point);
    CHECK(grass.color[1] > grass.color[0] * 1.4f);
    CHECK_NEAR(grass.depth, expected_depth(point), .00002f);
    const Vector3 buried_wall{0, -.5f, 3.18f};
    const auto occluded = sample_world(screen, buried_wall);
    CHECK(occluded.color[1] > occluded.color[0] * 1.4f);
    CHECK(occluded.depth < expected_depth(buried_wall) - .00002f);
    const auto captured = sample_world(water::SceneValidation::target(scene), point);
    CHECK(captured.color[3] == 0 && captured.depth == 1);
    EndDrawing();
}

TEST_CASE("appearance", grass_mesh_has_upward_winding_and_exact_coping_hole) {
    Context context;
    const auto config = small_config();
    water::SceneRenderer scene(config);
    const auto& mesh = water::SceneValidation::grass_mesh(scene);
    CHECK(mesh.vertexCount == 24 && mesh.triangleCount == 8 && mesh.vaoId);
    const float hx = config.grid.lx / 2 + .18f, hz = config.grid.lz / 2 + .18f;
    const float outer = float(rlGetCullDistanceFar()) + 4 * std::max(config.grid.lx, config.grid.lz);
    for (int i = 0; i < mesh.vertexCount; i += 3) {
        bool left = true, right = true, front = true, back = true;
        for (int j = 0; j < 3; ++j) {
            const int n = 3 * (i + j);
            const float x = mesh.vertices[n], z = mesh.vertices[n + 2];
            CHECK(mesh.vertices[n + 1] == 0);
            CHECK(mesh.normals[n] == 0 && mesh.normals[n + 1] == 1 && mesh.normals[n + 2] == 0);
            CHECK(std::abs(x) <= outer && std::abs(z) <= outer);
            left &= x <= -hx; right &= x >= hx;
            front &= z <= -hz; back &= z >= hz;
        }
        CHECK(left || right || front || back);
        const float* a = mesh.vertices + 3 * i;
        const float* b = a + 3;
        const float* c = a + 6;
        CHECK((b[2]-a[2])*(c[0]-a[0]) - (b[0]-a[0])*(c[2]-a[2]) > 0);
    }
}

TEST_CASE("appearance", deep_floor_capture_remains_visible_at_drawable_scales) {
    Context context;
    auto config = small_config();
    const Camera3D overhead{{0, 12, 0}, {0, 0, 0}, {0, 0, -1}, 45, CAMERA_PERSPECTIVE};
    for (float depth : {.25f, 3.f, 6.f}) {
        config.depth = depth;
        config.impact_strength = std::min(config.impact_strength, .02f * depth);
        water::SceneRenderer scene(config);
        for (const auto size : {std::array<int, 2>{840, 560}, {1260, 840}}) {
            CHECK(scene.resize(size[0], size[1]));
            BeginDrawing();
            water::SceneValidation::capture(scene, overhead, context.boat());
            const auto target = water::SceneValidation::target(scene);
            const Vector3 point{0, -depth, 0};
            const auto floor = sample_world(target, point, overhead);
            CHECK(floor.color[3] == 255 && floor.color[0] > 35);
            CHECK_NEAR(floor.depth, expected_depth(point, overhead), .000002f);
            const auto outside = sample_world(target, {3, 0, 0}, overhead);
            CHECK(outside.color[3] == 0 && outside.depth == 1);
            EndDrawing();
        }
    }
    CHECK(glGetError() == GL_NO_ERROR);
}
