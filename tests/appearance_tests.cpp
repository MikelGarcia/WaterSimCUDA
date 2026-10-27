#include "test_support.hpp"
#include "appearance_validation.hpp"
#include "water_renderer.hpp"
#include "timeline.hpp"
#include <external/glad.h>
#include <rlgl.h>
#include <array>
#include <iostream>

namespace {

struct Context {
    Context() {
        SetTraceLogLevel(LOG_WARNING);
        SetConfigFlags(FLAG_WINDOW_HIDDEN);
        InitWindow(840, 560, "Water appearance validation");
        if (!IsWindowReady()) throw std::runtime_error("hidden OpenGL context could not be created");
        water::select_gl_cuda_device();
        std::cout << "GL renderer: " << glGetString(GL_RENDERER) << '\n';
    }
    ~Context() { water::appearance_fault = water::AppearanceFault::none; CloseWindow(); }
};

water::Config small_config() {
    water::Config config;
    config.grid = {8, 12, 4, 6};
    config.impact_radius = 1;
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

} // namespace

TEST_CASE("appearance", capture_contains_only_submerged_stone) {
    Context context;
    water::SceneRenderer scene(small_config());
    CHECK(scene.resize(840, 560));
    BeginDrawing();
    water::SceneValidation::capture(scene, camera);
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
    water::SceneValidation::capture(scene, overhead);
    for (const Vector3 point : {Vector3{-1.7f, -1, -2.7f}, Vector3{1.7f, -1, 2.7f}}) {
        const auto pixel = sample_world(target, point, overhead);
        CHECK(pixel.color[3] == 255);
        CHECK_NEAR(pixel.depth, expected_depth(point, overhead), 0.00002f);
    }
    const Camera3D grazing{{0, 1, 8}, {0, -0.4f, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};
    water::SceneValidation::capture(scene, grazing);
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
                            water::AppearanceFault::basin_mesh}) {
        water::appearance_fault = fault;
        CHECK_THROWS(water::SceneRenderer(small_config()), std::runtime_error);
        const auto handles = water::appearance_handles;
        CHECK(!glIsProgram(handles.stone_shader));
        CHECK(!glIsProgram(handles.water_shader));
        CHECK(!glIsVertexArray(handles.vertex_array));
        CHECK(!glIsBuffer(handles.positions));
        CHECK(!glIsBuffer(handles.normals));
        water::appearance_fault = water::AppearanceFault::none;
        water::SceneRenderer recovered(small_config());
        CHECK(recovered.resize(640, 480));
        CHECK(glGetError() == GL_NO_ERROR);
    }
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
    water::SceneValidation::capture(scene, camera);
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
    water::SceneValidation::capture(scene, camera);
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
        scene.draw(camera, water, {}, 0, 0.5f);
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
        CHECK(water_pixel.color[2] > water_pixel.color[0] + 20);
        const auto stone_pixel = sample_world(screen, {0, 0.12f, 3.09f});
        CHECK(stone_pixel.color[0] > 35);
        CHECK(std::abs(int(stone_pixel.color[0]) - int(stone_pixel.color[2])) < 20);
        const auto background = sample(screen, 5, GetRenderHeight() - 6);
        CHECK(background.color[0] == 225 && background.color[1] == 231 && background.color[2] == 234);
        DrawRectangle(0, 0, 20, 20, MAGENTA);
        rlDrawRenderBatchActive();
        unsigned char ui[4]{};
        glReadPixels(5, GetRenderHeight() - 6, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, ui);
        CHECK(ui[0] == 255 && ui[1] == 0 && ui[2] == 255);
        EndDrawing();
        CHECK(glGetError() == GL_NO_ERROR);
    }
}
