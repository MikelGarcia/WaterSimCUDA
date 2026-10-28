#pragma once

#include "config.hpp"
#include <raylib.h>
#include <string>
#include <vector>

namespace water {

class WaterRenderer;
struct Drop;

class SceneRenderer {
public:
    void set_depth(float depth);
    explicit SceneRenderer(const Config& config);
    ~SceneRenderer();
    SceneRenderer(const SceneRenderer&) = delete;
    SceneRenderer& operator=(const SceneRenderer&) = delete;

    bool resize(int drawable_width, int drawable_height);
    void draw(const Camera3D&, const WaterRenderer&, const std::vector<Drop>&,
              double sim_time, float reflection);

private:
#ifdef WATER_APPEARANCE_VALIDATION
    friend struct SceneValidation;
#endif
    void capture(const Camera3D& camera);
    static std::string fragment_source(const char* fragment);
    static Shader load_shader(const char* vertex, const char* fragment);
    void draw_basin(bool submerged);
    void release() noexcept;

    Config config_;
    Mesh basin_{};
    Mesh drop_{};
    Mesh grass_{};
    Shader grass_shader_{};
    Shader stone_shader_{};
    Shader water_shader_{};
    Shader sky_shader_{};
    Shader drop_shader_{};
    int drop_eye_ = -1, drop_time_ = -1;
    struct {
        int eye, view, projection, color, depth, size, planes, time, reflection;
    } water_locations_{};
    struct {
        int forward, right, up, size, aspect, fov, time;
    } sky_locations_{};
    RenderTexture2D capture_{};
    Matrix capture_view_{}, capture_projection_{};
    Matrix view_{}, projection_{};
};

} // namespace water
