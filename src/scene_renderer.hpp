#pragma once

#include "config.hpp"
#include <raylib.h>
#include <vector>

namespace water {

class WaterRenderer;
struct Drop;

class SceneRenderer {
public:
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
    void draw_basin(bool submerged);
    void release() noexcept;

    Config config_;
    Mesh basin_{};
    Shader stone_shader_{};
    Shader water_shader_{};
    RenderTexture2D capture_{};
    Matrix capture_view_{}, capture_projection_{};
    Matrix view_{}, projection_{};
};

} // namespace water
