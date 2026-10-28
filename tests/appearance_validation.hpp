#pragma once

#include "scene_renderer.hpp"

namespace water {

enum class AppearanceFault {
    none, stone_shader, water_shader, basin_mesh, framebuffer, color_texture,
    depth_texture, color_attachment, depth_attachment, complete
};
extern AppearanceFault appearance_fault;
struct AppearanceHandles {
    unsigned framebuffer = 0, color = 0, depth = 0;
    unsigned stone_shader = 0, water_shader = 0, vertex_array = 0;
    unsigned positions = 0, normals = 0;
};
extern AppearanceHandles appearance_handles;

struct SceneValidation {
    static const Mesh& basin_mesh(const SceneRenderer& scene) { return scene.basin_; }
    static const Mesh& drop_mesh(const SceneRenderer& scene) { return scene.drop_; }
    static std::string fragment_source(const char* fragment) { return SceneRenderer::fragment_source(fragment); }
    static RenderTexture2D target(const SceneRenderer& scene) { return scene.capture_; }
    static void capture(SceneRenderer& scene, const Camera3D& camera) { scene.capture(camera); }
    static Matrix capture_projection(const SceneRenderer& scene) { return scene.capture_projection_; }
    static Matrix projection(const SceneRenderer& scene) { return scene.projection_; }
    static Matrix capture_view(const SceneRenderer& scene) { return scene.capture_view_; }
    static Matrix view(const SceneRenderer& scene) { return scene.view_; }
};

} // namespace water
