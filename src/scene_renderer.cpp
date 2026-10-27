#include "scene_renderer.hpp"
#include "water_renderer.hpp"
#include "timeline.hpp"
#include <external/glad.h>
#include <rlgl.h>
#include <config.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef WATER_APPEARANCE_VALIDATION
#include "appearance_validation.hpp"
#endif

namespace water {

#ifdef WATER_APPEARANCE_VALIDATION
AppearanceFault appearance_fault = AppearanceFault::none;
AppearanceHandles appearance_handles{};

static void checkpoint(AppearanceFault stage) {
    if (appearance_fault == stage) throw std::runtime_error("injected scene allocation failure");
}
#define SCENE_CHECKPOINT(stage) checkpoint(AppearanceFault::stage)
#else
#define SCENE_CHECKPOINT(stage) ((void)0)
#endif

namespace {

constexpr Matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

Shader load_shader(const char* vertex, const char* fragment) {
    const auto assets = std::filesystem::path(GetApplicationDirectory()) / "assets";
    const auto vs = (assets / vertex).string(), fs = (assets / fragment).string();
    if (!FileExists(vs.c_str()) || !FileExists(fs.c_str()))
        throw std::runtime_error("scene shader assets are missing: " + fs);
    Shader shader = LoadShader(vs.c_str(), fs.c_str());
    if (!shader.id || shader.id == rlGetShaderIdDefault()) {
        UnloadShader(shader);
        throw std::runtime_error("could not load scene shader: " + fs);
    }
    return shader;
}

void unload_target(RenderTexture2D target) {
    if (target.texture.id) rlUnloadTexture(target.texture.id);
    // Pinned rlgl owns and deletes the framebuffer's attached depth texture.
    if (target.id) rlUnloadFramebuffer(target.id);
}

void begin_3d(const Camera3D& camera) {
    BeginMode3D(camera);
    const double top = rlGetCullDistanceNear() * std::tan(camera.fovy * 0.5 * DEG2RAD);
    const double right = top * GetScreenWidth() / GetScreenHeight();
    rlMatrixMode(RL_PROJECTION);
    rlLoadIdentity();
    rlFrustum(-right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar());
    rlMatrixMode(RL_MODELVIEW);
}

} // namespace

SceneRenderer::SceneRenderer(const Config& config) : config_(config) {
    if (!IsWindowReady()) throw std::runtime_error("scene renderer requires a current OpenGL context");
    try {
#ifdef WATER_APPEARANCE_VALIDATION
        appearance_handles = {};
#endif
        stone_shader_ = load_shader("basin.vs", "basin.fs");
#ifdef WATER_APPEARANCE_VALIDATION
        appearance_handles.stone_shader = stone_shader_.id;
#endif
        SCENE_CHECKPOINT(stone_shader);
        water_shader_ = load_shader("water.vs", "water.fs");
#ifdef WATER_APPEARANCE_VALIDATION
        appearance_handles.water_shader = water_shader_.id;
#endif
        SCENE_CHECKPOINT(water_shader);
        const float thickness = 0.18f, rim = 0.12f;
        const auto& g = config.grid;
        const float wall_y = (rim - config.depth) / 2;
        const Vector3 centers[] = {{0, -config.depth - thickness / 2, 0},
            {-(g.lx + thickness) / 2, wall_y, 0}, {(g.lx + thickness) / 2, wall_y, 0},
            {0, wall_y, -(g.lz + thickness) / 2}, {0, wall_y, (g.lz + thickness) / 2}};
        const Vector3 sizes[] = {{g.lx + 2 * thickness, thickness, g.lz + 2 * thickness},
            {thickness, config.depth + rim, g.lz + 2 * thickness},
            {thickness, config.depth + rim, g.lz + 2 * thickness},
            {g.lx, config.depth + rim, thickness}, {g.lx, config.depth + rim, thickness}};
        Mesh cube = GenMeshCube(1, 1, 1);
        const int box_vertices = cube.triangleCount * 3;
        basin_.vertexCount = 5 * box_vertices;
        basin_.triangleCount = 5 * cube.triangleCount;
        const auto bytes = basin_.vertexCount * 3 * sizeof(float);
        basin_.vertices = static_cast<float*>(MemAlloc(static_cast<unsigned>(bytes)));
        basin_.normals = static_cast<float*>(MemAlloc(static_cast<unsigned>(bytes)));
        if (!basin_.vertices || !basin_.normals) {
            UnloadMesh(cube);
            throw std::bad_alloc();
        }
        for (int box = 0; box < 5; ++box) {
            const int offset = box * box_vertices * 3;
            for (int v = 0; v < box_vertices; ++v) {
                const int source = cube.indices[v] * 3;
                std::memcpy(basin_.normals + offset + 3 * v, cube.normals + source, 3 * sizeof(float));
                basin_.vertices[offset + 3 * v] = cube.vertices[source] * sizes[box].x + centers[box].x;
                basin_.vertices[offset + 3 * v + 1] = cube.vertices[source + 1] * sizes[box].y + centers[box].y;
                basin_.vertices[offset + 3 * v + 2] = cube.vertices[source + 2] * sizes[box].z + centers[box].z;
            }
        }
        UnloadMesh(cube);
        UploadMesh(&basin_, false);
        if (!basin_.vaoId || glGetError() != GL_NO_ERROR)
            throw std::runtime_error("could not upload stone basin mesh");
#ifdef WATER_APPEARANCE_VALIDATION
        appearance_handles.vertex_array = basin_.vaoId;
        appearance_handles.positions = basin_.vboId[0];
        appearance_handles.normals = basin_.vboId[2];
#endif
        SCENE_CHECKPOINT(basin_mesh);
    } catch (...) { release(); throw; }
}

SceneRenderer::~SceneRenderer() { release(); }

bool SceneRenderer::resize(int width, int height) {
    if (width <= 0 || height <= 0) return false;
    if (capture_.texture.width == width && capture_.texture.height == height) return true;
    GLint draw_fbo = 0, read_fbo = 0, texture = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_fbo);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &texture);
    RenderTexture2D next{};
    bool depth_attached = false, success = false;
    try {
#ifdef WATER_APPEARANCE_VALIDATION
        appearance_handles = {};
#endif
        next.id = rlLoadFramebuffer();
#ifdef WATER_APPEARANCE_VALIDATION
        appearance_handles.framebuffer = next.id;
#endif
        if (!next.id) throw std::runtime_error("allocate refraction framebuffer");
        SCENE_CHECKPOINT(framebuffer);
        next.texture = {rlLoadTexture(nullptr, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1),
                        width, height, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
#ifdef WATER_APPEARANCE_VALIDATION
        appearance_handles.color = next.texture.id;
#endif
        if (!next.texture.id) throw std::runtime_error("allocate refraction color texture");
        SetTextureFilter(next.texture, TEXTURE_FILTER_POINT);
        SetTextureWrap(next.texture, TEXTURE_WRAP_CLAMP);
        SCENE_CHECKPOINT(color_texture);
        next.depth = {rlLoadTextureDepth(width, height, false), width, height, 1, 0};
#ifdef WATER_APPEARANCE_VALIDATION
        appearance_handles.depth = next.depth.id;
#endif
        if (!next.depth.id) throw std::runtime_error("allocate refraction depth texture");
        SCENE_CHECKPOINT(depth_texture);
        rlFramebufferAttach(next.id, next.texture.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0);
        SCENE_CHECKPOINT(color_attachment);
        rlFramebufferAttach(next.id, next.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0);
        depth_attached = true;
        SCENE_CHECKPOINT(depth_attachment);
        if (!rlFramebufferComplete(next.id)) throw std::runtime_error("incomplete refraction framebuffer");
        glBindFramebuffer(GL_FRAMEBUFFER, next.id);
        GLint depth_type = 0;
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                            GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &depth_type);
        if (depth_type != GL_TEXTURE || glGetError() != GL_NO_ERROR)
            throw std::runtime_error("refraction requires a sampled depth texture");
        SCENE_CHECKPOINT(complete);
        unload_target(capture_);
        capture_ = next;
        success = true;
    } catch (const std::exception& error) {
        if (next.texture.id) rlUnloadTexture(next.texture.id);
        if (depth_attached) rlUnloadFramebuffer(next.id);
        else {
            if (next.depth.id) rlUnloadTexture(next.depth.id);
            if (next.id) glDeleteFramebuffers(1, &next.id);
        }
        TraceLog(LOG_WARNING, "SCENE: %s", error.what());
    }
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    return success;
}

void SceneRenderer::draw_basin(bool submerged) {
    const int mode = submerged ? 1 : 0;
    SetShaderValue(stone_shader_, GetShaderLocation(stone_shader_, "submergedOnly"), &mode, SHADER_UNIFORM_INT);
    MaterialMap maps[MAX_MATERIAL_MAPS]{};
    maps[MATERIAL_MAP_DIFFUSE].color = WHITE;
    maps[MATERIAL_MAP_DIFFUSE].texture.id = rlGetTextureIdDefault();
    Material material{};
    material.shader = stone_shader_;
    material.maps = maps;
    DrawMesh(basin_, material, identity);
}

void SceneRenderer::capture(const Camera3D& camera) {
    if (!capture_.id) return;
    rlDrawRenderBatchActive();
    GLint draw_fbo = 0, read_fbo = 0, viewport[4]{}, depth_func = 0;
    GLboolean color_mask[4]{}, depth_mask = GL_TRUE;
    GLfloat clear_color[4]{};
    GLdouble clear_depth = 1;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_fbo);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo);
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetIntegerv(GL_DEPTH_FUNC, &depth_func);
    glGetBooleanv(GL_COLOR_WRITEMASK, color_mask);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &depth_mask);
    glGetFloatv(GL_COLOR_CLEAR_VALUE, clear_color);
    glGetDoublev(GL_DEPTH_CLEAR_VALUE, &clear_depth);
    const bool depth_test = glIsEnabled(GL_DEPTH_TEST), scissor = glIsEnabled(GL_SCISSOR_TEST);
    const Matrix old_view = rlGetMatrixModelview(), old_projection = rlGetMatrixProjection();
    glBindFramebuffer(GL_FRAMEBUFFER, capture_.id);
    glViewport(0, 0, capture_.texture.width, capture_.texture.height);
    glDisable(GL_SCISSOR_TEST);
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glClearColor(0, 0, 0, 0);
    glClearDepth(1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    begin_3d(camera);
    capture_view_ = rlGetMatrixModelview();
    capture_projection_ = rlGetMatrixProjection();
    glDepthFunc(GL_LESS);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    draw_basin(false);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthFunc(GL_LEQUAL);
    draw_basin(true);
    EndMode3D();
    glDepthFunc(depth_func);
    glDepthMask(depth_mask);
    glColorMask(color_mask[0], color_mask[1], color_mask[2], color_mask[3]);
    if (depth_test) glEnable(GL_DEPTH_TEST);
    if (scissor) glEnable(GL_SCISSOR_TEST);
    glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);
    glClearDepth(clear_depth);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo);
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    rlSetMatrixModelview(old_view);
    rlSetMatrixProjection(old_projection);
}

void SceneRenderer::draw(const Camera3D& camera, const WaterRenderer& water,
                         const std::vector<Drop>&, double, float) {
    capture(camera);
    ClearBackground(Color{225, 231, 234, 255});
    rlViewport(0, 0, GetRenderWidth(), GetRenderHeight());
    begin_3d(camera);
    view_ = rlGetMatrixModelview();
    projection_ = rlGetMatrixProjection();
    draw_basin(false);
    SetShaderValue(water_shader_, GetShaderLocation(water_shader_, "eyePosition"), &camera.position, SHADER_UNIFORM_VEC3);
    water.draw(water_shader_, identity);
    EndMode3D();
}

void SceneRenderer::release() noexcept {
    unload_target(capture_);
    UnloadMesh(basin_);
    if (stone_shader_.id) UnloadShader(stone_shader_);
    if (water_shader_.id) UnloadShader(water_shader_);
    capture_ = {};
    basin_ = {};
    stone_shader_ = {};
    water_shader_ = {};
}

} // namespace water
