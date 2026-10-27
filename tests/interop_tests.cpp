#include "test_support.hpp"
#include "interop_validation.hpp"
#include "validation.hpp"

#include <rlgl.h>
#include <external/glad.h>
#include <algorithm>
#include <iostream>

namespace {

struct Context {
    Context() {
        SetTraceLogLevel(LOG_WARNING);
        SetConfigFlags(FLAG_WINDOW_HIDDEN);
        InitWindow(128, 128, "Water interop validation");
        if (!IsWindowReady()) throw std::runtime_error("hidden OpenGL context could not be created");
        try {
            const int device = water::select_gl_cuda_device();
            cudaDeviceProp properties{};
            CHECK(cudaGetDeviceProperties(&properties, device) == cudaSuccess);
            std::cout << "GL renderer: " << glGetString(GL_RENDERER)
                      << "; CUDA device " << device << ": " << properties.name << '\n';
        } catch (...) {
            CloseWindow();
            throw;
        }
    }
    ~Context() { CloseWindow(); }
};

water::Config small_config() {
    water::Config config;
    config.grid = {8, 12, 4.0f, 6.0f};
    config.impact_radius = 1.0f;
    return config;
}

std::vector<water::Vertex> read_buffer(const water::WaterRenderer& renderer, int slot) {
    const auto& mesh = water::RendererValidation::mesh(renderer);
    std::vector<water::Vertex> vertices(mesh.vertexCount);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vboId[slot]);
    GLint bytes = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bytes);
    CHECK(bytes == vertices.size() * sizeof(water::Vertex));
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, bytes, vertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CHECK(glGetError() == GL_NO_ERROR);
    return vertices;
}

void draw(const water::WaterRenderer& renderer) {
    BeginDrawing();
    ClearBackground(BLACK);
    Camera3D camera{{5, 7, 9}, {0, 0, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};
    BeginMode3D(camera);
    const Matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    renderer.draw({rlGetShaderIdDefault(), rlGetShaderLocsDefault()}, identity);
    EndMode3D();
    unsigned char center[4]{};
    glReadPixels(64, 64, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, center);
    CHECK(center[0] > 0 && center[1] > 0 && center[2] > 0);
    EndDrawing();
    CHECK(glGetError() == GL_NO_ERROR);
    CHECK(cudaGetLastError() == cudaSuccess);
}

void check_extents(const std::vector<water::Vertex>& positions, const water::Grid& grid) {
    float min_x = grid.lx, max_x = -grid.lx, min_z = grid.lz, max_z = -grid.lz;
    for (const auto& p : positions) {
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
    }
    CHECK_NEAR(min_x, -grid.lx / 2, 1e-6f);
    CHECK_NEAR(max_x, grid.lx / 2, 1e-6f);
    CHECK_NEAR(min_z, -grid.lz / 2, 1e-6f);
    CHECK_NEAR(max_z, grid.lz / 2, 1e-6f);
    const auto& last = positions.back();
    CHECK_NEAR(last.x, grid.lx / 2, 1e-6f);
    CHECK_NEAR(last.z, grid.lz / 2, 1e-6f);
    for (std::size_t i = 0; i < positions.size(); i += 3) {
        const auto a = positions[i], b = positions[i + 1], c = positions[i + 2];
        CHECK((b.z - a.z) * (c.x - a.x) - (b.x - a.x) * (c.z - a.z) > 0);
    }
}

} // namespace

TEST_CASE("interop", flat_rectangular_mesh_covers_every_cell) {
    Context context;
    const auto config = small_config();
    water::Simulation simulation(config);
    water::WaterRenderer renderer(config.grid);
    renderer.update(simulation);
    const auto positions = read_buffer(renderer, 0), normals = read_buffer(renderer, 2);
    CHECK(positions.size() == 6 * config.grid.nx * config.grid.ny);
    check_extents(positions, config.grid);
    for (std::size_t i = 0; i < positions.size(); ++i) {
        CHECK_NEAR(positions[i].y, 0.0f, 1e-6f);
        CHECK_NEAR(normals[i].x, 0.0f, 1e-6f);
        CHECK_NEAR(normals[i].y, 1.0f, 1e-6f);
        CHECK_NEAR(normals[i].z, 0.0f, 1e-6f);
    }
    draw(renderer);
}

TEST_CASE("interop", tilted_field_interpolates_nodes_and_boundary_normals) {
    Context context;
    auto config = small_config();
    config.grid = {8, 12, 4.0f, 4.8f};
    water::Simulation simulation(config);
    auto state = water::download(simulation);
    const auto& g = config.grid;
    const float dx = g.lx / g.nx, dz = g.lz / g.ny;
    for (int z = 0; z < g.ny; ++z) for (int x = 0; x < g.nx; ++x)
        state.eta[z * g.nx + x] = 0.03f * (x + 0.5f) * dx - 0.02f * (z + 0.5f) * dz;
    water::upload(simulation, state);
    water::WaterRenderer renderer(g);
    renderer.update(simulation);
    const auto positions = read_buffer(renderer, 0), normals = read_buffer(renderer, 2);
    auto node_height = [&](int x, int z) {
        return 0.03f * std::clamp(x * dx, dx / 2, g.lx - dx / 2) -
               0.02f * std::clamp(z * dz, dz / 2, g.lz - dz / 2);
    };
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const int x = static_cast<int>(std::lround((positions[i].x + g.lx / 2) / dx));
        const int z = static_cast<int>(std::lround((positions[i].z + g.lz / 2) / dz));
        CHECK_NEAR(positions[i].y, node_height(x, z), 1e-6f);
        const int left = std::max(0, x - 1), right = std::min(g.nx, x + 1);
        const int near = std::max(0, z - 1), far = std::min(g.ny, z + 1);
        const float sx = (node_height(right, z) - node_height(left, z)) / ((right - left) * dx);
        const float sz = (node_height(x, far) - node_height(x, near)) / ((far - near) * dz);
        const float length = std::sqrt(sx * sx + 1 + sz * sz);
        CHECK_NEAR(normals[i].x, -sx / length, 1e-4f);
        CHECK_NEAR(normals[i].y, 1 / length, 1e-4f);
        CHECK_NEAR(normals[i].z, -sz / length, 1e-4f);
    }
    check_extents(positions, g);
    draw(renderer);
}

TEST_CASE("interop", default_mesh_is_unindexed_and_changes_after_impact) {
    Context context;
    const water::Config config;
    water::Simulation simulation(config);
    water::WaterRenderer renderer(config.grid);
    const auto& mesh = water::RendererValidation::mesh(renderer);
    CHECK(mesh.vertexCount == 6 * 512 * 512);
    CHECK(mesh.indices == nullptr);
    CHECK(mesh.vboId[6] == 0);
    renderer.update(simulation);
    const auto flat = read_buffer(renderer, 0);
    const water::Impact impact{10, 6, config.impact_radius, config.impact_strength};
    simulation.apply_impacts(&impact, 1);
    renderer.update(simulation);
    const auto changed = read_buffer(renderer, 0);
    bool moved = false;
    for (std::size_t i = 0; i < flat.size(); ++i) moved |= std::abs(changed[i].y - flat[i].y) > 1e-5f;
    CHECK(moved);
    check_extents(changed, config.grid);
    CHECK(read_buffer(renderer, 2).size() == changed.size());
    draw(renderer);
}

TEST_CASE("interop", repeated_lifetimes_release_all_ownership) {
    Context context;
    for (int i = 0; i < 20; ++i) {
        const auto config = small_config();
        unsigned position_buffer = 0, normal_buffer = 0, vertex_array = 0;
        {
            water::Simulation simulation(config);
            water::WaterRenderer renderer(config.grid);
            const auto& mesh = water::RendererValidation::mesh(renderer);
            position_buffer = mesh.vboId[0];
            normal_buffer = mesh.vboId[2];
            vertex_array = mesh.vaoId;
            renderer.update(simulation);
            draw(renderer);
            CHECK(water::interop_registrations == 2);
            CHECK(water::interop_mappings == 0);
        }
        CHECK(water::interop_registrations == 0);
        CHECK(water::interop_mappings == 0);
        CHECK(!glIsBuffer(position_buffer));
        CHECK(!glIsBuffer(normal_buffer));
        CHECK(!glIsVertexArray(vertex_array));
        CHECK(glGetError() == GL_NO_ERROR);
        CHECK(cudaGetLastError() == cudaSuccess);
    }
}

TEST_CASE("interop", partial_failures_release_resources_and_allow_recovery) {
    Context context;
    const auto config = small_config();
    water::Simulation simulation(config);
    water::interop_fault = water::InteropFault::after_first_registration;
    CHECK_THROWS(water::WaterRenderer(config.grid), std::runtime_error);
    CHECK(water::interop_registrations == 0);
    CHECK(water::interop_mappings == 0);
    water::interop_fault = water::InteropFault::none;
    {
        water::WaterRenderer renderer(config.grid);
        water::interop_fault = water::InteropFault::after_mapping;
        CHECK_THROWS(renderer.update(simulation), std::runtime_error);
        CHECK(water::interop_mappings == 0);
        water::interop_fault = water::InteropFault::none;
        renderer.update(simulation);
        draw(renderer);
    }
    CHECK(water::interop_registrations == 0);
    CHECK(water::interop_mappings == 0);
    water::WaterRenderer recovered(config.grid);
    recovered.update(simulation);
    draw(recovered);
}

TEST_CASE("interop", incompatible_adapter_selection_is_rejected) {
    CHECK_THROWS(water::detail::choose_gl_cuda_device(nullptr, 0), std::runtime_error);
    const int devices[]{3, 7};
    CHECK(water::detail::choose_gl_cuda_device(devices, 2) == 3);
}

TEST_CASE("interop", invalid_surface_requests_preserve_renderer_ownership) {
    Context context;
    const auto config = small_config();
    water::Simulation simulation(config);
    water::WaterRenderer renderer(config.grid);
    const Matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    CHECK_THROWS(renderer.draw({rlGetShaderIdDefault(), rlGetShaderLocsDefault()}, identity),
                 std::logic_error);
    CHECK_THROWS(simulation.write_surface(nullptr, nullptr, 6 * 8 * 12), std::invalid_argument);
    auto other_config = config;
    other_config.grid.nx = 12;
    water::Simulation other(other_config);
    CHECK_THROWS(renderer.update(other), std::invalid_argument);
    CHECK(water::interop_mappings == 0);
    renderer.update(simulation);
    draw(renderer);
}
