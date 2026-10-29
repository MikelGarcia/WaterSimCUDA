#include "test_support.hpp"
#include "boat.hpp"
#include <raylib.h>
#include <external/cgltf.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

namespace {
const char* asset = "assets/69ChrisCraftBoat/ChrisCraft.glb";
void require_asset(const char* path) {
    if (!std::filesystem::is_regular_file(path)) throw std::runtime_error(std::string("missing packaged boat asset: ") + path);
}
void check_mapping(const Model& model) {
    if (model.meshCount <= 0) throw std::runtime_error("boat model contains no meshes");
    for (int i = 0; i < model.meshCount; ++i)
        if (model.meshMaterial[i] < 0 || model.meshMaterial[i] >= model.materialCount)
            throw std::runtime_error("boat mesh has invalid material mapping");
}
struct Context {
    Context() {
        SetConfigFlags(FLAG_WINDOW_HIDDEN);
        InitWindow(900, 650, "Boat asset validation");
        CHECK(IsWindowReady());
    }
    ~Context() { CloseWindow(); }
};
}

TEST_CASE("boat_asset", packaged_boat_loads_and_renders) {
    require_asset(asset);
    require_asset("assets/69ChrisCraftBoat/BLENDSWAP_LICENSE.txt");
    Context context;
    Model model = LoadModel(asset);
    CHECK(model.meshCount > 0);
    check_mapping(model);
    for (int i = 0; i < model.meshCount; ++i) {
        const auto& mesh = model.meshes[i];
        CHECK(mesh.vertices != nullptr);
        CHECK(mesh.normals != nullptr);
        for (int j = 0; j < mesh.vertexCount * 3; ++j) {
            CHECK(std::isfinite(mesh.vertices[j]));
            CHECK(std::isfinite(mesh.normals[j]));
        }
        CHECK(IsTextureValid(model.materials[model.meshMaterial[i]].maps[MATERIAL_MAP_DIFFUSE].texture));
    }
    std::filesystem::create_directories("boat-asset-captures");
    const Camera3D views[] = {{{.64f,.46f,.64f},{0,0,0},{0,1,0},45,CAMERA_PERSPECTIVE},
                             {{0,.96f,0},{0,0,0},{1,0,0},45,CAMERA_PERSPECTIVE},
                             {{.16f,.14f,-.76f},{0,0,0},{0,1,0},45,CAMERA_PERSPECTIVE}};
    RenderTexture2D target = LoadRenderTexture(900, 650);
    for (int i = 0; i < 3; ++i) {
        BeginTextureMode(target);
        ClearBackground(Color{45,55,65,255});
        BeginMode3D(views[i]);
        DrawModel(model, {0,0,0}, 1, WHITE);
        EndMode3D();
        EndTextureMode();
        Image image = LoadImageFromTexture(target.texture);
        ImageFlipVertical(&image);
        Color* pixels = LoadImageColors(image);
        int boat_pixels = 0;
        for (int p = 0; p < image.width * image.height; ++p)
            if (pixels[p].r != 45 || pixels[p].g != 55 || pixels[p].b != 65) ++boat_pixels;
        UnloadImageColors(pixels);
        CHECK(boat_pixels > 10000);
        CHECK(ExportImage(image, TextFormat("boat-asset-captures/view-%d.png", i)));
        UnloadImage(image);
    }
    UnloadRenderTexture(target);
    std::cout << "boat meshes=" << model.meshCount << " materials=" << model.materialCount << '\n';
    UnloadModel(model);
}

TEST_CASE("boat_asset", fault_diagnostics_are_descriptive) {
    try { require_asset("missing-boat.glb"); CHECK(false); }
    catch (const std::runtime_error& e) { CHECK(std::string(e.what()).find("missing packaged boat asset") != std::string::npos); }
    Model invalid{};
    invalid.meshCount = 1;
    invalid.materialCount = 1;
    int mapping = 1;
    invalid.meshMaterial = &mapping;
    try { check_mapping(invalid); CHECK(false); }
    catch (const std::runtime_error& e) { CHECK(std::string(e.what()).find("invalid material mapping") != std::string::npos); }
}

TEST_CASE("boat_asset", measured_hull_and_baked_hierarchy) {
    require_asset(asset);
    cgltf_options options{};
    cgltf_data* data = nullptr;
    CHECK(cgltf_parse_file(&options, asset, &data) == cgltf_result_success);
    CHECK(cgltf_load_buffers(&options, data, asset) == cgltf_result_success);
    CHECK(cgltf_validate(data) == cgltf_result_success);
    bool found = false;
    for (cgltf_size n = 0; n < data->nodes_count; ++n) {
        auto& node = data->nodes[n];
        CHECK(!node.has_matrix);
        CHECK(!node.has_translation);
        CHECK(!node.has_rotation);
        CHECK(!node.has_scale);
        if (!node.name || std::string(node.name) != "BoatHull") continue;
        CHECK(node.mesh != nullptr);
        found = true;
        float lo[3]{1,1,1}, hi[3]{-1,-1,-1};
        float landmark_distance[16];
        std::fill_n(landmark_distance, 16, std::numeric_limits<float>::max());
        for (cgltf_size p = 0; p < node.mesh->primitives_count; ++p) {
            auto& primitive = node.mesh->primitives[p];
            for (cgltf_size a = 0; a < primitive.attributes_count; ++a) {
                if (primitive.attributes[a].type != cgltf_attribute_type_position) continue;
                auto* positions = primitive.attributes[a].data;
                for (cgltf_size v = 0; v < positions->count; ++v) {
                    float xyz[3];
                    CHECK(cgltf_accessor_read_float(positions, v, xyz, 3));
                    for (int k = 0; k < 3; ++k) {
                        lo[k] = std::min(lo[k], xyz[k]);
                        hi[k] = std::max(hi[k], xyz[k]);
                    }
                }
                for (cgltf_size t = 0; t < primitive.indices->count; t += 3) {
                    float triangle[3][3];
                    for (int c = 0; c < 3; ++c)
                        CHECK(cgltf_accessor_read_float(positions,
                            cgltf_accessor_read_index(primitive.indices, t + c), triangle[c], 3));
                    for (int c = 0; c < 3; ++c) {
                        const float* a = triangle[c];
                        const float* b = triangle[(c + 1) % 3];
                        if (a[1] * b[1] >= 0) continue;
                        const float fraction = -a[1] / (b[1] - a[1]);
                        const float x = a[0] + fraction * (b[0] - a[0]);
                        const float z = a[2] + fraction * (b[2] - a[2]);
                        for (int k = 0; k < water::boat_waterline_count; ++k)
                            landmark_distance[k] = std::min(landmark_distance[k],
                                std::hypot(x - water::boat_waterline[k].x, z - water::boat_waterline[k].z));
                    }
                }
            }
        }
        CHECK_NEAR(hi[0]-lo[0], .6f, .0003f);
        CHECK_NEAR(hi[0]-lo[0], water::boat_geometry.length, .0003f);
        CHECK_NEAR(hi[2]-lo[2], water::boat_geometry.beam, .00001f);
        CHECK_NEAR((hi[2]-lo[2])/(hi[0]-lo[0]), 3.199835777282715/8.443901062011719, .00037896);
        CHECK_NEAR((hi[0]+lo[0])/2, 0, .001f);
        CHECK_NEAR((hi[2]+lo[2])/2, 0, .001f);
        CHECK_NEAR(lo[1], (-1.4154112339019775+.5)*.07105720396219954, .001f);
        CHECK_NEAR(hi[1], (.3108023405075073+.5)*.07105720396219954, .001f);
        CHECK(water::boat_waterline_count >= 3 && water::boat_waterline_count <= 16);
        for (int k = 0; k < water::boat_waterline_count; ++k) CHECK(landmark_distance[k] < .001f);
        CHECK_NEAR(water::boat_geometry.spawn_clearance,
                   std::max(.01f, .1f * water::boat_geometry.beam), 1e-8f);
    }
    CHECK(found);
    CHECK(data->images_count == 0);
    cgltf_free(data);
}
