#include "water_renderer.hpp"

#include <external/glad.h>
#include <cuda_gl_interop.h>
#include <rlgl.h>
#include <config.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef WATER_INTEROP_VALIDATION
#include "interop_validation.hpp"
#endif

namespace water {

#ifdef WATER_INTEROP_VALIDATION
InteropFault interop_fault = InteropFault::none;
int interop_registrations = 0;
int interop_mappings = 0;
#endif

namespace {

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void check_cleanup(cudaError_t error, const char* operation) noexcept {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    }
}

} // namespace

namespace detail {

int choose_gl_cuda_device(const int* devices, unsigned count) {
    if (count == 0) throw std::runtime_error("the OpenGL context has no associated CUDA device");
    return devices[0];
}

} // namespace detail

int select_gl_cuda_device() {
    unsigned count = 0;
    int devices[32]{};
    const auto error = cudaGLGetDevices(&count, devices, 32, cudaGLDeviceListAll);
    if (error == cudaErrorNoDevice) {
        throw std::runtime_error("the OpenGL context has no associated CUDA device");
    }
    check(error, "query OpenGL CUDA devices");
    const int device = detail::choose_gl_cuda_device(devices, count);
    check(cudaSetDevice(device), "select OpenGL CUDA device");
    return device;
}

WaterRenderer::WaterRenderer(const Grid& grid) : grid_(grid) {
    if (!IsWindowReady()) throw std::runtime_error("water renderer requires a current OpenGL context");
    if (grid.nx < 8 || grid.nx > 2048 || grid.ny < 8 || grid.ny > 2048 ||
        !std::isfinite(grid.lx) || grid.lx <= 0 || !std::isfinite(grid.lz) || grid.lz <= 0) {
        throw std::invalid_argument("water renderer requires valid grid dimensions");
    }
    static_assert(sizeof(Vertex) == 3 * sizeof(float));
    mesh_.vertexCount = 6 * grid.nx * grid.ny;
    mesh_.triangleCount = 2 * grid.nx * grid.ny;
    try {
        mesh_.vboId = static_cast<unsigned*>(MemAlloc(MAX_MESH_VERTEX_BUFFERS * sizeof(unsigned)));
        if (!mesh_.vboId) throw std::bad_alloc();
        std::memset(mesh_.vboId, 0, MAX_MESH_VERTEX_BUFFERS * sizeof(unsigned));
        mesh_.vaoId = rlLoadVertexArray();
        if (!mesh_.vaoId) throw std::runtime_error("allocate water vertex array");
        rlEnableVertexArray(mesh_.vaoId);
        const int slots[2] = {RL_DEFAULT_SHADER_ATTRIB_LOCATION_POSITION,
                              RL_DEFAULT_SHADER_ATTRIB_LOCATION_NORMAL};
        for (int i = 0; i < 2; ++i) {
            const int slot = slots[i];
            mesh_.vboId[slot] = rlLoadVertexBuffer(nullptr, mesh_.vertexCount * sizeof(Vertex), true);
            rlSetVertexAttribute(slot, 3, RL_FLOAT, false, sizeof(Vertex), 0);
            rlEnableVertexAttribute(slot);
            if (!mesh_.vboId[slot] || glGetError() != GL_NO_ERROR) {
                throw std::runtime_error("allocate water vertex buffer");
            }
        }
        rlDisableVertexArray();
        rlDisableVertexBuffer();
        for (int i = 0; i < 2; ++i) {
            check(cudaGraphicsGLRegisterBuffer(&resources_[i], mesh_.vboId[slots[i]],
                                               cudaGraphicsRegisterFlagsWriteDiscard),
                  "register water vertex buffer");
#ifdef WATER_INTEROP_VALIDATION
            ++interop_registrations;
            if (i == 0 && interop_fault == InteropFault::after_first_registration) {
                throw std::runtime_error("injected failure after first registration");
            }
#endif
        }
    } catch (...) {
        release();
        throw;
    }
}

WaterRenderer::~WaterRenderer() { release(); }

void WaterRenderer::unmap() {
    check(cudaGraphicsUnmapResources(2, resources_, mapped_stream_), "unmap water vertex buffers");
    mapped_ = false;
#ifdef WATER_INTEROP_VALIDATION
    interop_mappings -= 2;
#endif
    check(cudaStreamSynchronize(mapped_stream_), "complete water ownership transfer");
    mapped_stream_ = nullptr;
}

void WaterRenderer::update(Simulation& simulation) {
    const auto& grid = simulation.config().grid;
    if (grid.nx != grid_.nx || grid.ny != grid_.ny || grid.lx != grid_.lx || grid.lz != grid_.lz) {
        throw std::invalid_argument("renderer and simulation grids must match");
    }
    ready_ = false;
    mapped_stream_ = simulation.stream();
    check(cudaGraphicsMapResources(2, resources_, mapped_stream_), "map water vertex buffers");
    mapped_ = true;
#ifdef WATER_INTEROP_VALIDATION
    interop_mappings += 2;
#endif
    try {
#ifdef WATER_INTEROP_VALIDATION
        if (interop_fault == InteropFault::after_mapping) {
            throw std::runtime_error("injected failure during mapping");
        }
#endif
        Vertex* pointers[2]{};
        const std::size_t expected_bytes = mesh_.vertexCount * sizeof(Vertex);
        for (int i = 0; i < 2; ++i) {
            std::size_t bytes = 0;
            check(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pointers[i]),
                                                        &bytes, resources_[i]),
                  "get mapped water vertex buffer");
            if (bytes < expected_bytes) throw std::runtime_error("mapped water buffer is too small");
        }
        simulation.write_surface(pointers[0], pointers[1], mesh_.vertexCount);
        unmap();
        ready_ = true;
    } catch (...) {
        if (mapped_) {
            try { unmap(); }
            catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); }
        }
        throw;
    }
}

void WaterRenderer::draw(Shader shader, Matrix transform) const {
    if (!ready_ || mapped_) throw std::logic_error("water surface must be updated before drawing");
    const float white[4] = {1, 1, 1, 1};
    rlSetVertexAttributeDefault(RL_DEFAULT_SHADER_ATTRIB_LOCATION_COLOR, white, SHADER_ATTRIB_VEC4, 4);
    MaterialMap maps[MAX_MATERIAL_MAPS]{};
    maps[MATERIAL_MAP_DIFFUSE].color = WHITE;
    maps[MATERIAL_MAP_DIFFUSE].texture.id = rlGetTextureIdDefault();
    Material material{};
    material.shader = shader;
    material.maps = maps;
    DrawMesh(mesh_, material, transform);
}

void WaterRenderer::release() noexcept {
    if (mapped_) {
        const auto error = cudaGraphicsUnmapResources(2, resources_, mapped_stream_);
        check_cleanup(error, "unmap water vertex buffers during cleanup");
        if (error == cudaSuccess) {
            mapped_ = false;
#ifdef WATER_INTEROP_VALIDATION
            interop_mappings -= 2;
#endif
        }
        check_cleanup(cudaStreamSynchronize(mapped_stream_), "synchronize water cleanup");
    }
    for (auto& resource : resources_) {
        if (resource) {
            const auto error = cudaGraphicsUnregisterResource(resource);
            check_cleanup(error, "unregister water vertex buffer");
#ifdef WATER_INTEROP_VALIDATION
            if (error == cudaSuccess) --interop_registrations;
#endif
            resource = nullptr;
        }
    }
    rlDisableVertexArray();
    rlDisableVertexBuffer();
    UnloadMesh(mesh_);
    mesh_ = {};
}

} // namespace water
