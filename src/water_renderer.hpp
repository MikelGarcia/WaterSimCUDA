#pragma once

#include "simulation.hpp"
#include <raylib.h>

struct cudaGraphicsResource;

namespace water {

// Call with the current OpenGL context before allocating a Simulation.
int select_gl_cuda_device();

class WaterRenderer {
public:
    explicit WaterRenderer(const Grid& grid);
    ~WaterRenderer();
    WaterRenderer(const WaterRenderer&) = delete;
    WaterRenderer& operator=(const WaterRenderer&) = delete;

    void update(Simulation& simulation);
    void draw(Shader shader, Matrix transform) const;

private:
    friend struct RendererValidation;
    void unmap();
    void release() noexcept;

    const Grid grid_;
    Mesh mesh_{};
    cudaGraphicsResource* resources_[2]{};
    cudaStream_t mapped_stream_ = nullptr;
    bool mapped_ = false;
    bool ready_ = false;
};

} // namespace water
