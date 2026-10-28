#pragma once
#include "simulation.hpp"
#include <raylib.h>
struct cudaGraphicsResource;
namespace water {
class BoatRenderer {
public:
    explicit BoatRenderer(const Grid&);
    ~BoatRenderer();
    BoatRenderer(const BoatRenderer&) = delete;
    BoatRenderer& operator=(const BoatRenderer&) = delete;
    void update(Simulation&);
    void draw(const Camera3D&, bool submerged) const;
    unsigned pose_texture() const;
private:
    friend struct BoatRenderValidation;
    void release() noexcept;
    void unmap();
    Grid grid_;
    Model model_{};
    Shader shader_{};
    unsigned buffer_ = 0, texture_ = 0;
    cudaGraphicsResource* resource_ = nullptr;
    cudaStream_t stream_ = nullptr;
    bool mapped_ = false, ready_ = false;
};
}
