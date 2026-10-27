#pragma once

#include "water_renderer.hpp"

namespace water {

enum class InteropFault { none, after_first_registration, after_mapping };
extern InteropFault interop_fault;
extern int interop_registrations;
extern int interop_mappings;

struct RendererValidation {
    static const Mesh& mesh(const WaterRenderer& renderer) { return renderer.mesh_; }
};

namespace detail {
int choose_gl_cuda_device(const int* devices, unsigned count);
}

} // namespace water
