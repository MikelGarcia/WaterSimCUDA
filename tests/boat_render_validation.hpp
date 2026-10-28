#pragma once
#include "boat_renderer.hpp"
namespace water {
enum class BoatRenderFault { none, shader, model, buffer, texture, registration, after_registration, map, after_mapping, unmap };
extern BoatRenderFault boat_render_fault;
extern int boat_registrations, boat_mappings, boat_registration_calls;
struct BoatRenderHandles { unsigned buffer=0, texture=0, shader=0; bool deleted_after_unregister=false; };
extern BoatRenderHandles boat_render_handles;
struct BoatRenderValidation {
    static unsigned buffer(const BoatRenderer& boat) { return boat.buffer_; }
    static const Model& model(const BoatRenderer& boat) { return boat.model_; }
    static void release(BoatRenderer& boat) { boat.release(); }
};
}
