#include "boat_renderer.hpp"
#include <external/glad.h>
#include <cuda_gl_interop.h>
#include <rlgl.h>
#include <config.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <stdexcept>
#include <string>
#ifdef WATER_INTEROP_VALIDATION
#include "boat_render_validation.hpp"
#endif
namespace water {
#ifdef WATER_INTEROP_VALIDATION
BoatRenderFault boat_render_fault = BoatRenderFault::none;
int boat_registrations=0, boat_mappings=0, boat_registration_calls=0;
BoatRenderHandles boat_render_handles{};
#define BOAT_FAULT(stage) if (boat_render_fault == BoatRenderFault::stage) throw std::runtime_error("injected boat " #stage " failure")
#else
#define BOAT_FAULT(stage) ((void)0)
#endif
namespace {
void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
}
void cleanup(cudaError_t error, const char* operation) noexcept {
    if (error != cudaSuccess) std::fprintf(stderr,"%s: %s\n",operation,cudaGetErrorString(error));
}
std::string shader_source(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("could not read boat shader: "+path.string());
    return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
}
// DrawMesh changes these bindings even when its model transform is identity.
struct DrawState {
    GLint active=0,
          pose=0,
          program=0,
          vao=0,
          array=0,
          textures[MAX_MATERIAL_MAPS]{};
    GLboolean depth_mask=GL_TRUE;
    bool depth=false,
         blend=false,
         cull=false;
    DrawState() {
        glGetIntegerv(GL_ACTIVE_TEXTURE,&active);
        glActiveTexture(GL_TEXTURE15);
        glGetIntegerv(GL_TEXTURE_BINDING_BUFFER,&pose);
        for (int i=0;i<MAX_MATERIAL_MAPS;++i) {
            glActiveTexture(GL_TEXTURE0+i);
            glGetIntegerv(GL_TEXTURE_BINDING_2D,&textures[i]);
        }
        glGetIntegerv(GL_CURRENT_PROGRAM,&program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING,&vao);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&array);
        glGetBooleanv(GL_DEPTH_WRITEMASK,&depth_mask);
        depth=glIsEnabled(GL_DEPTH_TEST);
        blend=glIsEnabled(GL_BLEND);
        cull=glIsEnabled(GL_CULL_FACE);
        rlDrawRenderBatchActive();
    }
    ~DrawState() {
        rlDrawRenderBatchActive();
        glUseProgram(program);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER,array);
        for (int i=0;i<MAX_MATERIAL_MAPS;++i) {
            glActiveTexture(GL_TEXTURE0+i);
            glBindTexture(GL_TEXTURE_2D,textures[i]);
        }
        glActiveTexture(GL_TEXTURE15);
        glBindTexture(GL_TEXTURE_BUFFER,pose);
        glActiveTexture(active);
        glDepthMask(depth_mask);
        if (depth) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
        if (blend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
        if (cull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    }
};
}
BoatRenderer::BoatRenderer(const Grid& grid) : grid_(grid) {
    if (!IsWindowReady()) throw std::runtime_error("boat renderer requires a current OpenGL context");
    GLint binding=0,active=0,texture=0;
    rlDrawRenderBatchActive();
    glGetIntegerv(GL_TEXTURE_BUFFER,&binding);
    glGetIntegerv(GL_ACTIVE_TEXTURE,&active);
    glActiveTexture(GL_TEXTURE15);
    glGetIntegerv(GL_TEXTURE_BINDING_BUFFER,&texture);
    try {
        const auto assets=std::filesystem::path(GetApplicationDirectory())/"assets";
        BOAT_FAULT(shader);
        const auto vs=shader_source(assets/"boat.vs"), fs=shader_source(assets/"boat.fs");
        shader_=LoadShaderFromMemory(vs.c_str(),fs.c_str());
        if (!shader_.id || shader_.id == rlGetShaderIdDefault()) throw std::runtime_error("could not compile boat shader");
#ifdef WATER_INTEROP_VALIDATION
        boat_render_handles.shader=shader_.id;
#endif
        BOAT_FAULT(model);
        const auto path=assets/"69ChrisCraftBoat"/"ChrisCraft.glb";
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("missing boat model: "+path.string());
        model_=LoadModel(path.string().c_str());
        if (model_.meshCount <= 0) throw std::runtime_error("boat model contains no meshes");
        for (int i=0;i<model_.meshCount;++i)
            if (model_.meshMaterial[i]<0 || model_.meshMaterial[i]>=model_.materialCount)
                throw std::runtime_error("boat model has invalid material mapping");
        BOAT_FAULT(buffer);
        glGenBuffers(1,&buffer_);
        glBindBuffer(GL_TEXTURE_BUFFER,buffer_);
        glBufferData(GL_TEXTURE_BUFFER,64,nullptr,GL_DYNAMIC_DRAW);
#ifdef WATER_INTEROP_VALIDATION
        boat_render_handles.buffer=buffer_;
#endif
        BOAT_FAULT(texture);
        glGenTextures(1,&texture_);
        glActiveTexture(GL_TEXTURE15);
        glBindTexture(GL_TEXTURE_BUFFER,texture_);
        glTexBuffer(GL_TEXTURE_BUFFER,GL_RGBA32F,buffer_);
#ifdef WATER_INTEROP_VALIDATION
        boat_render_handles.texture=texture_;
#endif
        if (!buffer_ || !texture_ || glGetError()!=GL_NO_ERROR) throw std::runtime_error("could not allocate boat pose texture buffer");
        BOAT_FAULT(registration);
        check(cudaGraphicsGLRegisterBuffer(&resource_,buffer_,cudaGraphicsRegisterFlagsWriteDiscard),"register boat pose buffer");
#ifdef WATER_INTEROP_VALIDATION
        ++boat_registrations;
        ++boat_registration_calls;
#endif
        BOAT_FAULT(after_registration);
    } catch (...) {
        release();
        glBindBuffer(GL_TEXTURE_BUFFER,binding);
        glActiveTexture(GL_TEXTURE15);
        glBindTexture(GL_TEXTURE_BUFFER,texture);
        glActiveTexture(active);
        throw;
    }
    glBindBuffer(GL_TEXTURE_BUFFER,binding);
    glActiveTexture(GL_TEXTURE15);
    glBindTexture(GL_TEXTURE_BUFFER,texture);
    glActiveTexture(active);
}
BoatRenderer::~BoatRenderer() { release(); }
void BoatRenderer::unmap() {
    check(cudaGraphicsUnmapResources(1,&resource_,stream_),"unmap boat pose buffer");
    mapped_=false;
#ifdef WATER_INTEROP_VALIDATION
    --boat_mappings;
#endif
    check(cudaStreamSynchronize(stream_),"complete boat ownership transfer");
    stream_=nullptr;
}
void BoatRenderer::update(Simulation& simulation) {
    const auto& grid=simulation.config().grid;
    if (grid.nx!=grid_.nx || grid.ny!=grid_.ny || grid.lx!=grid_.lx || grid.lz!=grid_.lz)
        throw std::invalid_argument("boat renderer and simulation grids must match");
    if (mapped_) throw std::logic_error("boat pose still owned by CUDA");
    ready_=false;
    stream_=simulation.stream();
    BOAT_FAULT(map);
    check(cudaGraphicsMapResources(1,&resource_,stream_),"map boat pose buffer");
    mapped_=true;
#ifdef WATER_INTEROP_VALIDATION
    ++boat_mappings;
#endif
    try {
        BOAT_FAULT(after_mapping);
        float* pointer=nullptr;
        std::size_t bytes=0;
        check(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pointer),&bytes,resource_),"get boat pose pointer");
        if (bytes!=64) throw std::runtime_error("boat pose buffer must contain exactly 16 floats");
        simulation.write_boat_transform(pointer,16);
        BOAT_FAULT(unmap);
        unmap();
        ready_=true;
    } catch (...) {
        if (mapped_) {
            try { unmap(); }
            catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); }
        }
        throw;
    }
}
void BoatRenderer::draw(const Camera3D& camera,bool submerged) const {
    if (!ready_ || mapped_) throw std::logic_error("boat pose must be updated before drawing");
    DrawState state;
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glActiveTexture(GL_TEXTURE15);
    glBindTexture(GL_TEXTURE_BUFFER,texture_);
    const int slot=15,clip=submerged ? 1 : 0;
    SetShaderValue(shader_,GetShaderLocation(shader_,"boatPose"),&slot,SHADER_UNIFORM_INT);
    SetShaderValue(shader_,GetShaderLocation(shader_,"submergedOnly"),&clip,SHADER_UNIFORM_INT);
    SetShaderValue(shader_,GetShaderLocation(shader_,"eyePosition"),&camera.position,SHADER_UNIFORM_VEC3);
    constexpr Matrix identity{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    // Opaque meshes establish depth before the small transparent windshield.
    for (int transparent=0;transparent<2;++transparent) {
        glDepthMask(transparent ? GL_FALSE : GL_TRUE);
        if (transparent) glEnable(GL_BLEND); else glDisable(GL_BLEND);
        for (int i=0;i<model_.meshCount;++i) {
            Material material=model_.materials[model_.meshMaterial[i]];
            if ((material.maps[MATERIAL_MAP_DIFFUSE].color.a<255) != bool(transparent)) continue;
            material.shader=shader_;
            DrawMesh(model_.meshes[i],material,identity);
        }
    }
}
unsigned BoatRenderer::pose_texture() const { return texture_; }
void BoatRenderer::release() noexcept {
    ready_=false;
    if (mapped_) {
        const auto error=cudaGraphicsUnmapResources(1,&resource_,stream_);
        cleanup(error,"unmap boat pose during cleanup");
        if (error!=cudaSuccess) return;
        mapped_=false;
#ifdef WATER_INTEROP_VALIDATION
        --boat_mappings;
#endif
        cleanup(cudaStreamSynchronize(stream_),"complete boat cleanup transfer");
    }
    if (resource_) {
        const auto error=cudaGraphicsUnregisterResource(resource_);
        cleanup(error,"unregister boat pose buffer");
        if (error!=cudaSuccess) return;
        resource_=nullptr;
#ifdef WATER_INTEROP_VALIDATION
        --boat_registrations;
#endif
    }
#ifdef WATER_INTEROP_VALIDATION
    if (buffer_) boat_render_handles.deleted_after_unregister=true;
#endif
    if (texture_) glDeleteTextures(1,&texture_);
    if (buffer_) glDeleteBuffers(1,&buffer_);
    if (model_.meshCount) UnloadModel(model_);
    if (shader_.id && shader_.id!=rlGetShaderIdDefault()) UnloadShader(shader_);
    texture_=0;
    buffer_=0;
    model_={};
    shader_={};
    stream_=nullptr;
}
}
