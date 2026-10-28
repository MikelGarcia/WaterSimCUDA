#include "test_support.hpp"
#include "boat_render_validation.hpp"
#include "water_renderer.hpp"
#include "validation.hpp"
#include "appearance_validation.hpp"
#include "timeline.hpp"
#include <external/glad.h>
#include <array>
#include <filesystem>
#include <rlgl.h>
namespace {
struct Context {
    Context() { SetTraceLogLevel(LOG_WARNING); SetConfigFlags(FLAG_WINDOW_HIDDEN); InitWindow(640,480,"Boat validation"); water::select_gl_cuda_device(); }
    ~Context() { CloseWindow(); }
};
}
TEST_CASE("boat_interop", gpu_pose_is_four_rgba32f_texels) {
    Context context;
    water::Config config;
    config.grid = {8,8,4,4}; config.impact_radius = 1;
    water::Simulation simulation(config);
    water::BoatRenderer boat(config.grid);
    CHECK(boat.pose_texture() != 0);
    const water::BoatState poses[]{{2.3f,1.7f,0,0,0,0},{1.4f,2.4f,1.570796327f,0,0,0},{2,2,-0.73f,0,0,0}};
    for (auto pose : poses) {
        water::upload_boat(simulation,pose);
        boat.update(simulation);
        std::array<float,16> matrix{};
        glBindBuffer(GL_TEXTURE_BUFFER,water::BoatRenderValidation::buffer(boat));
        GLint bytes=0; glGetBufferParameteriv(GL_TEXTURE_BUFFER,GL_BUFFER_SIZE,&bytes);
        CHECK(bytes == 64);
        glGetBufferSubData(GL_TEXTURE_BUFFER,0,64,matrix.data());
        CHECK_NEAR(matrix[0],std::cos(pose.yaw),1e-6f);
        CHECK_NEAR(matrix[2],std::sin(pose.yaw),1e-6f);
        CHECK_NEAR(matrix[8],-std::sin(pose.yaw),1e-6f);
        CHECK_NEAR(matrix[10],std::cos(pose.yaw),1e-6f);
        CHECK_NEAR(matrix[12],pose.x-2,1e-6f);
        CHECK_NEAR(matrix[14],pose.z-2,1e-6f);
        CHECK(matrix[5] == 1 && matrix[15] == 1 && matrix[13] == 0);
        glBindTexture(GL_TEXTURE_BUFFER,boat.pose_texture());
        GLint format=0,width=0;
        glGetTexLevelParameteriv(GL_TEXTURE_BUFFER,0,GL_TEXTURE_INTERNAL_FORMAT,&format);
        glGetTexLevelParameteriv(GL_TEXTURE_BUFFER,0,GL_TEXTURE_WIDTH,&width);
        CHECK(format == GL_RGBA32F && width == 4);
    }
    CHECK(water::BoatRenderValidation::model(boat).meshCount == 35);
    glBindBuffer(GL_TEXTURE_BUFFER,0); glBindTexture(GL_TEXTURE_BUFFER,0);
    CHECK(glGetError() == GL_NO_ERROR);
}
TEST_CASE("boat_interop", ownership_faults_cleanup_and_recover) {
    Context context;
    water::Config config; config.grid={8,8,4,4}; config.impact_radius=1;
    water::Simulation simulation(config);
    for (auto fault : {water::BoatRenderFault::shader, water::BoatRenderFault::model,
            water::BoatRenderFault::buffer,water::BoatRenderFault::texture,
            water::BoatRenderFault::registration,water::BoatRenderFault::after_registration}) {
        water::boat_render_handles={}; water::boat_render_fault=fault;
        CHECK_THROWS(water::BoatRenderer(config.grid),std::runtime_error);
        CHECK(water::boat_registrations==0 && water::boat_mappings==0);
        CHECK(!glIsBuffer(water::boat_render_handles.buffer));
        CHECK(!glIsTexture(water::boat_render_handles.texture));
        CHECK(!glIsProgram(water::boat_render_handles.shader));
    }
    water::boat_render_fault=water::BoatRenderFault::none;
    water::BoatRenderer boat(config.grid);
    const Camera3D camera{{.3f,.3f,.3f},{0,0,0},{0,1,0},45,CAMERA_PERSPECTIVE};
    CHECK_THROWS(boat.draw(camera,false),std::logic_error);
    const int registrations=water::boat_registration_calls;
    for (auto fault : {water::BoatRenderFault::map,water::BoatRenderFault::after_mapping,water::BoatRenderFault::unmap}) {
        water::boat_render_fault=fault;
        CHECK_THROWS(boat.update(simulation),std::runtime_error);
        CHECK(water::boat_mappings==0 && water::boat_registrations==1);
        CHECK_THROWS(boat.draw(camera,false),std::logic_error);
        water::boat_render_fault=water::BoatRenderFault::none;
        boat.update(simulation);
    }
    CHECK(water::boat_registration_calls==registrations);
    const unsigned buffer=water::BoatRenderValidation::buffer(boat),texture=boat.pose_texture();
    water::BoatRenderValidation::release(boat); water::BoatRenderValidation::release(boat);
    CHECK(water::boat_registrations==0 && water::boat_mappings==0);
    CHECK(water::boat_render_handles.deleted_after_unregister);
    CHECK(!glIsBuffer(buffer) && !glIsTexture(texture));
    CHECK(glGetError()==GL_NO_ERROR);
}
TEST_CASE("boat_interop", draw_preserves_nondefault_caller_state) {
    Context context;
    water::Config config; config.grid={8,8,4,4}; config.impact_radius=1;
    water::Simulation simulation(config); water::BoatRenderer boat(config.grid); boat.update(simulation);
    GLuint buffers[2]{},textures[2]{},vao=0;
    glGenBuffers(2,buffers); glGenTextures(2,textures); glGenVertexArrays(1,&vao);
    glBindBuffer(GL_TEXTURE_BUFFER,buffers[0]); glBufferData(GL_TEXTURE_BUFFER,64,nullptr,GL_STATIC_DRAW);
    glActiveTexture(GL_TEXTURE15); glBindTexture(GL_TEXTURE_BUFFER,textures[0]); glTexBuffer(GL_TEXTURE_BUFFER,GL_RGBA32F,buffers[0]);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,textures[1]);
    Camera3D camera{{.3f,.3f,.3f},{0,0,0},{0,1,0},45,CAMERA_PERSPECTIVE};
    BeginDrawing(); BeginMode3D(camera);
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,buffers[1]);
    glActiveTexture(GL_TEXTURE7); glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE); glDisable(GL_BLEND); glEnable(GL_CULL_FACE);
    boat.draw(camera,false);
    GLint value=0; glGetIntegerv(GL_ACTIVE_TEXTURE,&value); CHECK(value==GL_TEXTURE7);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING,&value); CHECK(value==vao);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&value); CHECK(value==buffers[1]);
    CHECK(!glIsEnabled(GL_DEPTH_TEST) && !glIsEnabled(GL_BLEND) && glIsEnabled(GL_CULL_FACE));
    GLboolean depth=GL_TRUE; glGetBooleanv(GL_DEPTH_WRITEMASK,&depth); CHECK(!depth);
    glActiveTexture(GL_TEXTURE15); glGetIntegerv(GL_TEXTURE_BINDING_BUFFER,&value); CHECK(value==textures[0]);
    glActiveTexture(GL_TEXTURE0); glGetIntegerv(GL_TEXTURE_BINDING_2D,&value); CHECK(value==textures[1]);
    glBindTexture(GL_TEXTURE_2D,0); glActiveTexture(GL_TEXTURE15); glBindTexture(GL_TEXTURE_BUFFER,0); glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(0); glBindBuffer(GL_ARRAY_BUFFER,0); glBindBuffer(GL_TEXTURE_BUFFER,0); glDepthMask(GL_TRUE);
    EndMode3D(); EndDrawing();
    glDeleteTextures(2,textures); glDeleteBuffers(2,buffers); glDeleteVertexArrays(1,&vao);
    CHECK(glGetError()==GL_NO_ERROR);
}
TEST_CASE("boat_interop", scene_capture_contains_submerged_hull) {
    Context context;
    water::Config config; config.grid={32,32,4,4}; config.impact_radius=.3f; config.depth=.25f; config.impact_strength=.003f;
    water::Simulation simulation(config); water::BoatRenderer boat(config.grid); boat.update(simulation);
    water::SceneRenderer scene(config); CHECK(scene.resize(640,480));
    const Camera3D camera{{0,.05f,.43f},{0,-.015f,0},{0,1,0},38,CAMERA_PERSPECTIVE};
    BeginDrawing(); water::SceneValidation::capture(scene,camera,boat);
    const auto target=water::SceneValidation::target(scene);
    glBindFramebuffer(GL_FRAMEBUFFER,target.id);
    std::vector<unsigned char> pixels(640*480*4); glReadPixels(0,0,640,480,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    int blue=0;
    for (int y=160;y<300;++y) for (int x=140;x<500;++x) {
        const auto p=4*(y*640+x);
        if (pixels[p+2]>pixels[p]*1.5f && pixels[p+2]>pixels[p+1]*1.15f) ++blue;
    }
    glBindFramebuffer(GL_FRAMEBUFFER,0); EndDrawing();
    CHECK(blue>300);
}


namespace {
float reference_depth(Vector3 p, const Camera3D& camera) {
    const Vector3 d{camera.target.x-camera.position.x,camera.target.y-camera.position.y,camera.target.z-camera.position.z};
    const float length=std::sqrt(d.x*d.x+d.y*d.y+d.z*d.z);
    const float distance=((p.x-camera.position.x)*d.x+(p.y-camera.position.y)*d.y+(p.z-camera.position.z)*d.z)/length;
    const double near=rlGetCullDistanceNear(), far=rlGetCullDistanceFar();
    return float(far/(far-near)-far*near/((far-near)*distance));
}
Vector3 world_point(Vector3 p,water::BoatState pose) {
    return {pose.x-.5f+std::cos(pose.yaw)*p.x-std::sin(pose.yaw)*p.z,p.y,
            pose.z-.5f+std::sin(pose.yaw)*p.x+std::cos(pose.yaw)*p.z};
}
struct Pixel { std::array<unsigned char,4> color{}; float depth=1; };
Pixel read_pixel(int x,int y) {
    Pixel p; glReadPixels(x,y,1,1,GL_RGBA,GL_UNSIGNED_BYTE,p.color.data());
    glReadPixels(x,y,1,1,GL_DEPTH_COMPONENT,GL_FLOAT,&p.depth); return p;
}
}
TEST_CASE("boat_interop", real_mesh_landmarks_occlude_water_and_clip_capture_at_all_mixes) {
    Context context;
    water::Config config; config.grid={64,64,1,1}; config.depth=.25f; config.impact_radius=.1f; config.impact_strength=.003f;
    water::Simulation simulation(config); water::WaterRenderer water(config.grid); water.update(simulation);
    water::BoatRenderer boat(config.grid); water::SceneRenderer scene(config); CHECK(scene.resize(640,480));
    const Camera3D views[]{{{0,.63f,0},{0,0,0},{0,0,-1},38,CAMERA_PERSPECTIVE},
                           {{.31f,.28f,.42f},{0,0,0},{0,1,0},38,CAMERA_PERSPECTIVE}};
    const water::BoatState poses[]{{.5f,.5f,0,0,0,0},{.53f,.48f,1.570796327f,0,0,0},{.47f,.52f,-.65f,0,0,0}};
    std::filesystem::create_directories("boat-render-captures");
    int frame=0;
    for (auto pose:poses) for (auto camera:views) {
        water::upload_boat(simulation,pose); boat.update(simulation);
        struct Landmark { int x,y,mesh; float expected; Pixel reference; };
        std::vector<Landmark> landmarks;
        BeginDrawing(); ClearBackground(BLANK); BeginMode3D(camera); boat.draw(camera,false); EndMode3D();
        const auto& model=water::BoatRenderValidation::model(boat);
        int blue=0,white=0;
        std::vector<unsigned char> pixels(640*480*4); glReadPixels(0,0,640,480,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
        for (int p=0;p<640*480;++p) {
            const auto r=pixels[4*p],g=pixels[4*p+1],b=pixels[4*p+2];
            blue+=b>r*1.5f && b>g*1.15f; white+=r>150 && g>150 && b>150;
        }
        CHECK(blue>300 && white>300);
        for (int m=0;m<model.meshCount;++m) {
            const auto& mesh=model.meshes[m];
            if (model.materials[model.meshMaterial[m]].maps[MATERIAL_MAP_DIFFUSE].color.a<255) continue;
            int accepted=0;
            for (int t=0;t<mesh.triangleCount && accepted<3;++t) {
                Vector3 center{};
                for (int v=0;v<3;++v) {
                    const int i=3*(mesh.indices ? mesh.indices[3*t+v] : 3*t+v);
                    center.x+=mesh.vertices[i]/3; center.y+=mesh.vertices[i+1]/3; center.z+=mesh.vertices[i+2]/3;
                }
                if (center.y<.006f) continue;
                const auto point=world_point(center,pose);
                const auto projected=GetWorldToScreenEx(point,camera,640,480);
                const int x=int(projected.x),y=479-int(projected.y);
                if (x<2 || y<2 || x>=638 || y>=478) continue;
                const auto pixel=read_pixel(x,y); const float expected=reference_depth(point,camera);
                if (pixel.color[3]==255 && std::abs(pixel.depth-expected)<.00012f) {
                    landmarks.push_back({x,y,m,expected,pixel}); ++accepted;
                }
            }
        }
        CHECK(landmarks.size()>15);
        EndDrawing();
        for (float mix : {0.f,.7f,1.f}) {
            const std::string path=TextFormat("boat-render-captures/scene-%02d-%.1f.png",frame,mix);
            std::filesystem::remove(path);
            BeginDrawing(); scene.draw(camera,water,boat,{},12,mix);
            int exact=0;
            for (const auto& landmark:landmarks) {
                const auto pixel=read_pixel(landmark.x,landmark.y);
                if (std::abs(pixel.depth-landmark.reference.depth)<.000002f && pixel.color==landmark.reference.color) ++exact;
            }
            CHECK(exact>15);
            Image image=LoadImageFromScreen();
            const bool captured=image.data && image.width==640 && image.height==480;
            const bool exported=captured && ExportImage(image,path.c_str());
            UnloadImage(image);
            EndDrawing();
            CHECK(captured);
            CHECK(exported);
            CHECK(std::filesystem::is_regular_file(path));
            CHECK(std::filesystem::file_size(path)>0);
            Image saved=LoadImage(path.c_str());
            const bool valid=saved.data && saved.width==640 && saved.height==480;
            UnloadImage(saved);
            CHECK(valid);
        }
        water::SceneValidation::capture(scene,camera,boat);
        const auto target=water::SceneValidation::target(scene);
        glBindFramebuffer(GL_FRAMEBUFFER,target.id);
        int clipped=0;
        for (const auto& landmark:landmarks) {
            const auto pixel=read_pixel(landmark.x,landmark.y);
            if (pixel.color[3]==0 && std::abs(pixel.depth-landmark.reference.depth)<.000002f) ++clipped;
        }
        glBindFramebuffer(GL_FRAMEBUFFER,0);
        CHECK(clipped>15);
        Image capture=LoadImageFromTexture(target.texture); ImageFlipVertical(&capture);
        CHECK(ExportImage(capture,TextFormat("boat-render-captures/submerged-%02d.png",frame))); UnloadImage(capture);
        ++frame;
    }
    CHECK(glGetError()==GL_NO_ERROR);
}


TEST_CASE("boat_interop", recessed_cockpit_floor_remains_visible_at_all_water_mixes) {
    Context context;
    water::Config config; config.grid={64,64,1,1}; config.depth=.25f; config.impact_radius=.1f; config.impact_strength=.003f;
    water::Simulation simulation(config); water::WaterRenderer water(config.grid); water.update(simulation);
    water::BoatRenderer boat(config.grid); water::SceneRenderer scene(config); CHECK(scene.resize(640,480));
    const Camera3D camera{{0,.63f,0},{0,0,0},{0,0,-1},38,CAMERA_PERSPECTIVE};
    for (auto pose : {water::BoatState{.5f,.5f,0,0,0,0},water::BoatState{.53f,.48f,1.570796327f,0,0,0},water::BoatState{.47f,.52f,-.65f,0,0,0}}) {
        water::upload_boat(simulation,pose); boat.update(simulation);
        const Vector3 point=world_point({-.025f,0,.027f},pose);
        const auto projected=GetWorldToScreenEx(point,camera,640,480);
        std::array<unsigned char,4> initial{};
        for (float mix : {0.f,.7f,1.f}) {
            BeginDrawing(); scene.draw(camera,water,boat,{},12,mix);
            const auto pixel=read_pixel(int(projected.x),479-int(projected.y));
            CHECK(pixel.color[2]>pixel.color[0]*2 && pixel.color[2]>pixel.color[1]*1.3f);
            CHECK(pixel.depth>reference_depth(point,camera)+.0001f);
            if (mix==0) initial=pixel.color; else CHECK(pixel.color==initial);
            EndDrawing();
        }
    }
}
TEST_CASE("boat_interop", water_mask_tracks_measured_edges_on_gpu_at_each_heading) {
    Context context;
    water::Config config; config.grid={64,64,1,1}; config.impact_radius=.1f;
    water::Simulation simulation(config); water::BoatRenderer boat(config.grid);
    auto source=water::SceneValidation::fragment_source("water.fs");
    source.insert(source.find('\n')+1,"#define main waterMain\n");
    source+="\n#undef main\nuniform vec2 probePoint;\nvoid main(){finalColor=vec4(insideBoat(probePoint)?1.0:0.0,0,0,1);}";
    Shader shader=LoadShaderFromMemory(nullptr,source.c_str()); CHECK(shader.id!=rlGetShaderIdDefault());
    const int slot=15;
    SetShaderValue(shader,GetShaderLocation(shader,"boatPose"),&slot,SHADER_UNIFORM_INT);
    SetShaderValueV(shader,GetShaderLocation(shader,"boatWaterline[0]"),water::boat_waterline,SHADER_UNIFORM_VEC2,16);
    SetShaderValue(shader,GetShaderLocation(shader,"boatWaterlineCount"),&water::boat_waterline_count,SHADER_UNIFORM_INT);
    for (float yaw : {0.f,1.570796327f,-.65f}) {
        water::BoatState pose{.53f,.48f,yaw,0,0,0}; water::upload_boat(simulation,pose); boat.update(simulation);
        for (int edge=0;edge<16;++edge) {
            const auto a=water::boat_waterline[edge],b=water::boat_waterline[(edge+1)%16];
            const float dx=b.x-a.x,dz=b.z-a.z,length=std::sqrt(dx*dx+dz*dz);
            for (int inside : {0,1}) {
                const float distance=inside ? .0005f : -.0005f;
                const auto point=world_point({(a.x+b.x)/2-distance*dz/length,0,(a.z+b.z)/2+distance*dx/length},pose);
                const Vector2 world{point.x,point.z};
                SetShaderValue(shader,GetShaderLocation(shader,"probePoint"),&world,SHADER_UNIFORM_VEC2);
                BeginDrawing(); glActiveTexture(GL_TEXTURE15); glBindTexture(GL_TEXTURE_BUFFER,boat.pose_texture()); glActiveTexture(GL_TEXTURE0);
                BeginShaderMode(shader); DrawRectangle(0,0,640,480,WHITE); EndShaderMode(); rlDrawRenderBatchActive();
                const auto pixel=read_pixel(320,240); CHECK(pixel.color[0]==(inside ? 255 : 0));
                glActiveTexture(GL_TEXTURE15); glBindTexture(GL_TEXTURE_BUFFER,0); glActiveTexture(GL_TEXTURE0); EndDrawing();
            }
        }
    }
    UnloadShader(shader); CHECK(glGetError()==GL_NO_ERROR);
}
TEST_CASE("boat_interop", paused_and_reset_pose_updates_are_current_without_reregistering) {
    Context context;
    water::Config config; config.grid={8,8,4,4}; config.impact_radius=1;
    water::Simulation simulation(config); water::BoatRenderer boat(config.grid);
    water::upload_boat(simulation,{2.3f,1.8f,.7f,.1f,.2f,.3f});
    auto read=[&]() {
        boat.update(simulation); std::array<float,16> matrix{};
        glBindBuffer(GL_TEXTURE_BUFFER,water::BoatRenderValidation::buffer(boat));
        glGetBufferSubData(GL_TEXTURE_BUFFER,0,64,matrix.data()); glBindBuffer(GL_TEXTURE_BUFFER,0); return matrix;
    };
    const int registrations=water::boat_registration_calls;
    const auto first=read(); CHECK(read()==first);
    simulation.reset(); const auto reset=read();
    CHECK_NEAR(reset[0],1,1e-6f); CHECK_NEAR(reset[12],0,1e-6f); CHECK_NEAR(reset[14],0,1e-6f);
    CHECK(reset!=first && read()==reset);
    simulation.set_depth(1); CHECK(read()==reset);
    CHECK(water::boat_registration_calls==registrations);
}
