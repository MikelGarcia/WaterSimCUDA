cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
project(WaterSimCUDA LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

option(WATER_BUILD_DEMO "Build desktop targets and acquire raylib" ON)
if(WATER_BUILD_DEMO)
    include(FetchContent)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Build raylib examples" FORCE)
    set(PLATFORM Desktop CACHE STRING "raylib platform" FORCE)
    set(OPENGL_VERSION 3.3 CACHE STRING "raylib OpenGL version" FORCE)
    FetchContent_Declare(raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG c1ab645ca298a2801097931d1079b10ff7eb9df8
    )
    FetchContent_MakeAvailable(raylib)
endif()

add_library(water_core src/config.cpp src/simulation.cu)
target_include_directories(water_core PUBLIC src)
target_link_libraries(water_core PUBLIC CUDA::cudart)

if(WATER_BUILD_DEMO)
    add_library(water_timeline src/timeline.cpp)
    target_link_libraries(water_timeline PUBLIC water_core raylib)
    add_library(water_renderer src/water_renderer.cpp)
    target_link_libraries(water_renderer PUBLIC water_core raylib)
    add_library(water_boat_renderer src/boat_renderer.cpp)
    target_link_libraries(water_boat_renderer PUBLIC water_renderer)
    add_library(water_scene_renderer src/scene_renderer.cpp)
    target_link_libraries(water_scene_renderer PUBLIC water_boat_renderer water_timeline)
    add_library(water_controller src/app_controller.cpp)
    target_link_libraries(water_controller PUBLIC water_timeline)
    add_executable(water_demo src/main.cpp)
    target_link_libraries(water_demo PRIVATE water_controller water_scene_renderer)
    set(WATER_ASSET_DIR "${CMAKE_CURRENT_BINARY_DIR}/assets")
    if(CMAKE_CONFIGURATION_TYPES)
        set(WATER_ASSET_DIR "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/assets")
    endif()
    add_custom_target(water_assets
        COMMAND ${CMAKE_COMMAND} -E make_directory "${WATER_ASSET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/boat.vs"
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/boat.fs"
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/water.vs"
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/water.fs"
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/basin.vs"
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/basin.fs"
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/grass.fs"
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/grass.png"
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/drop.fs"
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/sky.fs"
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/sky.glsl"
                "${WATER_ASSET_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${WATER_ASSET_DIR}/69ChrisCraftBoat"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/69ChrisCraftBoat/ChrisCraft.glb"
                "${CMAKE_CURRENT_SOURCE_DIR}/assets/69ChrisCraftBoat/BLENDSWAP_LICENSE.txt"
                "${WATER_ASSET_DIR}/69ChrisCraftBoat"
        VERBATIM)
    add_dependencies(water_demo water_assets)
endif()

include(CTest)
if(BUILD_TESTING)
    add_library(water_validation src/validation.cu)
    target_link_libraries(water_validation PUBLIC water_core)
    add_executable(water_tests tests/test_main.cpp tests/config_tests.cpp tests/numerics_tests.cu tests/impact_tests.cu tests/timing_tests.cpp tests/boat_tests.cu tests/wake_tests.cu)
    target_link_libraries(water_tests PRIVATE water_validation)
    if(WATER_BUILD_DEMO)
        target_sources(water_tests PRIVATE tests/timeline_tests.cpp tests/controller_tests.cu tests/soak_tests.cu)
        target_link_libraries(water_tests PRIVATE water_controller)
        add_test(NAME timeline COMMAND water_tests timeline)
        add_test(NAME controller COMMAND water_tests controller)
        add_test(NAME soak COMMAND water_tests soak)
        set_tests_properties(soak PROPERTIES TIMEOUT 180 LABELS soak)
        add_test(NAME demo_help COMMAND water_demo --help)
        set_tests_properties(demo_help PROPERTIES PASS_REGULAR_EXPRESSION "--rain-rate")
        add_test(NAME demo_smoke COMMAND ${CMAKE_COMMAND}
            "-DDEMO=$<TARGET_FILE:water_demo>"
            "-DSMOKE_DIR=${CMAKE_CURRENT_BINARY_DIR}/demo-smoke"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/demo_smoke.cmake")
        set_tests_properties(demo_smoke PROPERTIES TIMEOUT 30 RUN_SERIAL TRUE)
        add_executable(water_interop_tests tests/test_main.cpp tests/interop_tests.cpp)
        add_library(water_renderer_validation src/water_renderer.cpp)
        target_compile_definitions(water_renderer_validation PRIVATE WATER_INTEROP_VALIDATION)
        target_include_directories(water_renderer_validation PRIVATE tests)
        target_link_libraries(water_renderer_validation PUBLIC water_core raylib)
        target_link_libraries(water_interop_tests PRIVATE water_validation water_renderer_validation)
        add_library(water_boat_renderer_validation src/boat_renderer.cpp)
        target_compile_definitions(water_boat_renderer_validation PRIVATE WATER_INTEROP_VALIDATION)
        target_include_directories(water_boat_renderer_validation PRIVATE tests)
        target_link_libraries(water_boat_renderer_validation PUBLIC water_renderer_validation)
        add_executable(water_boat_interop_tests tests/test_main.cpp tests/boat_interop_tests.cpp)
        target_compile_definitions(water_boat_interop_tests PRIVATE WATER_APPEARANCE_VALIDATION)
        target_link_libraries(water_boat_interop_tests PRIVATE water_boat_renderer_validation water_scene_renderer_validation water_validation)
        add_dependencies(water_boat_interop_tests water_assets)
        add_test(NAME boat_interop COMMAND water_boat_interop_tests boat_interop)
        add_test(NAME interop COMMAND water_interop_tests interop)
        set_tests_properties(interop PROPERTIES TIMEOUT 60)
        add_library(water_scene_renderer_validation src/scene_renderer.cpp)
        target_compile_definitions(water_scene_renderer_validation PRIVATE WATER_APPEARANCE_VALIDATION)
        target_include_directories(water_scene_renderer_validation PRIVATE tests)
        target_link_libraries(water_scene_renderer_validation PUBLIC water_boat_renderer_validation water_validation water_timeline)
        add_executable(water_appearance_tests tests/test_main.cpp tests/appearance_tests.cpp)
        target_compile_definitions(water_appearance_tests PRIVATE WATER_APPEARANCE_VALIDATION)
        target_link_libraries(water_appearance_tests PRIVATE water_scene_renderer_validation water_controller)
        add_dependencies(water_appearance_tests water_assets)
        add_executable(water_boat_asset_tests tests/test_main.cpp tests/boat_asset_tests.cpp)
        target_link_libraries(water_boat_asset_tests PRIVATE raylib)
        target_include_directories(water_boat_asset_tests PRIVATE src)
        add_dependencies(water_boat_asset_tests water_assets)
        add_test(NAME boat_asset COMMAND water_boat_asset_tests boat_asset)
        add_test(NAME appearance COMMAND water_appearance_tests appearance)
        set_tests_properties(appearance PROPERTIES TIMEOUT 60)
    endif()
    add_test(NAME config COMMAND water_tests config)
    add_test(NAME boat COMMAND water_tests boat)
    add_test(NAME wake COMMAND water_tests wake)
    add_test(NAME numerics COMMAND water_tests numerics)
    add_test(NAME impacts COMMAND water_tests impacts)
    add_test(NAME timing COMMAND water_tests timing)
endif()
