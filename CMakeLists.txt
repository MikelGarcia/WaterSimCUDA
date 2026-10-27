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
    add_library(water_controller src/app_controller.cpp)
    target_link_libraries(water_controller PUBLIC water_timeline)
    add_executable(water_demo src/main.cpp)
    target_link_libraries(water_demo PRIVATE water_controller water_renderer)
    add_custom_command(TARGET water_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/assets" "$<TARGET_FILE_DIR:water_demo>/assets")
endif()

include(CTest)
if(BUILD_TESTING)
    add_library(water_validation src/validation.cu)
    target_link_libraries(water_validation PUBLIC water_core)
    add_executable(water_tests tests/test_main.cpp tests/config_tests.cpp tests/numerics_tests.cu tests/impact_tests.cu)
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
        add_test(NAME interop COMMAND water_interop_tests interop)
        set_tests_properties(interop PROPERTIES TIMEOUT 60)
    endif()
    add_test(NAME config COMMAND water_tests config)
    add_test(NAME numerics COMMAND water_tests numerics)
    add_test(NAME impacts COMMAND water_tests impacts)
endif()
