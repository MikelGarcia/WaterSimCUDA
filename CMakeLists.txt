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

add_library(water_core src/config.cpp)
target_include_directories(water_core PUBLIC src)
target_link_libraries(water_core PUBLIC CUDA::cudart)

include(CTest)
if(BUILD_TESTING)
    add_executable(water_tests tests/test_main.cpp tests/config_tests.cpp)
    target_link_libraries(water_tests PRIVATE water_core)
    add_test(NAME config COMMAND water_tests config)
endif()
