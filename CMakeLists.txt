cmake_minimum_required(VERSION 3.21)
project(FlashLineage3D
    VERSION 0.1.0
    DESCRIPTION "Forward renderer for a Tanki-style tank game (Vulkan 1.1 + OpenGL 3.3 core)"
    LANGUAGES C CXX)

# The render engine alone: the engine libraries, the offline tools (fl_shaderc, fl_assetc), the shader packages and the
# quality tiers. Add your game's executable at the end of this file (see README.md, "Using the engine from a game").

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(FL_DEV "Developer features: validation, debug UI panels, shader hot reload" ON)
option(FL_TRACY "Link the Tracy profiler client" ON)
option(FL_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# Runtime data (shader packages, quality tiers, cooked assets) lives next to the executables.
set(FL_DATA_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(FlCompiler)
include(FlShaders)

find_package(SDL3 CONFIG REQUIRED)
find_package(VulkanHeaders CONFIG REQUIRED)
find_package(volk CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glslang CONFIG REQUIRED)
find_package(spirv_cross_core CONFIG REQUIRED)
find_package(spirv_cross_glsl CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(yyjson CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)
find_package(Ktx CONFIG REQUIRED)
find_package(mikktspace CONFIG REQUIRED)
find_path(CGLTF_INCLUDE_DIR "cgltf.h" REQUIRED)
if(FL_TRACY)
    find_package(Tracy CONFIG REQUIRED)
endif()

# Dev builds bundle the Khronos validation layer built by vcpkg next to the executables, so validation works
# without a Vulkan SDK install. An installed SDK layer is used instead when present.
set(FL_VK_LAYER_SOURCE "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
if(FL_DEV AND WIN32 AND EXISTS "${FL_VK_LAYER_SOURCE}/VkLayer_khronos_validation.json")
    file(COPY "${FL_VK_LAYER_SOURCE}/VkLayer_khronos_validation.json" "${FL_VK_LAYER_SOURCE}/VkLayer_khronos_validation.dll"
         DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/vulkan_layers")
endif()

add_subdirectory(engine/core)
add_subdirectory(engine/shaderformat)
add_subdirectory(engine/platform)
add_subdirectory(engine/rhi)          # also adds the Vulkan and GL backends and fl_rhi_factory
add_subdirectory(engine/assets)
add_subdirectory(engine/anim)
add_subdirectory(engine/renderer)
add_subdirectory(engine/world)
add_subdirectory(tools/shaderc)
add_subdirectory(tools/assetc)

# Engine data: the quality tiers (assets/source/settings/tiers.json), validated and copied to data/settings by fl_assetc.
# Add your game's content to assets/source/cook.json (textures, models, maps) and it is cooked into data/ the same way.
add_custom_target(fl_cook ALL
    COMMAND fl_assetc cook
            --manifest "${CMAKE_SOURCE_DIR}/assets/source/cook.json"
            --source "${CMAKE_SOURCE_DIR}/assets/source"
            --external "${CMAKE_SOURCE_DIR}/assets/external"
            --out "${FL_DATA_DIR}"
    DEPENDS fl_assetc
    COMMENT "Cooking assets"
    VERBATIM)

# Shader packages are declared from this top-level directory so that their depfiles use paths
# relative to CMAKE_BINARY_DIR, which is what Ninja expects.
include(shaders/Shaders.cmake)

# Your game goes here, for example:
#   add_subdirectory(game)
# with game/CMakeLists.txt:
#   add_executable(my_game main.cpp)
#   target_link_libraries(my_game PRIVATE fl_world fl_renderer fl_rhi_factory fl_platform)
#   add_dependencies(my_game fl_shaders fl_cook)
#   fl_target_defaults(my_game)
