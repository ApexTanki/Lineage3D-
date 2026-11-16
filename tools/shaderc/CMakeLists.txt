# fl_shadercore: GLSL 450 → SPIR-V (glslang) → GLSL 330 (SPIRV-Cross) + reflection + validation.
# A core-level library: the offline tool uses it always, dev renderer builds will use it for hot reload (M9).
add_library(fl_shadercore STATIC
    ShaderCompiler.cpp
    Manifest.cpp
    PackageWriter.cpp)
target_include_directories(fl_shadercore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(fl_shadercore
    PUBLIC fl_core fl_shaderformat
    PRIVATE glslang::glslang glslang::glslang-default-resource-limits glslang::SPIRV
            spirv-cross-core spirv-cross-glsl yyjson::yyjson)
fl_target_defaults(fl_shadercore)

add_executable(fl_shaderc main.cpp)
target_link_libraries(fl_shaderc PRIVATE fl_shadercore)
fl_target_defaults(fl_shaderc)
