# fl_add_shader_package(<name> <manifest>)
# Compiles one shader template, with every permutation listed in its manifest, into
# <data>/shaders/<name>.fsp at build time. The pipeline is GLSL 450 → SPIR-V (glslang) →
# GLSL 330 (SPIRV-Cross), plus reflection (ARCHITECTURE.md §4.10, §7.5).
# Call it from the top-level CMakeLists so the depfile's target path is relative to CMAKE_BINARY_DIR.
function(fl_add_shader_package name manifest)
    set(output "${FL_DATA_DIR}/shaders/${name}.fsp")
    set(depfile "${CMAKE_BINARY_DIR}/shaders/${name}.fsp.d")
    file(RELATIVE_PATH depTarget "${CMAKE_BINARY_DIR}" "${output}")
    add_custom_command(
        OUTPUT "${output}"
        COMMAND fl_shaderc
                --manifest "${manifest}"
                --include "${CMAKE_SOURCE_DIR}/shaders/include"
                --out "${output}"
                --depfile "${depfile}"
                --depfile-target "${depTarget}"
        DEPENDS fl_shaderc "${manifest}"
        DEPFILE "${depfile}"
        COMMENT "Compiling shader package ${name}"
        VERBATIM)
    set_property(GLOBAL APPEND PROPERTY FL_SHADER_PACKAGES "${output}")
endfunction()

# fl_finalize_shader_packages(): one target that builds every declared package.
function(fl_finalize_shader_packages)
    get_property(packages GLOBAL PROPERTY FL_SHADER_PACKAGES)
    add_custom_target(fl_shaders ALL DEPENDS ${packages})
endfunction()
