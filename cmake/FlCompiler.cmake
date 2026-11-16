# fl_target_defaults(<target>): warning level and platform defines shared by every first-party target.
# Third-party headers are included with angle brackets and treated as external, so their warnings don't count.
function(fl_target_defaults target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} PRIVATE
            /W4 /permissive- /utf-8 /Zc:__cplusplus /EHsc
            /external:anglebrackets /external:W0
            /wd4324)   # "structure was padded due to alignment specifier": intended for aligned structs
        if(FL_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    elseif(MSVC)   # clang-cl
        # -Wno-missing-field-initializers: `VkXxxCreateInfo ci{VK_STRUCTURE_TYPE_...}` relies on the remaining
        # members being zero-initialized; that is the intended Vulkan idiom, not a missing initializer.
        target_compile_options(${target} PRIVATE /W4 /EHsc /utf-8 -Wno-c++98-compat -Wno-unused-parameter
            -Wno-missing-field-initializers)
        if(FL_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    endif()
    if(WIN32)
        target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    endif()
endfunction()
