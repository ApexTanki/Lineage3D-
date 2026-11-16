# Backend-agnostic RHI: types, descriptors, the CommandList encoder and the transient ring.
# It knows nothing about meshes, materials or lights (ARCHITECTURE.md §3.1).
add_library(fl_rhi STATIC
    CommandList.cpp
    MemoryTracker.cpp
    TransientRing.cpp
    Types.cpp)
target_link_libraries(fl_rhi PUBLIC fl_core fl_shaderformat fl_platform)
fl_target_defaults(fl_rhi)

add_subdirectory(vulkan)
add_subdirectory(gl)

# Chooses a backend at startup. Only the application links this, never the renderer.
add_library(fl_rhi_factory STATIC Factory.cpp)
target_link_libraries(fl_rhi_factory PUBLIC fl_rhi PRIVATE fl_rhi_vulkan fl_rhi_gl)
fl_target_defaults(fl_rhi_factory)
