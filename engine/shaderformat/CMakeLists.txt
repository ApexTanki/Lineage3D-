# Header-only: the GL binding-slot table and the .fsp package format, shared by the RHI, the asset
# readers and the shader compiler without making them depend on each other (ARCHITECTURE.md §3.1).
add_library(fl_shaderformat INTERFACE)
target_link_libraries(fl_shaderformat INTERFACE fl_core)
