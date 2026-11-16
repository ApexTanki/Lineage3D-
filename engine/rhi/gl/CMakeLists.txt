add_library(fl_rhi_gl STATIC
    GlFormats.cpp
    GlDevice.cpp
    GlResources.cpp
    GlCommands.cpp)
target_link_libraries(fl_rhi_gl
    PUBLIC fl_rhi
    PRIVATE glad::glad)
fl_target_defaults(fl_rhi_gl)
