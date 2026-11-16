add_library(fl_world STATIC
    RenderWorld.cpp)
target_link_libraries(fl_world PUBLIC fl_renderer)
fl_target_defaults(fl_world)
