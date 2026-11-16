# fl_anim: skeletal and node animation on the CPU (ARCHITECTURE.md §3.1, §5.5; M6 plan D6): poses, clip sampling and
# binding, layered instances, skin palettes and bounds. No GPU code.
add_library(fl_anim STATIC
    Pose.cpp
    Clip.cpp
    Instance.cpp
    Skin.cpp)
target_link_libraries(fl_anim PUBLIC fl_core fl_assets)
fl_target_defaults(fl_anim)
