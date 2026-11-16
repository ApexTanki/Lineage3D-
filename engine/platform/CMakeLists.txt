add_library(fl_platform STATIC
    JobSystem.cpp
    Window.cpp
    Timer.cpp
    FileSystem.cpp
    FileWatcher.cpp
    ImGuiPlatform.cpp)
target_link_libraries(fl_platform
    PUBLIC fl_core imgui::imgui
    PRIVATE SDL3::SDL3)
fl_target_defaults(fl_platform)
