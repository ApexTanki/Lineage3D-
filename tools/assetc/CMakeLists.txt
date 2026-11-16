# fl_assetc: glTF / images / builtin content → .fmodel, .ktx2, .fmat; baked maps (ARCHITECTURE.md §5.7, M1 plan D2,
# M3 plan D1).
add_library(fl_assetcore STATIC
    MeshBuilder.cpp
    Rigs.cpp
    LightmapUv.cpp
    Builtins.cpp
    ImageCook.cpp
    HdrCodec.cpp
    Sky.cpp
    RayTracer.cpp
    Baker.cpp
    MapCook.cpp
    GltfImport.cpp
    Cook.cpp)
target_include_directories(fl_assetcore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(fl_assetcore SYSTEM PRIVATE "${Stb_INCLUDE_DIR}" "${CGLTF_INCLUDE_DIR}")
target_link_libraries(fl_assetcore
    PUBLIC fl_assets
    PRIVATE meshoptimizer::meshoptimizer mikktspace::mikktspace KTX::ktx yyjson::yyjson)
fl_target_defaults(fl_assetcore)

add_executable(fl_assetc main.cpp)
target_link_libraries(fl_assetc PRIVATE fl_assetcore)
fl_target_defaults(fl_assetc)
