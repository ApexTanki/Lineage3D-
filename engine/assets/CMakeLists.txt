# fl_assets: binary/JSON asset formats shared by fl_assetc (writer) and the runtime (reader). No GPU code
# (ARCHITECTURE.md §3.1).
add_library(fl_assets STATIC
    Model.cpp
    AnimFile.cpp
    Texture.cpp
    Material.cpp
    MapFormat.cpp
    ParticleFormat.cpp
    TierFormat.cpp)
target_link_libraries(fl_assets
    PUBLIC fl_core
    PRIVATE meshoptimizer::meshoptimizer KTX::ktx yyjson::yyjson)
fl_target_defaults(fl_assets)
