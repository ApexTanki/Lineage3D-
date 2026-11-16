#pragma once

#include "assets/Model.h"
#include "core/Math.h"

#include <glm/gtc/type_precision.hpp>

#include <string>
#include <vector>

namespace tools {

struct RawVertex {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec4 tangent{1.0f, 0.0f, 0.0f, 1.0f}; // w = handedness
    glm::vec2 uv0{0.0f};                       // glTF convention: v = 0 is the top image row
    glm::vec2 uv1{0.0f};
    // Skinned meshes (M6 plan D4): up to four joints (indices into RawMesh::skinJoints) and their weights. buildMesh()
    // first rounds the weights to the 8-bit values the engine stores, so vertices whose final skin is equal weld.
    glm::u16vec4 joints{0};
    glm::vec4 weights{0.0f};
};
static_assert(sizeof(RawVertex) == 80, "RawVertex is welded by binary comparison, so it must have no padding");

struct RawSubmesh {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    uint32_t materialSlot = 0;
};

// An indexed triangle mesh before optimization; each submesh is an index range with one material.
struct RawMesh {
    std::vector<RawVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<RawSubmesh> submeshes;
    bool hasNormals = true;   // false → flat normals (the glTF rule for missing normals)
    bool hasTangents = false; // false → MikkTSpace tangents
    bool hasUv0 = true;       // without UVs, tangents are just any vector perpendicular to the normal
    bool hasUv1 = false;      // false → lightmap UVs from the chart unwrapper (LightmapUv.h)
    bool generateLods = true; // M5 plan D5; the cook manifest's "lods": false turns it off per model
    // Skin (M6 plan D1, D4): the nodes that drive the joints (indices into the model's node table) and their inverse bind
    // matrices (mesh space → joint space); empty for rigid meshes. At most assets::kMaxSkinJoints: larger skins go
    // through splitSkin() first.
    std::vector<uint32_t> skinJoints;
    std::vector<glm::mat4> inverseBinds;
};

// Splits a skinned mesh into parts that each use at most assets::kMaxSkinJoints joints (M6 plan D4): unused joints are
// dropped first, then triangles are assigned greedily in index order. A rigid mesh, or one that already fits, comes back
// as a single part. Each part keeps the submesh (material) structure of its triangles.
bool splitSkin(const RawMesh& mesh, std::vector<RawMesh>& parts, std::string& error);

// LOD generation (M5 plan D5): up to 4 levels at 50 / 25 / 12.5 % of LOD0's triangles. A level that saves less than
// a quarter, or whose error exceeds kLodMaxError of the mesh's extent, ends the chain. Its threshold is the screen size
// (bounding-sphere diameter over viewport height) at which its error covers one pixel at 1080p.
inline constexpr float kLodMaxError = 0.04f;
inline constexpr float kLodReferenceHeight = 1080.0f;
inline constexpr uint32_t kLodMinTriangles = 64; // smaller submeshes keep one level

// The converter steps of ARCHITECTURE.md §7.2: normals and MikkTSpace tangents where missing, vertex welding, LODs,
// per-LOD vertex-cache and overdraw optimization, one vertex-fetch optimization over every level, then quantization
// into the engine's streams. Appends the mesh's submeshes to `submeshes` and sets out.info.firstSubmesh/submeshCount.
// Every submesh of a mesh gets the same LOD count (one that can't simplify further repeats its last level).
// Skinned meshes also get their skin stream and joints (with joint-space bound radii); their LODs keep every vertex of a
// triangle whose corners don't share one skin in place, so only rigidly weighted regions simplify.
bool buildMesh(RawMesh raw, assets::ModelData::Mesh& out, std::vector<assets::ModelSubmesh>& submeshes, std::string& error);

} // namespace tools
