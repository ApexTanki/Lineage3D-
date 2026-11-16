#pragma once

// .fmodel: a model's node table, meshes, submeshes and material slots, with vertex streams and indices
// compressed by meshoptimizer's codecs (ARCHITECTURE.md §7.2). Little-endian. The file is read into memory
// once and validated by openModel(); sections are 16-byte aligned so the views need no copies.
//
// Layout: ModelHeader | ModelNode[] | ModelMesh[] | ModelSubmesh[] | ModelMaterialSlot[] | ModelJoint[] | blob (encoded data)

#include <cstdint>

namespace assets {

inline constexpr char kModelMagic[4] = {'F', 'M', 'D', 'L'};
inline constexpr uint16_t kModelVersion = 3; // v2 (M3): ModelMesh::lightmapWidth/Height; v3 (M6): name hashes, skins
inline constexpr uint32_t kModelNameLength = 48;
inline constexpr uint32_t kModelPathLength = 128;
inline constexpr uint32_t kMaxLods = 4;
inline constexpr uint32_t kMaxMeshVertices = 1u << 24; // per mesh; far above any game asset
inline constexpr uint32_t kMaxMeshIndices = 3u << 24;  // 16M triangles
inline constexpr uint32_t kMaxSkinJoints = 128;          // per skinned mesh: one bone palette (ARCHITECTURE.md R3)

// Vertex streams (ARCHITECTURE.md §7.2, D-VERTEX). Positions are int16 grid coordinates; dequantization
// (one uniform power-of-two step + offset per mesh) is folded into the object transform on the CPU.
struct PositionVertex {
    int16_t x, y, z, w; // RGBA16_SINT, w = 0
};
struct SurfaceVertex {
    int8_t normal[4];  // RGBA8_SNORM, w = 0
    int8_t tangent[4]; // RGBA8_SNORM, w = handedness (±127)
    float uv0[2];      // RG32F
    uint16_t uv1[2];   // RG16_UNORM (lightmap UVs, M3)
};
static_assert(sizeof(PositionVertex) == 8, "position stream stride is 8 bytes");
static_assert(sizeof(SurfaceVertex) == 20, "surface stream stride is 20 bytes");
// Skinned meshes only (M6 plan D2): four joint indices into the mesh's joints and their weights, which sum to exactly 255
// so that the normalized weights add up to 1 on every backend.
struct SkinVertex {
    uint8_t joints[4];  // RGBA8_UINT
    uint8_t weights[4]; // RGBA8_UNORM
};
static_assert(sizeof(SkinVertex) == 8, "skin stream stride is 8 bytes");

struct ModelHeader {
    char magic[4];
    uint16_t version;
    uint16_t flags;
    uint32_t fileSize;
    uint32_t nodeCount, meshCount, submeshCount, materialCount;
    uint32_t nodeOffset, meshOffset, submeshOffset, materialOffset;
    uint32_t blobOffset, blobSize;
    uint32_t jointCount, jointOffset; // v3
};

struct ModelNode {
    char name[kModelNameLength];
    int32_t parent;     // −1 for roots; parents always come before their children
    int32_t mesh;       // −1 = no mesh
    float translation[3];
    float rotation[4];  // quaternion x, y, z, w
    float scale[3];
    uint32_t nameHash;  // nameHash(name) (NameHash.h): clips bind to nodes by it (M6 plan D3)
};

struct ModelMesh {
    float boundsMin[3];      // object space, dequantized
    float boundsMax[3];
    float sphere[4];         // center xyz, radius
    float dequantStep;       // position = grid * step + offset
    float dequantOffset[3];
    uint32_t vertexCount;
    uint32_t indexCount;
    uint32_t indexBytes;     // 2 or 4
    uint32_t firstSubmesh;
    uint32_t submeshCount;
    uint32_t positionOffset, positionSize; // meshopt-encoded streams, relative to the blob
    uint32_t surfaceOffset, surfaceSize;
    uint32_t indexOffset, indexSize;       // meshopt-encoded index buffer
    // Reference lightmap size of the UV1 layout, in texels at 4 texels/m. At this size every chart keeps one texel of
    // padding on each side, so the baker never gives an instance much less (M3 plan D3). 0 = no UV1 layout.
    uint16_t lightmapWidth, lightmapHeight;
    // Skin (M6 plan D1, D2): joints [firstJoint, firstJoint + jointCount) of ModelJoint[]; 0 joints = a rigid mesh. A
    // skinned mesh's vertices are in bind space and its node's own transform is ignored (the glTF rule).
    uint32_t firstJoint, jointCount;
    uint32_t skinOffset, skinSize; // meshopt-encoded SkinVertex stream (skinned meshes only)
};

struct ModelLod {
    uint32_t firstIndex;
    uint32_t indexCount;
    float screenSize; // projected-size threshold (M5); 0 for the only LOD in M1
};

struct ModelSubmesh {
    uint32_t materialSlot;
    uint32_t lodCount; // 1..kMaxLods
    ModelLod lods[kMaxLods];
    float boundsMin[3];
    float boundsMax[3];
};

struct ModelMaterialSlot {
    char name[kModelNameLength];
    char path[kModelPathLength]; // data-relative .fmat path
};

// One joint of a skin (M6 plan D1, D10). Its palette entry is instance · nodeModel[node] · inverseBind · dequant.
struct ModelJoint {
    float inverseBind[12]; // 3 × 4 rows: model (bind) space → joint space
    uint32_t node;         // the node that drives the joint
    float radius;          // joint-space distance from the joint's origin to the farthest vertex it influences (culling
                           // bounds); negative = it influences no vertex
};

} // namespace assets
