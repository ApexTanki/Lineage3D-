#pragma once

#include "assets/ModelFormat.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace assets {

// Validated views into a loaded .fmodel file. Every offset, count, name and cross-reference is checked by
// openModel(), so a truncated or corrupt file yields an error instead of out-of-bounds reads.
struct ModelView {
    ModelHeader header{};
    std::span<const ModelNode> nodes;
    std::span<const ModelMesh> meshes;
    std::span<const ModelSubmesh> submeshes;
    std::span<const ModelMaterialSlot> materials;
    std::span<const ModelJoint> joints; // every skin's joints (M6 plan D2)
    std::span<const std::byte> blob;
};

bool openModel(std::span<const std::byte> file, ModelView& out, std::string* error);

// One mesh's decoded streams, ready to upload: positions (8 B/vertex), surface (20 B/vertex), skin (8 B/vertex, skinned
// meshes only), indices.
struct DecodedMesh {
    std::vector<std::byte> positions;
    std::vector<std::byte> surface;
    std::vector<std::byte> skin;
    std::vector<std::byte> indices;
};

// Decodes the meshopt streams and checks every index against the vertex count, and every skin vertex's joints against
// the skin and its weights against the exact sum of 255.
bool decodeMesh(const ModelView& model, uint32_t meshIndex, DecodedMesh& out, std::string* error);

// In-memory model description for the writer (fl_assetc, tests). Streams are raw; writeModel() encodes them.
struct ModelData {
    struct Mesh {
        ModelMesh info{}; // offsets/sizes are filled by writeModel()
        std::vector<PositionVertex> positions;
        std::vector<SurfaceVertex> surface;
        std::vector<SkinVertex> skin;   // skinned meshes: one per vertex
        std::vector<ModelJoint> joints; // skinned meshes: ≤ kMaxSkinJoints; writeModel() sets info.firstJoint/jointCount
        std::vector<uint32_t> indices; // written as u16 when info.indexBytes == 2
    };
    std::vector<ModelNode> nodes;
    std::vector<Mesh> meshes;
    std::vector<ModelSubmesh> submeshes;
    std::vector<ModelMaterialSlot> materials;
};

std::vector<std::byte> writeModel(const ModelData& model);

} // namespace assets
