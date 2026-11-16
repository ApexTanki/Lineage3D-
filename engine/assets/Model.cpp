#include "assets/Model.h"

#include "assets/NameHash.h"

#include <meshoptimizer.h>

#include <cmath>
#include <cstring>

namespace assets {
namespace {

bool fail(std::string* error, const char* message) {
    if (error) *error = message;
    return false;
}

size_t align16(size_t v) {
    return (v + 15) & ~size_t(15);
}

template <typename T>
bool section(std::span<const std::byte> file, uint32_t offset, uint32_t count, std::span<const T>& out) {
    if (count == 0) {
        out = {};
        return true;
    }
    if (offset % 16 != 0) return false;
    if (size_t(offset) + size_t(count) * sizeof(T) > file.size()) return false;
    out = {reinterpret_cast<const T*>(file.data() + offset), count};
    return true;
}

template <size_t N>
bool terminated(const char (&s)[N]) {
    return std::memchr(s, 0, N) != nullptr;
}

bool inBlob(uint32_t offset, uint32_t size, size_t blobSize) {
    return size_t(offset) + size_t(size) <= blobSize;
}

template <typename T>
void put(std::vector<std::byte>& out, size_t offset, const T* data, size_t count) {
    if (count) std::memcpy(out.data() + offset, data, sizeof(T) * count);
}

} // namespace

bool openModel(std::span<const std::byte> file, ModelView& out, std::string* error) {
    if (file.size() < sizeof(ModelHeader)) return fail(error, "file smaller than its header");
    if (reinterpret_cast<uintptr_t>(file.data()) % 16 != 0) return fail(error, "file buffer is not 16-byte aligned");
    std::memcpy(&out.header, file.data(), sizeof(ModelHeader));
    const ModelHeader& h = out.header;
    if (std::memcmp(h.magic, kModelMagic, 4) != 0) return fail(error, "bad magic");
    if (h.version != kModelVersion) return fail(error, "unsupported model version");
    if (h.fileSize != file.size()) return fail(error, "size mismatch (truncated file?)");
    if (!section(file, h.nodeOffset, h.nodeCount, out.nodes) || !section(file, h.meshOffset, h.meshCount, out.meshes) ||
        !section(file, h.submeshOffset, h.submeshCount, out.submeshes) ||
        !section(file, h.materialOffset, h.materialCount, out.materials) ||
        !section(file, h.jointOffset, h.jointCount, out.joints))
        return fail(error, "section out of bounds or misaligned");
    if (size_t(h.blobOffset) + h.blobSize > file.size()) return fail(error, "blob out of bounds");
    out.blob = file.subspan(h.blobOffset, h.blobSize);

    for (uint32_t i = 0; i < h.nodeCount; ++i) {
        const ModelNode& n = out.nodes[i];
        if (!terminated(n.name)) return fail(error, "unterminated node name");
        if (n.parent < -1 || n.parent >= int32_t(i)) return fail(error, "node parent must precede the node");
        if (n.mesh < -1 || n.mesh >= int32_t(h.meshCount)) return fail(error, "node mesh index out of range");
        if (n.nameHash != nameHash(n.name)) return fail(error, "node name hash mismatch");
    }
    for (const ModelMaterialSlot& m : out.materials)
        if (!terminated(m.name) || !terminated(m.path)) return fail(error, "unterminated material slot");
    for (const ModelJoint& j : out.joints) {
        if (j.node >= h.nodeCount) return fail(error, "joint node out of range");
        if (!(j.radius >= -1.0f && j.radius < 1e9f)) return fail(error, "invalid joint radius");
        for (float v : j.inverseBind)
            if (!std::isfinite(v)) return fail(error, "invalid inverse bind matrix");
    }
    for (const ModelMesh& m : out.meshes) {
        if (m.vertexCount == 0 || m.indexCount == 0 || m.indexCount % 3 != 0) return fail(error, "empty or non-triangle mesh");
        // The decoder allocates from these counts, so a corrupt header must not be able to ask for gigabytes.
        if (m.vertexCount > kMaxMeshVertices || m.indexCount > kMaxMeshIndices) return fail(error, "mesh exceeds the size limits");
        if (m.indexBytes != 2 && m.indexBytes != 4) return fail(error, "index size must be 2 or 4");
        if (m.indexBytes == 2 && m.vertexCount > 65536) return fail(error, "16-bit indices with more than 65536 vertices");
        if (!(m.dequantStep > 0.0f) || !std::isfinite(m.dequantStep)) return fail(error, "invalid dequantization step");
        if (m.submeshCount == 0 || m.firstSubmesh > h.submeshCount || m.submeshCount > h.submeshCount - m.firstSubmesh)
            return fail(error, "mesh submesh range out of bounds");
        if (!inBlob(m.positionOffset, m.positionSize, h.blobSize) || !inBlob(m.surfaceOffset, m.surfaceSize, h.blobSize) ||
            !inBlob(m.indexOffset, m.indexSize, h.blobSize))
            return fail(error, "mesh stream out of the blob");
        if (m.jointCount > 0) {
            if (m.jointCount > kMaxSkinJoints) return fail(error, "skin exceeds the joint limit");
            if (m.firstJoint > h.jointCount || m.jointCount > h.jointCount - m.firstJoint) return fail(error, "skin joint range out of bounds");
            if (m.skinSize == 0 || !inBlob(m.skinOffset, m.skinSize, h.blobSize)) return fail(error, "skin stream out of the blob");
        }
        for (uint32_t s = m.firstSubmesh; s < m.firstSubmesh + m.submeshCount; ++s) {
            const ModelSubmesh& sub = out.submeshes[s];
            if (sub.materialSlot >= h.materialCount) return fail(error, "submesh material slot out of range");
            if (sub.lodCount == 0 || sub.lodCount > kMaxLods) return fail(error, "submesh LOD count out of range");
            for (uint32_t l = 0; l < sub.lodCount; ++l) {
                const ModelLod& lod = sub.lods[l];
                if (lod.indexCount == 0 || lod.indexCount % 3 != 0 || lod.firstIndex > m.indexCount ||
                    lod.indexCount > m.indexCount - lod.firstIndex)
                    return fail(error, "submesh index range out of bounds");
                // M5 plan D5: level l ≥ 1 serves screen sizes up to its threshold; coarser levels never need more.
                if (l > 0 && !(lod.screenSize > 0.0f && lod.screenSize <= 1e6f && (l == 1 || lod.screenSize <= sub.lods[l - 1].screenSize)))
                    return fail(error, "submesh LOD thresholds must be positive and decreasing");
            }
            if (sub.lodCount != out.submeshes[m.firstSubmesh].lodCount) return fail(error, "the submeshes of a mesh need one LOD count");
        }
    }
    return true;
}

bool decodeMesh(const ModelView& model, uint32_t meshIndex, DecodedMesh& out, std::string* error) {
    if (meshIndex >= model.meshes.size()) return fail(error, "mesh index out of range");
    const ModelMesh& m = model.meshes[meshIndex];
    out.positions.resize(size_t(m.vertexCount) * sizeof(PositionVertex));
    out.surface.resize(size_t(m.vertexCount) * sizeof(SurfaceVertex));
    out.indices.resize(size_t(m.indexCount) * m.indexBytes);
    const auto* blob = reinterpret_cast<const unsigned char*>(model.blob.data());
    if (meshopt_decodeVertexBuffer(out.positions.data(), m.vertexCount, sizeof(PositionVertex), blob + m.positionOffset,
                                   m.positionSize) != 0)
        return fail(error, "position stream is corrupt");
    if (meshopt_decodeVertexBuffer(out.surface.data(), m.vertexCount, sizeof(SurfaceVertex), blob + m.surfaceOffset,
                                   m.surfaceSize) != 0)
        return fail(error, "surface stream is corrupt");
    if (meshopt_decodeIndexBuffer(out.indices.data(), m.indexCount, m.indexBytes, blob + m.indexOffset, m.indexSize) != 0)
        return fail(error, "index buffer is corrupt");
    out.skin.clear();
    if (m.jointCount > 0) {
        out.skin.resize(size_t(m.vertexCount) * sizeof(SkinVertex));
        if (meshopt_decodeVertexBuffer(out.skin.data(), m.vertexCount, sizeof(SkinVertex), blob + m.skinOffset, m.skinSize) != 0)
            return fail(error, "skin stream is corrupt");
        // A joint past the skin would read another mesh's palette entry (or past the 128-entry block).
        const auto* sv = reinterpret_cast<const SkinVertex*>(out.skin.data());
        for (uint32_t v = 0; v < m.vertexCount; ++v) {
            uint32_t sum = 0;
            for (int k = 0; k < 4; ++k) {
                if (sv[v].joints[k] >= m.jointCount) return fail(error, "skin joint index out of range");
                sum += sv[v].weights[k];
            }
            if (sum != 255) return fail(error, "skin weights must sum to 255");
        }
    }
    // An out-of-range index would make the GPU fetch outside the vertex streams.
    if (m.indexBytes == 2) {
        const auto* ix = reinterpret_cast<const uint16_t*>(out.indices.data());
        for (uint32_t i = 0; i < m.indexCount; ++i)
            if (ix[i] >= m.vertexCount) return fail(error, "index out of range");
    } else {
        const auto* ix = reinterpret_cast<const uint32_t*>(out.indices.data());
        for (uint32_t i = 0; i < m.indexCount; ++i)
            if (ix[i] >= m.vertexCount) return fail(error, "index out of range");
    }
    return true;
}

std::vector<std::byte> writeModel(const ModelData& model) {
    std::vector<unsigned char> blob;
    std::vector<ModelMesh> meshes;
    std::vector<ModelJoint> joints;
    meshes.reserve(model.meshes.size());
    auto reserveAligned = [&](size_t bound) {
        const size_t offset = align16(blob.size());
        blob.resize(offset + bound);
        return offset;
    };
    for (const ModelData::Mesh& mesh : model.meshes) {
        ModelMesh info = mesh.info;
        info.vertexCount = uint32_t(mesh.positions.size());
        info.indexCount = uint32_t(mesh.indices.size());

        size_t offset = reserveAligned(meshopt_encodeVertexBufferBound(mesh.positions.size(), sizeof(PositionVertex)));
        size_t size = meshopt_encodeVertexBuffer(blob.data() + offset, blob.size() - offset, mesh.positions.data(),
                                                 mesh.positions.size(), sizeof(PositionVertex));
        blob.resize(offset + size);
        info.positionOffset = uint32_t(offset);
        info.positionSize = uint32_t(size);

        offset = reserveAligned(meshopt_encodeVertexBufferBound(mesh.surface.size(), sizeof(SurfaceVertex)));
        size = meshopt_encodeVertexBuffer(blob.data() + offset, blob.size() - offset, mesh.surface.data(),
                                          mesh.surface.size(), sizeof(SurfaceVertex));
        blob.resize(offset + size);
        info.surfaceOffset = uint32_t(offset);
        info.surfaceSize = uint32_t(size);

        offset = reserveAligned(meshopt_encodeIndexBufferBound(mesh.indices.size(), mesh.positions.size()));
        size = meshopt_encodeIndexBuffer(blob.data() + offset, blob.size() - offset, mesh.indices.data(), mesh.indices.size());
        blob.resize(offset + size);
        info.indexOffset = uint32_t(offset);
        info.indexSize = uint32_t(size);

        info.firstJoint = uint32_t(joints.size());
        info.jointCount = uint32_t(mesh.joints.size());
        info.skinOffset = info.skinSize = 0;
        if (!mesh.joints.empty()) {
            joints.insert(joints.end(), mesh.joints.begin(), mesh.joints.end());
            offset = reserveAligned(meshopt_encodeVertexBufferBound(mesh.skin.size(), sizeof(SkinVertex)));
            size = meshopt_encodeVertexBuffer(blob.data() + offset, blob.size() - offset, mesh.skin.data(), mesh.skin.size(),
                                              sizeof(SkinVertex));
            blob.resize(offset + size);
            info.skinOffset = uint32_t(offset);
            info.skinSize = uint32_t(size);
        } else {
            info.firstJoint = 0;
        }
        meshes.push_back(info);
    }

    ModelHeader h{};
    std::memcpy(h.magic, kModelMagic, 4);
    h.version = kModelVersion;
    h.nodeCount = uint32_t(model.nodes.size());
    h.meshCount = uint32_t(meshes.size());
    h.submeshCount = uint32_t(model.submeshes.size());
    h.materialCount = uint32_t(model.materials.size());
    size_t offset = align16(sizeof(ModelHeader));
    h.nodeOffset = uint32_t(offset);
    offset = align16(offset + model.nodes.size() * sizeof(ModelNode));
    h.meshOffset = uint32_t(offset);
    offset = align16(offset + meshes.size() * sizeof(ModelMesh));
    h.submeshOffset = uint32_t(offset);
    offset = align16(offset + model.submeshes.size() * sizeof(ModelSubmesh));
    h.materialOffset = uint32_t(offset);
    offset = align16(offset + model.materials.size() * sizeof(ModelMaterialSlot));
    h.jointCount = uint32_t(joints.size());
    h.jointOffset = uint32_t(offset);
    offset = align16(offset + joints.size() * sizeof(ModelJoint));
    h.blobOffset = uint32_t(offset);
    h.blobSize = uint32_t(blob.size());
    h.fileSize = uint32_t(offset + blob.size());

    std::vector<std::byte> out(h.fileSize);
    put(out, 0, &h, 1);
    std::vector<ModelNode> nodes = model.nodes;
    for (ModelNode& n : nodes) n.nameHash = nameHash(n.name); // the reader checks it, so the writer owns it
    put(out, h.nodeOffset, nodes.data(), nodes.size());
    put(out, h.meshOffset, meshes.data(), meshes.size());
    put(out, h.submeshOffset, model.submeshes.data(), model.submeshes.size());
    put(out, h.materialOffset, model.materials.data(), model.materials.size());
    put(out, h.jointOffset, joints.data(), joints.size());
    if (!blob.empty()) std::memcpy(out.data() + h.blobOffset, blob.data(), blob.size());
    return out;
}

} // namespace assets
