#include "MeshBuilder.h"

#include "LightmapUv.h"

#include "assets/Quantize.h"

#include <meshoptimizer.h>
#include <mikktspace.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace tools {
namespace {

// MikkTSpace over an unindexed corner list (3 corners per face).
struct MikkData {
    std::vector<RawVertex>* corners;
};

int mikkFaces(const SMikkTSpaceContext* c) {
    return int(static_cast<MikkData*>(c->m_pUserData)->corners->size() / 3);
}
int mikkFaceVertices(const SMikkTSpaceContext*, int) {
    return 3;
}
RawVertex& mikkCorner(const SMikkTSpaceContext* c, int face, int vert) {
    return (*static_cast<MikkData*>(c->m_pUserData)->corners)[size_t(face) * 3 + size_t(vert)];
}
void mikkPosition(const SMikkTSpaceContext* c, float out[], int face, int vert) {
    const glm::vec3& p = mikkCorner(c, face, vert).position;
    out[0] = p.x, out[1] = p.y, out[2] = p.z;
}
void mikkNormal(const SMikkTSpaceContext* c, float out[], int face, int vert) {
    const glm::vec3& n = mikkCorner(c, face, vert).normal;
    out[0] = n.x, out[1] = n.y, out[2] = n.z;
}
void mikkTexCoord(const SMikkTSpaceContext* c, float out[], int face, int vert) {
    // glTF's v runs down the image, while tangent-space normal maps have +Y up the image; flipping v makes the
    // generated bitangent sign match tangents exported by DCC tools (and the glTF reference viewer).
    const glm::vec2& uv = mikkCorner(c, face, vert).uv0;
    out[0] = uv.x, out[1] = 1.0f - uv.y;
}
void mikkSetTangent(const SMikkTSpaceContext* c, const float tangent[], float sign, int face, int vert) {
    mikkCorner(c, face, vert).tangent = glm::vec4(tangent[0], tangent[1], tangent[2], sign);
}

glm::vec3 anyPerpendicular(const glm::vec3& n) {
    const glm::vec3 axis = std::abs(n.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    return glm::normalize(glm::cross(axis, n));
}

int8_t snorm8(float v) {
    return int8_t(meshopt_quantizeSnorm(std::clamp(v, -1.0f, 1.0f), 8));
}

glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) {
    const float len = glm::length(v);
    return len > 1e-12f ? v / len : fallback;
}

// A vertex's skin as the engine stores it (M6 plan D2): duplicate joints merged, the four heaviest kept, weights rounded
// to 8 bits that sum to exactly 255 (largest remainder), heaviest first, unused slots joint 0 with weight 0. Written back
// as k / 255 so that vertices with the same final skin weld. False when no weight is positive or a joint is out of range.
bool canonicalSkin(RawVertex& v, size_t jointCount) {
    std::array<std::pair<uint32_t, float>, 4> slots{};
    uint32_t used = 0;
    for (int k = 0; k < 4; ++k) {
        const float w = v.weights[k];
        if (!(w > 0.0f)) continue; // negative and NaN weights count as 0
        if (v.joints[k] >= jointCount) return false;
        bool merged = false;
        for (uint32_t s = 0; s < used; ++s)
            if (slots[s].first == v.joints[k]) slots[s].second += w, merged = true;
        if (!merged) slots[used++] = {v.joints[k], w};
    }
    float total = 0.0f;
    for (uint32_t s = 0; s < used; ++s) total += slots[s].second;
    if (used == 0 || !(total > 0.0f)) return false;
    std::sort(slots.begin(), slots.begin() + used, [](const auto& a, const auto& b) { return a.second > b.second || (a.second == b.second && a.first < b.first); });
    std::array<int, 4> q{};
    std::array<float, 4> rest{};
    int sum = 0;
    for (uint32_t s = 0; s < used; ++s) {
        const float scaled = slots[s].second / total * 255.0f;
        q[s] = int(std::floor(scaled));
        rest[s] = scaled - float(q[s]);
        sum += q[s];
    }
    while (sum < 255) { // hand the missing units to the largest remainders
        uint32_t best = 0;
        for (uint32_t s = 1; s < used; ++s)
            if (rest[s] > rest[best]) best = s;
        ++q[best];
        rest[best] = -1.0f;
        ++sum;
    }
    for (int k = 0; k < 4; ++k) {
        const bool live = uint32_t(k) < used && q[size_t(k)] > 0;
        v.joints[k] = live ? uint16_t(slots[size_t(k)].first) : uint16_t(0);
        v.weights[k] = live ? float(q[size_t(k)]) / 255.0f : 0.0f;
    }
    return true;
}

bool sameSkin(const RawVertex& a, const RawVertex& b) {
    return a.joints == b.joints && a.weights == b.weights;
}

} // namespace

bool buildMesh(RawMesh raw, assets::ModelData::Mesh& out, std::vector<assets::ModelSubmesh>& submeshes, std::string& error) {
    if (raw.vertices.empty() || raw.indices.empty() || raw.indices.size() % 3 != 0) {
        error = "mesh has no triangles";
        return false;
    }
    for (uint32_t i : raw.indices)
        if (i >= raw.vertices.size()) {
            error = "index out of range";
            return false;
        }
    if (raw.submeshes.empty()) raw.submeshes.push_back({0, uint32_t(raw.indices.size()), 0});
    const bool skinned = !raw.skinJoints.empty();
    if (skinned) {
        if (raw.inverseBinds.size() != raw.skinJoints.size()) {
            error = "skin needs one inverse bind matrix per joint";
            return false;
        }
        if (raw.skinJoints.size() > assets::kMaxSkinJoints) {
            error = "skin has more than " + std::to_string(assets::kMaxSkinJoints) + " joints (split it first)";
            return false;
        }
        for (RawVertex& v : raw.vertices)
            if (!canonicalSkin(v, raw.skinJoints.size())) {
                error = "a skinned vertex has no positive weight or a joint outside the skin";
                return false;
            }
    } else {
        for (RawVertex& v : raw.vertices) v.joints = glm::u16vec4(0), v.weights = glm::vec4(0.0f); // never split welds
    }

    // 0. Lightmap UVs (M3 plan D3): the mesh's own UV1, or charts from the unwrapper. Either way the layout size is kept.
    uint16_t lightmapWidth = 0, lightmapHeight = 0;
    if (raw.hasUv1) lightmapSizeFromArea(raw, lightmapWidth, lightmapHeight);
    else generateLightmapUvs(raw, lightmapWidth, lightmapHeight);

    // 1. Flat normals and tangents are per corner, so work on an unindexed copy; welding re-indexes it.
    if (!raw.hasNormals || !raw.hasTangents) {
        std::vector<RawVertex> corners(raw.indices.size());
        for (size_t i = 0; i < raw.indices.size(); ++i) corners[i] = raw.vertices[raw.indices[i]];
        if (!raw.hasNormals) {
            for (size_t f = 0; f + 2 < corners.size(); f += 3) {
                const glm::vec3 n = safeNormalize(glm::cross(corners[f + 1].position - corners[f].position,
                                                             corners[f + 2].position - corners[f].position),
                                                  glm::vec3(0, 1, 0));
                corners[f].normal = corners[f + 1].normal = corners[f + 2].normal = n;
            }
        }
        for (RawVertex& v : corners) v.normal = safeNormalize(v.normal, glm::vec3(0, 1, 0));
        if (!raw.hasTangents) {
            bool generated = false;
            if (raw.hasUv0) {
                MikkData data{&corners};
                SMikkTSpaceInterface iface{};
                iface.m_getNumFaces = mikkFaces;
                iface.m_getNumVerticesOfFace = mikkFaceVertices;
                iface.m_getPosition = mikkPosition;
                iface.m_getNormal = mikkNormal;
                iface.m_getTexCoord = mikkTexCoord;
                iface.m_setTSpaceBasic = mikkSetTangent;
                SMikkTSpaceContext context{&iface, &data};
                generated = genTangSpaceDefault(&context) != 0;
            }
            if (!generated)
                for (RawVertex& v : corners) v.tangent = glm::vec4(anyPerpendicular(v.normal), 1.0f);
        }
        raw.vertices = std::move(corners);
        std::iota(raw.indices.begin(), raw.indices.end(), 0u);
    }

    // 2. Weld identical vertices.
    std::vector<unsigned int> remap(raw.vertices.size());
    const size_t unique = meshopt_generateVertexRemap(remap.data(), raw.indices.data(), raw.indices.size(), raw.vertices.data(),
                                                      raw.vertices.size(), sizeof(RawVertex));
    std::vector<RawVertex> vertices(unique);
    meshopt_remapVertexBuffer(vertices.data(), raw.vertices.data(), raw.vertices.size(), sizeof(RawVertex), remap.data());
    std::vector<uint32_t> indices(raw.indices.size());
    meshopt_remapIndexBuffer(indices.data(), raw.indices.data(), raw.indices.size(), remap.data());

    for (const RawSubmesh& s : raw.submeshes)
        if (s.indexCount == 0 || s.indexCount % 3 != 0 || size_t(s.firstIndex) + s.indexCount > indices.size()) {
            error = "submesh index range out of bounds";
            return false;
        }

    // 3. LODs (M5 plan D5), appended after LOD0's ranges: each level simplified from LOD0 with normals and both UV sets
    //    weighted (seams are kept: meshoptimizer treats vertices that share a position as one). Open borders are locked:
    //    neighbouring map chunks at different levels and the submeshes of a multi-material mesh stay watertight.
    const size_t submeshCount = raw.submeshes.size();
    std::vector<std::array<uint32_t, assets::kMaxLods>> lodFirst(submeshCount), lodCount(submeshCount);
    std::vector<std::array<float, assets::kMaxLods>> lodError(submeshCount);
    uint32_t meshLevels = 1;
    for (size_t s = 0; s < submeshCount; ++s) {
        lodFirst[s][0] = raw.submeshes[s].firstIndex;
        lodCount[s][0] = raw.submeshes[s].indexCount;
        lodError[s][0] = 0.0f;
    }
    std::vector<uint32_t> levels(submeshCount, 1);
    if (raw.generateLods) {
        std::vector<float> attributes(vertices.size() * 7);
        for (size_t i = 0; i < vertices.size(); ++i) {
            const RawVertex& v = vertices[i];
            const float a[7] = {v.normal.x, v.normal.y, v.normal.z, v.uv0.x, v.uv0.y, v.uv1.x, v.uv1.y};
            std::copy(a, a + 7, attributes.begin() + ptrdiff_t(i * 7));
        }
        const float weights[7] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
        const float scale = meshopt_simplifyScale(&vertices[0].position.x, vertices.size(), sizeof(RawVertex));
        const unsigned options = meshopt_SimplifyLockBorder;
        // Skinned (M6 plan D4): a collapse may only move a vertex onto a neighbour with the same skin, or the simplified
        // triangle would stretch between joints when the pose changes. Every corner of a triangle whose corners differ
        // in skin stays; rigidly weighted regions (a tank's parts) simplify as usual.
        std::vector<unsigned char> locks;
        if (skinned) {
            locks.assign(vertices.size(), 0);
            for (size_t t = 0; t + 2 < indices.size(); t += 3) {
                const uint32_t a = indices[t], b = indices[t + 1], c = indices[t + 2];
                if (!sameSkin(vertices[a], vertices[b]) || !sameSkin(vertices[a], vertices[c]))
                    locks[a] = locks[b] = locks[c] = meshopt_SimplifyVertex_Lock;
            }
        }
        std::vector<uint32_t> lodIndices, scratch;
        for (size_t s = 0; s < submeshCount; ++s) {
            const RawSubmesh& sub = raw.submeshes[s];
            if (sub.indexCount / 3 < kLodMinTriangles) continue;
            const uint32_t* source = indices.data() + sub.firstIndex;
            uint32_t previous = sub.indexCount;
            for (uint32_t l = 1; l < assets::kMaxLods; ++l) {
                const size_t target = std::max<size_t>(3, size_t(double(sub.indexCount) / double(1u << l)) / 3 * 3);
                scratch.resize(sub.indexCount);
                float relativeError = 0.0f;
                const size_t n = meshopt_simplifyWithAttributes(scratch.data(), source, sub.indexCount, &vertices[0].position.x,
                                                                vertices.size(), sizeof(RawVertex), attributes.data(), sizeof(float) * 7,
                                                                weights, 7, locks.empty() ? nullptr : locks.data(), target,
                                                                kLodMaxError, options, &relativeError);
                if (n == 0 || double(n) > double(previous) * 0.75) break; // not worth a level
                lodFirst[s][l] = uint32_t(indices.size() + lodIndices.size());
                lodCount[s][l] = uint32_t(n);
                lodError[s][l] = relativeError * scale;
                lodIndices.insert(lodIndices.end(), scratch.begin(), scratch.begin() + ptrdiff_t(n));
                levels[s] = l + 1;
                previous = uint32_t(n);
            }
            meshLevels = std::max(meshLevels, levels[s]);
        }
        indices.insert(indices.end(), lodIndices.begin(), lodIndices.end());
        // A submesh with fewer levels repeats its last one, so the whole mesh switches levels together.
        for (size_t s = 0; s < submeshCount; ++s)
            for (uint32_t l = levels[s]; l < meshLevels; ++l) {
                lodFirst[s][l] = lodFirst[s][l - 1];
                lodCount[s][l] = lodCount[s][l - 1];
                lodError[s][l] = lodError[s][l - 1];
            }
    }

    // Per LOD range: vertex cache, then overdraw (threshold 1.05, ARCHITECTURE.md §7.2). Repeated ranges once.
    for (size_t s = 0; s < submeshCount; ++s)
        for (uint32_t l = 0; l < std::max(levels[s], 1u); ++l) {
            uint32_t* range = indices.data() + lodFirst[s][l];
            meshopt_optimizeVertexCache(range, range, lodCount[s][l], vertices.size());
            meshopt_optimizeOverdraw(range, range, lodCount[s][l], &vertices[0].position.x, vertices.size(), sizeof(RawVertex), 1.05f);
        }

    // 4. One vertex-fetch pass over all indices (it also drops unreferenced vertices).
    const size_t used = meshopt_optimizeVertexFetch(vertices.data(), indices.data(), indices.size(), vertices.data(),
                                                    vertices.size(), sizeof(RawVertex));
    vertices.resize(used);

    // 5. Bounds.
    glm::vec3 lo(INFINITY), hi(-INFINITY);
    for (const RawVertex& v : vertices) {
        lo = glm::min(lo, v.position);
        hi = glm::max(hi, v.position);
    }
    const glm::vec3 center = (lo + hi) * 0.5f;
    float radius = 0.0f;
    for (const RawVertex& v : vertices) radius = std::max(radius, glm::length(v.position - center));

    // 6. Quantize into the engine streams.
    const assets::Dequantization dq = assets::chooseDequantization(lo, hi);
    out.positions.resize(vertices.size());
    out.surface.resize(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        const RawVertex& v = vertices[i];
        out.positions[i] = {assets::quantizeCoordinate(v.position.x, dq.step, dq.offset.x),
                            assets::quantizeCoordinate(v.position.y, dq.step, dq.offset.y),
                            assets::quantizeCoordinate(v.position.z, dq.step, dq.offset.z), 0};
        const glm::vec3 n = safeNormalize(v.normal, glm::vec3(0, 1, 0));
        const glm::vec3 t = safeNormalize(glm::vec3(v.tangent), anyPerpendicular(n));
        assets::SurfaceVertex& s = out.surface[i];
        s.normal[0] = snorm8(n.x), s.normal[1] = snorm8(n.y), s.normal[2] = snorm8(n.z), s.normal[3] = 0;
        s.tangent[0] = snorm8(t.x), s.tangent[1] = snorm8(t.y), s.tangent[2] = snorm8(t.z);
        s.tangent[3] = v.tangent.w < 0.0f ? int8_t(-127) : int8_t(127);
        s.uv0[0] = v.uv0.x, s.uv0[1] = v.uv0.y;
        s.uv1[0] = uint16_t(std::lround(std::clamp(v.uv1.x, 0.0f, 1.0f) * 65535.0f));
        s.uv1[1] = uint16_t(std::lround(std::clamp(v.uv1.y, 0.0f, 1.0f) * 65535.0f));
    }
    // Skin stream and joints (M6 plan D2, D10). A joint's bound radius is measured in its own space on the quantized
    // positions the GPU skins, so the runtime bounds contain every vertex exactly.
    out.skin.clear();
    out.joints.clear();
    if (skinned) {
        out.skin.resize(vertices.size());
        std::vector<float> jointRadius(raw.skinJoints.size(), -1.0f);
        for (size_t i = 0; i < vertices.size(); ++i) {
            const RawVertex& v = vertices[i];
            assets::SkinVertex& s = out.skin[i];
            const glm::vec3 p = assets::dequantize(&out.positions[i].x, dq);
            for (int k = 0; k < 4; ++k) {
                s.joints[k] = uint8_t(v.joints[k]);
                s.weights[k] = uint8_t(std::lround(v.weights[k] * 255.0f));
                if (s.weights[k] == 0) continue;
                const glm::vec3 local(raw.inverseBinds[v.joints[k]] * glm::vec4(p, 1.0f));
                jointRadius[v.joints[k]] = std::max(jointRadius[v.joints[k]], glm::length(local));
            }
        }
        for (size_t j = 0; j < raw.skinJoints.size(); ++j) {
            assets::ModelJoint joint{};
            const glm::mat4& m = raw.inverseBinds[j]; // column-major: m[column][row]
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c) joint.inverseBind[4 * r + c] = m[c][r];
            joint.node = raw.skinJoints[j];
            // A grid step and a relative ten-thousandth of slack: the runtime bound is exact in real numbers, not in floats.
            joint.radius = jointRadius[j] < 0.0f ? -1.0f : jointRadius[j] * 1.0001f + dq.step;
            out.joints.push_back(joint);
        }
    }
    out.indices = std::move(indices);

    assets::ModelMesh& info = out.info;
    info = {};
    for (int a = 0; a < 3; ++a) {
        info.boundsMin[a] = lo[a];
        info.boundsMax[a] = hi[a];
        info.sphere[a] = center[a];
        info.dequantOffset[a] = dq.offset[a];
    }
    info.sphere[3] = radius;
    info.dequantStep = dq.step;
    info.indexBytes = vertices.size() <= 65536 ? 2u : 4u;
    info.firstSubmesh = uint32_t(submeshes.size());
    info.submeshCount = uint32_t(raw.submeshes.size());
    info.lightmapWidth = lightmapWidth;
    info.lightmapHeight = lightmapHeight;
    // LOD thresholds from the mesh's worst error per level: the screen size (sphere diameter / viewport height) at which
    // that error spans one pixel at 1080p. A lossless level (error 0) is good at any size.
    float screenSize[assets::kMaxLods] = {};
    for (uint32_t l = 1; l < meshLevels; ++l) {
        float worst = 0.0f;
        for (size_t s = 0; s < submeshCount; ++s) worst = std::max(worst, lodError[s][l]);
        screenSize[l] = worst > 0.0f ? std::min(1e6f, 2.0f * radius / (worst * kLodReferenceHeight)) : 1e6f;
        if (l > 1) screenSize[l] = std::min(screenSize[l], screenSize[l - 1]); // coarser levels never need a larger size
    }
    for (size_t si = 0; si < submeshCount; ++si) {
        const RawSubmesh& s = raw.submeshes[si];
        assets::ModelSubmesh sub{};
        sub.materialSlot = s.materialSlot;
        sub.lodCount = meshLevels;
        for (uint32_t l = 0; l < meshLevels; ++l) sub.lods[l] = {lodFirst[si][l], lodCount[si][l], screenSize[l]};
        glm::vec3 slo(INFINITY), shi(-INFINITY);
        for (uint32_t i = s.firstIndex; i < s.firstIndex + s.indexCount; ++i) {
            slo = glm::min(slo, vertices[out.indices[i]].position);
            shi = glm::max(shi, vertices[out.indices[i]].position);
        }
        for (int a = 0; a < 3; ++a) {
            sub.boundsMin[a] = slo[a];
            sub.boundsMax[a] = shi[a];
        }
        submeshes.push_back(sub);
    }
    return true;
}

} // namespace tools

namespace tools {

bool splitSkin(const RawMesh& mesh, std::vector<RawMesh>& parts, std::string& error) {
    parts.clear();
    if (mesh.skinJoints.empty()) {
        parts.push_back(mesh);
        return true;
    }
    if (mesh.inverseBinds.size() != mesh.skinJoints.size()) {
        error = "skin needs one inverse bind matrix per joint";
        return false;
    }
    const size_t triangles = mesh.indices.size() / 3;
    std::vector<RawSubmesh> submeshes = mesh.submeshes;
    if (submeshes.empty()) submeshes.push_back({0, uint32_t(mesh.indices.size()), 0});
    std::vector<uint32_t> triangleSubmesh(triangles, 0);
    for (size_t s = 0; s < submeshes.size(); ++s)
        for (uint32_t i = submeshes[s].firstIndex; i + 2 < submeshes[s].firstIndex + submeshes[s].indexCount && i / 3 < triangles; i += 3)
            triangleSubmesh[i / 3] = uint32_t(s);

    // The joints a triangle really uses (positive weight), at most 12.
    auto triangleJoints = [&](size_t t, std::array<uint32_t, 12>& out) {
        uint32_t n = 0;
        for (int c = 0; c < 3; ++c) {
            const uint32_t vi = mesh.indices[t * 3 + size_t(c)];
            if (vi >= mesh.vertices.size()) continue;
            const RawVertex& v = mesh.vertices[vi];
            for (int k = 0; k < 4; ++k) {
                if (!(v.weights[k] > 0.0f) || v.joints[k] >= mesh.skinJoints.size()) continue;
                const uint32_t j = v.joints[k];
                if (std::find(out.begin(), out.begin() + n, j) == out.begin() + n) out[n++] = j;
            }
        }
        return n;
    };

    // Greedy: triangles in index order join the current part while its joint set stays within the limit.
    std::vector<uint32_t> trianglePart(triangles, 0);
    std::vector<uint8_t> inPart(mesh.skinJoints.size(), 0);
    std::vector<uint32_t> partJoints;
    uint32_t partCount = 1;
    std::array<uint32_t, 12> tj{};
    for (size_t t = 0; t < triangles; ++t) {
        const uint32_t n = triangleJoints(t, tj);
        uint32_t fresh = 0;
        for (uint32_t k = 0; k < n; ++k) fresh += inPart[tj[k]] ? 0u : 1u;
        if (partJoints.size() + fresh > assets::kMaxSkinJoints) {
            for (uint32_t j : partJoints) inPart[j] = 0;
            partJoints.clear();
            ++partCount;
        }
        for (uint32_t k = 0; k < n; ++k)
            if (!inPart[tj[k]]) inPart[tj[k]] = 1, partJoints.push_back(tj[k]);
        trianglePart[t] = partCount - 1;
    }

    // Each part: its own vertices and joints (in first-use order), its triangles grouped by submesh.
    for (uint32_t p = 0; p < partCount; ++p) {
        RawMesh part;
        part.hasNormals = mesh.hasNormals;
        part.hasTangents = mesh.hasTangents;
        part.hasUv0 = mesh.hasUv0;
        part.hasUv1 = mesh.hasUv1;
        part.generateLods = mesh.generateLods;
        std::vector<int32_t> vertexMap(mesh.vertices.size(), -1), jointMap(mesh.skinJoints.size(), -1);
        for (size_t s = 0; s < submeshes.size(); ++s) {
            const uint32_t first = uint32_t(part.indices.size());
            for (size_t t = 0; t < triangles; ++t) {
                if (trianglePart[t] != p || triangleSubmesh[t] != s) continue;
                for (int c = 0; c < 3; ++c) {
                    const uint32_t vi = mesh.indices[t * 3 + size_t(c)];
                    if (vi >= mesh.vertices.size()) {
                        error = "index out of range";
                        return false;
                    }
                    if (vertexMap[vi] < 0) {
                        RawVertex v = mesh.vertices[vi];
                        for (int k = 0; k < 4; ++k) {
                            if (!(v.weights[k] > 0.0f) || v.joints[k] >= mesh.skinJoints.size()) {
                                v.joints[k] = 0;
                                v.weights[k] = 0.0f;
                                continue;
                            }
                            int32_t& local = jointMap[v.joints[k]];
                            if (local < 0) {
                                local = int32_t(part.skinJoints.size());
                                part.skinJoints.push_back(mesh.skinJoints[v.joints[k]]);
                                part.inverseBinds.push_back(mesh.inverseBinds[v.joints[k]]);
                            }
                            v.joints[k] = uint16_t(local);
                        }
                        vertexMap[vi] = int32_t(part.vertices.size());
                        part.vertices.push_back(v);
                    }
                    part.indices.push_back(uint32_t(vertexMap[vi]));
                }
            }
            const uint32_t count = uint32_t(part.indices.size()) - first;
            if (count) part.submeshes.push_back({first, count, submeshes[s].materialSlot});
        }
        if (!part.indices.empty()) parts.push_back(std::move(part));
    }
    return true;
}

} // namespace tools
