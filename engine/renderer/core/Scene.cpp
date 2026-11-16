#include "renderer/core/Scene.h"

#include "anim/Skin.h"
#include "assets/Quantize.h"
#include "core/Log.h"
#include "core/Profile.h"
#include "platform/JobSystem.h"
#include "renderer/materials/MaterialSystem.h"
#include "rhi/TransientRing.h"

#include <algorithm>
#include <cfloat>
#include <tuple>

namespace renderer {
namespace {

glm::mat4 localMatrix(const assets::ModelNode& n) {
    const glm::quat q(n.rotation[3], n.rotation[0], n.rotation[1], n.rotation[2]); // glm: (w, x, y, z)
    return glm::translate(glm::mat4(1.0f), glm::vec3(n.translation[0], n.translation[1], n.translation[2])) *
           glm::mat4_cast(q) * glm::scale(glm::mat4(1.0f), glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
}

glm::vec4 worldSphere(const glm::mat4& m, const glm::vec4& sphere) {
    const float scale = std::max(glm::length(glm::vec3(m[0])), std::max(glm::length(glm::vec3(m[1])), glm::length(glm::vec3(m[2]))));
    return glm::vec4(glm::vec3(m * glm::vec4(glm::vec3(sphere), 1.0f)), sphere.w * scale);
}

// An object-space box under a transform: the world center and the three world half-axis vectors.
void orientedBox(const glm::mat4& m, const glm::vec3& boundsMin, const glm::vec3& boundsMax, glm::vec3& center, glm::vec3 halfAxes[3]) {
    center = glm::vec3(m * glm::vec4((boundsMin + boundsMax) * 0.5f, 1.0f));
    const glm::vec3 half = (boundsMax - boundsMin) * 0.5f;
    for (int c = 0; c < 3; ++c) halfAxes[c] = glm::vec3(m[c]) * half[c];
}

} // namespace

Scene::Scene(ResourceManager& resources, MaterialSystem& materials) : m_resources(resources), m_materials(materials) {}

Scene::~Scene() {
    clear();
}

InstanceId Scene::add(const InstanceDesc& desc) {
    if (!m_resources.model(desc.model)) {
        FL_LOG_ERROR("Scene::add: invalid model handle");
        return {};
    }
    Instance instance;
    instance.model = desc.model;
    instance.transform = desc.transform;
    instance.isStatic = desc.isStatic;
    instance.layer = desc.layer;
    instance.castsShadows = desc.castsShadows;
    instance.blobShadow = desc.blobShadow && !desc.isStatic;
    instance.occluder = desc.occluder && desc.isStatic;
    instance.animated = desc.animated && !desc.isStatic;
    instance.overrides.assign(desc.materialOverrides.begin(), desc.materialOverrides.end());
    if (desc.isStatic) {
        instance.lightmaps.assign(desc.lightmaps.begin(), desc.lightmaps.end());
        instance.reflectionSlot = std::min<uint8_t>(desc.reflectionSlot, uint8_t(kReflectionSlots - 1));
    }
    const InstanceId id = m_instances.create(std::move(instance));
    if (!id) {
        FL_LOG_ERROR("Scene instance pool exhausted");
        return {};
    }
    m_resources.addRef(desc.model);
    for (MaterialHandle m : desc.materialOverrides) m_materials.addRef(m);
    m_unrealized.push_back(id);
    return id;
}

void Scene::setTransform(InstanceId id, const glm::mat4& transform) {
    Instance* instance = m_instances.get(id);
    if (!instance) return;
    instance->transform = transform;
    if (!instance->realized) return;
    if (instance->animation != ~0u) return; // animate() applies it to the animated nodes every frame
    if (instance->isStatic) {
        m_staticDirty = true;
        return;
    }
    const ModelAsset* model = m_resources.model(instance->model);
    if (!model) return;
    for (uint32_t index : instance->dynamicIndices) {
        Dynamic& d = m_dynamics[index];
        writeDynamic(d, transform * m_dynamicNodeWorld[index], model->meshes[m_meshRefs[d.meshRef].mesh]);
    }
}

void Scene::remove(InstanceId id) {
    Instance* instance = m_instances.get(id);
    if (!instance) return;
    const bool wasRealized = instance->realized;
    const bool wasStatic = instance->isStatic;
    releaseAnimation(*instance);
    m_resources.release(instance->model);
    for (MaterialHandle m : instance->overrides) m_materials.release(m);
    m_instances.destroy(id);
    m_unrealized.erase(std::remove(m_unrealized.begin(), m_unrealized.end(), id), m_unrealized.end());
    if (wasRealized) {
        if (wasStatic) m_staticDirty = true;
        else rebuildDynamics();
    }
}

LightId Scene::addLight(const LightDesc& desc) {
    const LightId id = m_lights.create(desc);
    if (!id) FL_LOG_ERROR("Scene light pool exhausted (%u lights)", m_lights.capacity());
    return id;
}

void Scene::updateLight(LightId id, const LightDesc& desc) {
    if (LightDesc* l = m_lights.get(id)) *l = desc;
}

void Scene::removeLight(LightId id) {
    if (m_lights.get(id)) m_lights.destroy(id);
}

void Scene::clear() {
    std::vector<LightId> lights;
    m_lights.forEach([&](LightId id, LightDesc&) { lights.push_back(id); });
    for (LightId id : lights) m_lights.destroy(id);
    std::vector<InstanceId> ids;
    m_instances.forEach([&](InstanceId id, Instance&) { ids.push_back(id); });
    for (InstanceId id : ids) {
        Instance* instance = m_instances.get(id);
        m_resources.release(instance->model);
        for (MaterialHandle m : instance->overrides) m_materials.release(m);
        m_instances.destroy(id);
    }
    m_unrealized.clear();
    m_animations.clear();
    m_freeAnimations.clear();
    m_liveAnimations.clear();
    m_skinnedDynamics.clear();
    m_bindings.clear();
    m_dynamics.clear();
    m_dynamicNodeWorld.clear();
    m_dynamicMaterials.clear();
    m_groups.clear();
    m_staticRows.clear();
    m_staticSpheres.clear();
    m_staticRowGroup.clear();
    m_staticLightmapST.clear();
    m_staticDrawFlags.clear();
    m_blobs.clear();
    m_rowX.clear(), m_rowY.clear(), m_rowZ.clear(), m_rowR.clear();
    m_rowBoxMin.clear(), m_rowBoxMax.clear();
    m_occluders.clear();
    m_meshRefs.clear();
    m_meshRefIndex.clear();
    m_staticDirty = false;
    ++m_staticRevision;
    ++m_dynamicRevision;
}

void Scene::update() {
    FL_ZONE("Scene::update");
    for (size_t i = 0; i < m_unrealized.size();) {
        Instance* instance = m_instances.get(m_unrealized[i]);
        const LoadState state = instance ? m_resources.state(instance->model) : LoadState::Failed;
        if (instance && state == LoadState::Loading) {
            ++i;
            continue;
        }
        if (instance && state == LoadState::Ready) {
            instance->realized = true;
            if (instance->isStatic) m_staticDirty = true;
            else realizeDynamic(m_unrealized[i], *instance);
        }
        // A failed model was already reported by the resource manager; its instances simply never draw.
        m_unrealized[i] = m_unrealized.back();
        m_unrealized.pop_back();
    }
    if (m_staticDirty) {
        rebuildStatics();
        m_staticDirty = false;
    }
}

void Scene::nodeWorldMatrices(const ModelAsset& model, std::vector<glm::mat4>& out) const {
    out.resize(model.nodes.size());
    for (size_t i = 0; i < model.nodes.size(); ++i) {
        const assets::ModelNode& n = model.nodes[i];
        out[i] = n.parent >= 0 ? out[size_t(n.parent)] * localMatrix(n) : localMatrix(n); // parents come first
    }
}

MaterialHandle Scene::materialFor(const Instance& instance, const ModelAsset& model, uint32_t slot) const {
    if (slot < instance.overrides.size() && instance.overrides[slot]) return instance.overrides[slot];
    return slot < model.materials.size() ? model.materials[slot] : MaterialHandle{};
}

uint32_t Scene::meshRefFor(ModelHandle model, uint32_t mesh) {
    const uint64_t key = uint64_t(model.bits) << 32 | mesh;
    if (auto it = m_meshRefIndex.find(key); it != m_meshRefIndex.end()) return it->second;
    const auto index = uint32_t(m_meshRefs.size());
    m_meshRefs.push_back({model, mesh});
    m_meshRefIndex.emplace(key, index);
    return index;
}

void Scene::writeDynamic(Dynamic& d, const glm::mat4& objectToWorld, const GpuMesh& mesh) const {
    d.rows = assets::foldDequantization(core::Affine3x4::fromMat4(objectToWorld), mesh.dequant);
    d.sphere = worldSphere(objectToWorld, mesh.sphere);
    d.mirrored = glm::determinant(glm::mat3(objectToWorld)) < 0.0f;
    orientedBox(objectToWorld, mesh.boundsMin, mesh.boundsMax, d.boxCenter, d.boxHalfAxes);
    const glm::vec3 extent = glm::abs(d.boxHalfAxes[0]) + glm::abs(d.boxHalfAxes[1]) + glm::abs(d.boxHalfAxes[2]);
    d.boxMin = d.boxCenter - extent;
    d.boxMax = d.boxCenter + extent;
    d.lodCount = mesh.lodCount;
    std::copy(std::begin(mesh.lodScreenSize), std::end(mesh.lodScreenSize), std::begin(d.lodScreenSize));
}

void Scene::realizeDynamic(InstanceId id, Instance& instance) {
    const ModelAsset* model = m_resources.model(instance.model);
    if (!model) return;
    // Animated instances (M6 plan D8) take their nodes from the pose; skinned models always have one (the rest pose).
    if ((instance.animated || model->skinned) && instance.animation == ~0u) bindAnimation(id, instance, *model);
    const AnimState* state = instance.animation != ~0u ? m_animations[instance.animation].get() : nullptr;
    const core::Affine3x4 world = core::Affine3x4::fromMat4(instance.transform);
    nodeWorldMatrices(*model, m_scratchNodes);
    if (state)
        for (size_t n = 0; n < m_scratchNodes.size() && n < state->nodeModel.size(); ++n) m_scratchNodes[n] = state->nodeModel[n].toMat4();
    const auto first = uint32_t(m_dynamics.size());
    for (size_t n = 0; n < model->nodes.size(); ++n) {
        const int32_t meshIndex = model->nodes[n].mesh;
        if (meshIndex < 0) continue;
        const GpuMesh& mesh = model->meshes[size_t(meshIndex)];
        Dynamic d;
        d.meshRef = meshRefFor(instance.model, uint32_t(meshIndex));
        d.submeshCount = mesh.submeshCount;
        d.layer = instance.layer;
        d.castsShadows = instance.castsShadows;
        d.firstMaterial = uint32_t(m_dynamicMaterials.size());
        for (uint32_t s = 0; s < mesh.submeshCount; ++s)
            m_dynamicMaterials.push_back(materialFor(instance, *model, model->submeshes[mesh.firstSubmesh + s].materialSlot));
        d.node = uint32_t(n);
        d.animation = instance.animation;
        d.skinned = state && mesh.jointCount > 0;
        if (d.skinned) writeSkinned(d, world, *state, *model, mesh);
        else writeDynamic(d, instance.transform * m_scratchNodes[n], mesh);
        const auto index = uint32_t(m_dynamics.size());
        instance.dynamicIndices.push_back(index);
        if (d.skinned) m_skinnedDynamics.push_back(index);
        m_dynamics.push_back(d);
        m_dynamicNodeWorld.push_back(m_scratchNodes[n]);
    }
    // Its renderables were appended contiguously, so one range names them.
    if (instance.blobShadow && m_dynamics.size() > first) m_blobs.push_back({first, uint32_t(m_dynamics.size()) - first});
}

void Scene::rebuildDynamics() {
    ++m_dynamicRevision; // renderable indices change
    m_dynamics.clear();
    m_skinnedDynamics.clear();
    m_dynamicNodeWorld.clear();
    m_dynamicMaterials.clear();
    m_blobs.clear();
    m_instances.forEach([&](InstanceId id, Instance& instance) {
        instance.dynamicIndices.clear();
        if (instance.realized && !instance.isStatic) realizeDynamic(id, instance);
    });
}

void Scene::rebuildStatics() {
    FL_ZONE("Scene::rebuildStatics");
    struct Entry {
        uint32_t meshRef, submesh;
        MaterialHandle material;
        uint8_t layer;
        bool mirrored;
        bool lightmapped;
        uint8_t reflectionSlot;
        bool castsShadows;
        InstanceRows rows;
        glm::vec4 sphere;
        glm::vec4 lightmapST;
        uint32_t drawFlags;
        glm::vec3 boxMin, boxMax;
        const GpuMesh* mesh; // LOD levels and thresholds
    };
    std::vector<Entry> entries;
    m_occluders.clear();
    m_instances.forEach([&](InstanceId, Instance& instance) {
        if (!instance.realized || !instance.isStatic) return;
        const ModelAsset* model = m_resources.model(instance.model);
        if (!model) return;
        nodeWorldMatrices(*model, m_scratchNodes);
        uint32_t meshNode = 0; // lightmap placements are per mesh node, in node order
        for (size_t n = 0; n < model->nodes.size(); ++n) {
            const int32_t meshIndex = model->nodes[n].mesh;
            if (meshIndex < 0) continue;
            const LightmapPlacement placement = meshNode < instance.lightmaps.size() ? instance.lightmaps[meshNode] : LightmapPlacement{};
            ++meshNode;
            const GpuMesh& mesh = model->meshes[size_t(meshIndex)];
            const glm::mat4 objectToWorld = instance.transform * m_scratchNodes[n];
            const InstanceRows rows = assets::foldDequantization(core::Affine3x4::fromMat4(objectToWorld), mesh.dequant);
            const glm::vec4 sphere = worldSphere(objectToWorld, mesh.sphere);
            const bool mirrored = glm::determinant(glm::mat3(objectToWorld)) < 0.0f;
            const bool lightmapped = placement.page >= 0;
            const uint32_t ref = meshRefFor(instance.model, uint32_t(meshIndex));
            const uint32_t flags = packDrawFlags(0, lightmapped ? uint32_t(placement.page) : 0u, instance.reflectionSlot);
            // The world AABB of the mesh's box (culling, M5 plan D2); occluder instances also add the box itself (D4).
            glm::vec3 corners[8], boxMin(FLT_MAX), boxMax(-FLT_MAX);
            for (int k = 0; k < 8; ++k) {
                const glm::vec3 local((k & 1) ? mesh.boundsMax.x : mesh.boundsMin.x, (k & 2) ? mesh.boundsMax.y : mesh.boundsMin.y,
                                      (k & 4) ? mesh.boundsMax.z : mesh.boundsMin.z);
                corners[k] = glm::vec3(objectToWorld * glm::vec4(local, 1.0f));
                boxMin = glm::min(boxMin, corners[k]);
                boxMax = glm::max(boxMax, corners[k]);
            }
            if (instance.occluder) {
                Occluder o;
                std::copy(corners, corners + 8, o.corners);
                o.sphere = glm::vec4((boxMin + boxMax) * 0.5f, glm::length(boxMax - boxMin) * 0.5f);
                m_occluders.push_back(o);
            }
            for (uint32_t s = 0; s < mesh.submeshCount; ++s)
                entries.push_back({ref, s, materialFor(instance, *model, model->submeshes[mesh.firstSubmesh + s].materialSlot),
                                   instance.layer, mirrored, lightmapped, instance.reflectionSlot, instance.castsShadows, rows, sphere,
                                   placement.scaleOffset, flags, boxMin, boxMax, &mesh});
        }
    });
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return std::tie(a.meshRef, a.submesh, a.material.bits, a.layer, a.mirrored, a.lightmapped, a.reflectionSlot, a.castsShadows) <
               std::tie(b.meshRef, b.submesh, b.material.bits, b.layer, b.mirrored, b.lightmapped, b.reflectionSlot, b.castsShadows);
    });
    m_groups.clear();
    m_staticRows.clear();
    m_staticSpheres.clear();
    m_staticRowGroup.clear();
    m_staticLightmapST.clear();
    m_staticDrawFlags.clear();
    m_staticRows.reserve(entries.size());
    m_staticSpheres.reserve(entries.size());
    m_staticRowGroup.reserve(entries.size());
    m_staticLightmapST.reserve(entries.size());
    m_staticDrawFlags.reserve(entries.size());
    for (const Entry& e : entries) {
        StaticGroup* g = m_groups.empty() ? nullptr : &m_groups.back();
        if (!g || g->meshRef != e.meshRef || g->submesh != e.submesh || g->material != e.material || g->layer != e.layer ||
            g->mirrored != e.mirrored || g->lightmapped != e.lightmapped || g->reflectionSlot != e.reflectionSlot ||
            g->castsShadows != e.castsShadows) {
            StaticGroup group{e.meshRef, e.submesh, e.material, e.layer, e.mirrored, e.lightmapped, e.reflectionSlot, e.castsShadows,
                              uint32_t(m_staticRows.size()), 0};
            group.lodCount = e.mesh->lodCount;
            std::copy(std::begin(e.mesh->lodScreenSize), std::end(e.mesh->lodScreenSize), std::begin(group.lodScreenSize));
            m_groups.push_back(group);
            g = &m_groups.back();
        }
        m_staticRows.push_back(e.rows);
        m_staticSpheres.push_back(e.sphere);
        m_staticRowGroup.push_back(uint32_t(m_groups.size() - 1));
        m_staticLightmapST.push_back(e.lightmapST);
        m_staticDrawFlags.push_back(e.drawFlags);
        ++g->rowCount;
    }

    // Culling bounds (M5 plan D2): SoA spheres padded to a multiple of 4 with entries that are never visible, and boxes.
    const size_t rows = entries.size(), padded = (rows + 3) / 4 * 4;
    m_rowX.assign(padded, 0.0f), m_rowY.assign(padded, 0.0f), m_rowZ.assign(padded, 0.0f), m_rowR.assign(padded, -FLT_MAX);
    m_rowBoxMin.assign(padded, glm::vec3(0.0f)), m_rowBoxMax.assign(padded, glm::vec3(0.0f));
    for (size_t i = 0; i < rows; ++i) {
        const glm::vec4& s = entries[i].sphere;
        m_rowX[i] = s.x, m_rowY[i] = s.y, m_rowZ[i] = s.z, m_rowR[i] = s.w;
        m_rowBoxMin[i] = entries[i].boxMin;
        m_rowBoxMax[i] = entries[i].boxMax;
    }
    ++m_staticRevision;
}

SphereArrays Scene::staticBounds() const {
    SphereArrays a;
    a.x = m_rowX.data(), a.y = m_rowY.data(), a.z = m_rowZ.data(), a.r = m_rowR.data();
    a.boxMin = m_rowBoxMin.data(), a.boxMax = m_rowBoxMax.data();
    a.count = uint32_t(m_staticRows.size());
    return a;
}


// --- Animation (M6 plan D6–D8) ---

void Scene::bindAnimation(InstanceId id, Instance& instance, const ModelAsset& model) {
    uint32_t index;
    if (!m_freeAnimations.empty()) {
        index = m_freeAnimations.back();
        m_freeAnimations.pop_back();
    } else {
        index = uint32_t(m_animations.size());
        m_animations.emplace_back();
    }
    auto state = std::make_unique<AnimState>();
    state->owner = id;
    state->pose.bind(model.parents, model.rest);
    state->nodeModel.resize(model.nodes.size());
    state->pose.evaluate(state->nodeModel); // the rest pose until something plays
    m_animations[index] = std::move(state);
    instance.animation = index;
    m_liveAnimations.push_back(index);
}

void Scene::releaseAnimation(Instance& instance) {
    if (instance.animation == ~0u) return;
    m_animations[instance.animation].reset();
    m_freeAnimations.push_back(instance.animation);
    m_liveAnimations.erase(std::remove(m_liveAnimations.begin(), m_liveAnimations.end(), instance.animation), m_liveAnimations.end());
    instance.animation = ~0u;
}

// A skinned renderable: no object transform of its own (the palette carries it), and bounds that contain every vertex
// in the current pose (M6 plan D10), so culling stays conservative.
void Scene::writeSkinned(Dynamic& d, const core::Affine3x4& world, const AnimState& state, const ModelAsset& model,
                         const GpuMesh& mesh) const {
    glm::vec3 lo(world.rows[0].w, world.rows[1].w, world.rows[2].w), hi = lo;
    if (size_t(mesh.firstJoint) + mesh.jointCount <= model.joints.size())
        anim::skinBounds(world, state.nodeModel, std::span(model.joints).subspan(mesh.firstJoint, mesh.jointCount), lo, hi);
    d.rows = assets::foldDequantization(world, mesh.dequant);
    d.mirrored = glm::determinant(glm::mat3(world.toMat4())) < 0.0f;
    const glm::vec3 center = (lo + hi) * 0.5f, half = (hi - lo) * 0.5f;
    d.sphere = glm::vec4(center, glm::length(half));
    d.boxCenter = center;
    d.boxHalfAxes[0] = glm::vec3(half.x, 0.0f, 0.0f);
    d.boxHalfAxes[1] = glm::vec3(0.0f, half.y, 0.0f);
    d.boxHalfAxes[2] = glm::vec3(0.0f, 0.0f, half.z);
    d.boxMin = lo;
    d.boxMax = hi;
    d.lodCount = mesh.lodCount;
    std::copy(std::begin(mesh.lodScreenSize), std::end(mesh.lodScreenSize), std::begin(d.lodScreenSize));
}

anim::Instance* Scene::animation(InstanceId id) {
    const Instance* instance = m_instances.get(id);
    if (!instance || instance->animation == ~0u) return nullptr;
    return &m_animations[instance->animation]->pose;
}

anim::ClipRef Scene::clip(InstanceId id, AnimHandle library, std::string_view name) {
    const Instance* instance = m_instances.get(id);
    if (!instance || !instance->realized) return {};
    const ModelAsset* model = m_resources.model(instance->model);
    const AnimAsset* lib = m_resources.animation(library);
    if (!model || model->state != LoadState::Ready || !lib || lib->state != LoadState::Ready) return {};
    const uint64_t key = uint64_t(library.bits) << 32 | instance->model.bits;
    auto it = m_bindings.find(key);
    if (it == m_bindings.end()) {
        auto binding = std::make_unique<anim::LibraryBinding>(anim::bindLibrary(lib->view, model->nameHashes));
        if (binding->unboundTracks)
            FL_LOG_WARN("Clip library %s: %u track(s) target nodes that %s doesn't have", lib->path.c_str(), binding->unboundTracks,
                        model->path.c_str());
        it = m_bindings.emplace(key, std::move(binding)).first;
    }
    const int32_t index = anim::findClip(lib->view, name);
    if (index < 0) {
        FL_LOG_WARN("Clip library %s has no clip '%.*s'", lib->path.c_str(), int(name.size()), name.data());
        return {};
    }
    return {&lib->view, it->second.get(), uint32_t(index)};
}

int32_t Scene::findNode(InstanceId id, std::string_view name) const {
    const Instance* instance = m_instances.get(id);
    const ModelAsset* model = instance ? m_resources.model(instance->model) : nullptr;
    if (!model || model->state != LoadState::Ready) return -1;
    for (size_t i = 0; i < model->nodes.size(); ++i)
        if (name == std::string_view(model->nodes[i].name)) return int32_t(i);
    return -1;
}

bool Scene::nodeWorld(InstanceId id, uint32_t node, glm::mat4& out) const {
    const Instance* instance = m_instances.get(id);
    const ModelAsset* model = instance ? m_resources.model(instance->model) : nullptr;
    if (!model || model->state != LoadState::Ready || node >= model->nodes.size()) return false;
    if (instance->animation != ~0u) {
        out = instance->transform * m_animations[instance->animation]->nodeModel[node].toMat4();
        return true;
    }
    glm::mat4 m = localMatrix(model->nodes[node]); // walk up the parents: no allocation
    for (int32_t p = model->nodes[node].parent; p >= 0; p = model->nodes[size_t(p)].parent) m = localMatrix(model->nodes[size_t(p)]) * m;
    out = instance->transform * m;
    return true;
}

void Scene::animate(platform::JobSystem& jobs, float dt) {
    FL_ZONE("Scene::animate");
    m_animateDt = dt;
    if (!m_liveAnimations.empty()) jobs.parallelFor(uint32_t(m_liveAnimations.size()), 1, animateRange, this);
}

// One job per animated instance (M6 plan D8): no shared state, no allocation.
void Scene::animateRange(void* context, uint32_t begin, uint32_t end, uint32_t) {
    Scene& self = *static_cast<Scene*>(context);
    for (uint32_t i = begin; i < end; ++i) {
        AnimState& state = *self.m_animations[self.m_liveAnimations[i]];
        const Instance* instance = self.m_instances.get(state.owner);
        if (!instance || !instance->realized) continue;
        const ModelAsset* model = self.m_resources.model(instance->model);
        if (!model || model->state != LoadState::Ready) continue;
        state.pose.advance(self.m_animateDt);
        state.pose.evaluate(state.nodeModel);
        const core::Affine3x4 world = core::Affine3x4::fromMat4(instance->transform);
        for (uint32_t index : instance->dynamicIndices) {
            Dynamic& d = self.m_dynamics[index];
            const GpuMesh& mesh = model->meshes[self.m_meshRefs[d.meshRef].mesh];
            if (d.skinned) self.writeSkinned(d, world, state, *model, mesh);
            else if (d.node < state.nodeModel.size()) self.writeDynamic(d, instance->transform * state.nodeModel[d.node].toMat4(), mesh);
        }
    }
}

void Scene::writePalettes(platform::JobSystem& jobs, std::span<const uint8_t> dynamicMasks, rhi::TransientRing& ring) {
    FL_ZONE("Scene::writePalettes");
    m_paletteMasks = dynamicMasks;
    m_paletteRing = &ring;
    m_palettesWritten.store(0, std::memory_order_relaxed);
    m_palettesDropped.store(0, std::memory_order_relaxed);
    if (!m_skinnedDynamics.empty()) jobs.parallelFor(uint32_t(m_skinnedDynamics.size()), 4, paletteRange, this);
}

// One palette per skinned renderable that some view sees (M6 plan D8): written once, straight into the ring (write-
// combined memory on Vulkan: never read back).
void Scene::paletteRange(void* context, uint32_t begin, uint32_t end, uint32_t) {
    Scene& self = *static_cast<Scene*>(context);
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t index = self.m_skinnedDynamics[i];
        Dynamic& d = self.m_dynamics[index];
        d.paletteOffset = kNoPalette;
        if (index >= self.m_paletteMasks.size() || self.m_paletteMasks[index] == 0 || d.animation == ~0u) continue;
        const AnimState* state = self.m_animations[d.animation].get();
        const Instance* instance = state ? self.m_instances.get(state->owner) : nullptr;
        const ModelAsset* model = instance ? self.m_resources.model(instance->model) : nullptr;
        if (!model || model->state != LoadState::Ready) continue;
        const GpuMesh& mesh = model->meshes[self.m_meshRefs[d.meshRef].mesh];
        if (mesh.jointCount == 0 || mesh.jointCount > kMaxPaletteJoints || size_t(mesh.firstJoint) + mesh.jointCount > model->joints.size())
            continue;
        const rhi::TransientRing::Allocation a = self.m_paletteRing->allocate(kPaletteBytes);
        if (!a.cpu) { // the ring is full this frame: the renderable isn't drawn (counted and logged by the renderer)
            self.m_palettesDropped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        anim::writePalette(core::Affine3x4::fromMat4(instance->transform), state->nodeModel,
                           std::span(model->joints).subspan(mesh.firstJoint, mesh.jointCount), mesh.dequant,
                           static_cast<core::Affine3x4*>(a.cpu));
        d.paletteOffset = a.gpu.offset;
        self.m_palettesWritten.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace renderer
