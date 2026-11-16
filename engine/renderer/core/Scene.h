#pragma once

#include "anim/Instance.h"
#include "core/Handle.h"
#include "renderer/core/Lights.h"
#include "renderer/core/ShaderTypes.h"
#include "renderer/resources/ResourceManager.h"
#include "renderer/visibility/Frustum.h"

#include <atomic>
#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace platform {
class JobSystem;
}
namespace rhi {
class TransientRing;
}

namespace renderer {

class MaterialSystem;

using InstanceId = core::Handle<struct SceneInstanceTag>;
// M9 plan D8: effect lights come and go with every explosion, so the 1,024 slots trade index bits for generations.
using LightId = core::Handle<struct SceneLightTag, 10>;

// Where one mesh node of a static instance lives in the lightmap (M3 plan D3): UV1 · scale + offset in layer `page`.
struct LightmapPlacement {
    int32_t page = -1; // −1: not lightmapped
    glm::vec4 scaleOffset{0.0f};
};

struct InstanceDesc {
    ModelHandle model;
    glm::mat4 transform{1.0f};
    bool isStatic = false;
    uint8_t layer = 1; // sort-key layer (0 = big occluders)
    std::span<const MaterialHandle> materialOverrides; // per model material slot; empty = the model's materials
    std::span<const LightmapPlacement> lightmaps;      // static only: one per mesh node, in node order; empty = none
    uint8_t reflectionSlot = 0;                        // static only: 0 = the sky, 1–7 = a local reflection probe
    bool castsShadows = true;                          // into shadow maps, if its materials cast (M4 plan D10)
    bool blobShadow = false;                           // dynamic only: a grounding blob under it (M4 plan D9)
    // Static only (M5 plan D4): each mesh node's bounding box hides what is behind it from the camera. Only for solid meshes
    // that fill their boxes (walls, buildings, blocks); a hollow or open mesh would cull things you can see.
    bool occluder = false;
    // Dynamic only (M6 plan D6, D8): an anim::Instance drives its nodes (clips, layers, node overrides). Models with a skin
    // are always animated (at least their rest pose); see Scene::animation().
    bool animated = false;
};

// Renderer-side storage behind world::RenderWorld (ARCHITECTURE.md §5.4): flat arrays that the queue-build jobs read.
// A model instance turns into one renderable per mesh node once its model is Ready. Static renderables are kept in
// instance groups (mesh, submesh, material, layer) whose rows are stored contiguously (§4.15); dynamic ones are one
// renderable each. Arrays are rebuilt only when content changes; per-frame transform updates of dynamic instances
// don't allocate.
class Scene {
public:
    struct MeshRef {
        ModelHandle model;
        uint32_t mesh = 0;
    };
    struct Dynamic {
        uint32_t meshRef = 0;
        uint32_t firstMaterial = 0; // into dynamicMaterials(), one per submesh
        uint32_t submeshCount = 0;
        uint8_t layer = 1;
        bool mirrored = false;      // negative determinant: front faces turn clockwise
        bool castsShadows = true;   // the instance's flag (its materials decide per submesh)
        InstanceRows rows;          // object → world with dequantization folded in
        glm::vec4 sphere{0.0f};     // world space
        glm::vec3 boxCenter{0.0f};    // world-space oriented box (blob shadows): center and the three half-axis vectors
        glm::vec3 boxHalfAxes[3]{};
        glm::vec3 boxMin{0.0f}, boxMax{0.0f}; // its world AABB (culling, M5 plan D2)
        uint32_t lodCount = 1;        // the mesh's levels and thresholds (M5 plan D5)
        float lodScreenSize[assets::kMaxLods] = {};
        // Animation (M6 plan D8): the model node it draws, its instance's animation state, and for skinned meshes this frame's
        // bone palette in the uniform ring (kNoPalette when no view sees it or the ring is full).
        uint32_t node = 0;
        uint32_t animation = ~0u;
        uint32_t paletteOffset = ~0u;
        bool skinned = false;
    };
    struct StaticGroup {
        uint32_t meshRef = 0;
        uint32_t submesh = 0;
        MaterialHandle material;
        uint8_t layer = 1;
        bool mirrored = false;  // mirrored rows need the clockwise-front-face pipeline, so they group separately
        bool lightmapped = false; // lightmapped rows use the LIGHTMAP variants
        uint8_t reflectionSlot = 0; // bound per draw (set 3), so groups split by it
        bool castsShadows = true;   // shadow passes take whole groups (High), so groups split by it
        uint32_t firstRow = 0; // into staticRows() / staticSpheres()
        uint32_t rowCount = 0;
        uint32_t lodCount = 1;  // the mesh's levels and thresholds (M5 plan D5)
        float lodScreenSize[assets::kMaxLods] = {};
    };
    // An occluder box (M5 plan D4): world corners k = (x: bit 0, y: bit 1, z: bit 2) of a mesh node's box, and a sphere
    // around it for frustum culling.
    struct Occluder {
        glm::vec3 corners[8];
        glm::vec4 sphere{0.0f};
    };
    // A dynamic model with a blob shadow (M4 plan D9): its renderables are dynamics()[first, first + count).
    struct Blob {
        uint32_t firstDynamic = 0;
        uint32_t dynamicCount = 0;
    };

    Scene(ResourceManager& resources, MaterialSystem& materials);
    ~Scene();
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    InstanceId add(const InstanceDesc& desc); // takes references on the model and the override materials
    void setTransform(InstanceId id, const glm::mat4& transform);
    void remove(InstanceId id);

    // Local lights (M2 plan D4); the renderer reads them every frame, so updates are plain stores.
    LightId addLight(const LightDesc& desc);
    void updateLight(LightId id, const LightDesc& desc);
    void removeLight(LightId id);
    const LightDesc* light(LightId id) const { return m_lights.get(id); }
    template <typename Fn>
    void forEachLight(Fn&& fn) { m_lights.forEach(fn); }
    uint32_t lightCount() const { return m_lights.liveCount(); }

    void clear(); // instances and lights

    // Main thread, once per frame after ResourceManager::finalizeUploads(): realizes instances whose model became
    // Ready and rebuilds the static groups if anything static changed.
    void update();

    std::span<const MeshRef> meshRefs() const { return m_meshRefs; }
    std::span<const Dynamic> dynamics() const { return m_dynamics; }
    std::span<const MaterialHandle> dynamicMaterials() const { return m_dynamicMaterials; }
    std::span<const StaticGroup> staticGroups() const { return m_groups; }
    std::span<const InstanceRows> staticRows() const { return m_staticRows; }
    std::span<const glm::vec4> staticSpheres() const { return m_staticSpheres; }
    std::span<const uint32_t> staticRowGroups() const { return m_staticRowGroup; } // row → its group
    std::span<const glm::vec4> staticLightmapST() const { return m_staticLightmapST; } // row → UV1 scale/offset
    std::span<const uint32_t> staticDrawFlags() const { return m_staticDrawFlags; }     // row → page and slot bits (packDrawFlags)
    std::span<const Blob> blobs() const { return m_blobs; }
    // Culling bounds of the static rows (M5 plan D2): SoA spheres padded to a multiple of 4, plus world boxes.
    SphereArrays staticBounds() const;
    std::span<const Occluder> occluders() const { return m_occluders; }
    // Bumped when rows or dynamic renderables are re-indexed (statics rebuilt, a dynamic instance removed), so per-row and
    // per-renderable state elsewhere (LOD) knows to start over.
    uint32_t staticRevision() const { return m_staticRevision; }
    uint32_t dynamicRevision() const { return m_dynamicRevision; }
    uint32_t instanceCount() const { return m_instances.liveCount(); }
    uint32_t unrealizedCount() const { return uint32_t(m_unrealized.size()); }

    // Animation (M6 plan D6–D8). An animated instance gets its anim::Instance when its model becomes Ready: until then
    // animation() returns nullptr. The pointer stays valid until the instance is removed (or the scene cleared).
    static constexpr uint32_t kNoPalette = ~0u;
    anim::Instance* animation(InstanceId id);
    // A clip of `library` bound to the instance's model (the binding is built once per library and model, which
    // allocates: do it at setup). Empty until both are Ready, or when the library has no such clip.
    anim::ClipRef clip(InstanceId id, AnimHandle library, std::string_view name);
    int32_t findNode(InstanceId id, std::string_view name) const; // −1 when unknown or not loaded
    // The node's world transform as of the last animate() (or the rest pose for instances without animation).
    bool nodeWorld(InstanceId id, uint32_t node, glm::mat4& out) const;
    // Main thread, once per frame before visibility: advances every animated instance by dt on the job workers, then
    // updates its renderables (rigid nodes: rows and bounds; skinned meshes: conservative bounds from the joints).
    void animate(platform::JobSystem& jobs, float dt);
    // After visibility: the bone palette of every skinned renderable some view sees, into the frame's uniform ring.
    void writePalettes(platform::JobSystem& jobs, std::span<const uint8_t> dynamicMasks, rhi::TransientRing& ring);
    uint32_t animatedCount() const { return uint32_t(m_liveAnimations.size()); }
    uint32_t skinnedCount() const { return uint32_t(m_skinnedDynamics.size()); }
    uint32_t palettesWritten() const { return m_palettesWritten.load(std::memory_order_relaxed); }
    uint32_t palettesDropped() const { return m_palettesDropped.load(std::memory_order_relaxed); }
    std::span<const uint32_t> skinnedDynamics() const { return m_skinnedDynamics; }

private:
    struct Instance {
        ModelHandle model;
        glm::mat4 transform{1.0f};
        bool isStatic = false;
        uint8_t layer = 1;
        std::vector<MaterialHandle> overrides;
        std::vector<LightmapPlacement> lightmaps;
        uint8_t reflectionSlot = 0;
        bool castsShadows = true;
        bool blobShadow = false;
        bool occluder = false;
        bool animated = false;
        bool realized = false;
        uint32_t animation = ~0u;             // into m_animations
        std::vector<uint32_t> dynamicIndices; // its renderables in m_dynamics
    };
    // One animated instance's pose (M6 plan D6): heap-allocated so anim::Instance pointers given out stay valid.
    struct AnimState {
        InstanceId owner;
        anim::Instance pose;
        std::vector<core::Affine3x4> nodeModel; // node → model space, as of the last evaluation
    };

    void bindAnimation(InstanceId id, Instance& instance, const ModelAsset& model);
    void releaseAnimation(Instance& instance);
    void writeSkinned(Dynamic& d, const core::Affine3x4& world, const AnimState& state, const ModelAsset& model, const GpuMesh& mesh) const;
    static void animateRange(void* context, uint32_t begin, uint32_t end, uint32_t worker);
    static void paletteRange(void* context, uint32_t begin, uint32_t end, uint32_t worker);

    void realizeDynamic(InstanceId id, Instance& instance);
    void rebuildDynamics();
    void rebuildStatics();
    void nodeWorldMatrices(const ModelAsset& model, std::vector<glm::mat4>& out) const;
    MaterialHandle materialFor(const Instance& instance, const ModelAsset& model, uint32_t slot) const;
    uint32_t meshRefFor(ModelHandle model, uint32_t mesh);
    void writeDynamic(Dynamic& d, const glm::mat4& objectToWorld, const GpuMesh& mesh) const;

    ResourceManager& m_resources;
    MaterialSystem& m_materials;
    core::HandlePool<Instance, SceneInstanceTag> m_instances{65536};
    std::vector<InstanceId> m_unrealized;
    core::HandlePool<LightDesc, SceneLightTag, LightId::kMaxGeneration, 10> m_lights{1024};
    bool m_staticDirty = false;

    std::vector<MeshRef> m_meshRefs;
    std::unordered_map<uint64_t, uint32_t> m_meshRefIndex;
    std::vector<Dynamic> m_dynamics;
    std::vector<glm::mat4> m_dynamicNodeWorld; // per dynamic renderable: node → model space
    std::vector<MaterialHandle> m_dynamicMaterials;
    std::vector<StaticGroup> m_groups;
    std::vector<InstanceRows> m_staticRows;
    std::vector<glm::vec4> m_staticSpheres;
    std::vector<uint32_t> m_staticRowGroup;
    std::vector<glm::vec4> m_staticLightmapST;
    std::vector<uint32_t> m_staticDrawFlags;
    std::vector<Blob> m_blobs;
    std::vector<float> m_rowX, m_rowY, m_rowZ, m_rowR; // SoA spheres of the static rows, padded to a multiple of 4
    std::vector<glm::vec3> m_rowBoxMin, m_rowBoxMax;   // world AABBs of the static rows
    std::vector<Occluder> m_occluders;
    uint32_t m_staticRevision = 0;
    uint32_t m_dynamicRevision = 0;
    std::vector<glm::mat4> m_scratchNodes;

    std::vector<std::unique_ptr<AnimState>> m_animations;
    std::vector<uint32_t> m_freeAnimations;
    std::vector<uint32_t> m_liveAnimations;  // indices into m_animations, what animate() walks
    std::vector<uint32_t> m_skinnedDynamics; // dynamic renderables with a skin, what writePalettes() walks
    std::unordered_map<uint64_t, std::unique_ptr<anim::LibraryBinding>> m_bindings; // (library, model) → binding
    // Per-frame job inputs.
    float m_animateDt = 0.0f;
    std::span<const uint8_t> m_paletteMasks;
    rhi::TransientRing* m_paletteRing = nullptr;
    std::atomic<uint32_t> m_palettesWritten{0};
    std::atomic<uint32_t> m_palettesDropped{0};
};

} // namespace renderer
