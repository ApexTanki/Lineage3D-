#pragma once

#include "assets/Material.h"
#include "core/Handle.h"
#include "core/Result.h"
#include "renderer/core/SortKey.h"
#include "renderer/resources/ResourceManager.h"
#include "rhi/Device.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace renderer {

class ShaderLibrary;

// The engine's fixed vertex streams (ARCHITECTURE.md §4.7): 0 = positions, 1 = surface, 2 = per-instance data. Builds
// a layout that declares exactly the inputs a shader variant reads; false if one has no fixed place.
bool fixedVertexLayout(const shaderformat::Reflection& reflection, rhi::VertexLayout& out);

// The lit template's OBJECT axis (M3 plan D7): where an object's transform, light list and indirect light come from.
enum class ObjectVariant : uint8_t {
    Dynamic,          // per-draw block; probe SH (or the ambient as SH)
    Instanced,        // static instance group without lightmaps; hemispherical ambient
    Lightmap,         // a lightmapped static drawn on its own
    LightmapInstanced, // a lightmapped static instance group
    Skinned            // like Dynamic, with vertices blended by the bone palette in set 3 (M6 plan D9)
};
inline constexpr uint32_t kObjectVariants = 5;

// Materials: template + feature permutation + parameters + textures; one draw per submesh/material
// (ARCHITECTURE.md §5.4, M2 plan D1–D3, D6, D13; M3 plan D7, D11). The "lit" template takes up to five maps (albedo,
// normal, spec/gloss, emission, detail) and five blend modes. Pipelines live in a dense table keyed by (permutation,
// blend, double-sidedness, mirroring), shared by every material with the same state; a material resolves its eight
// slots (object variant × mirrored) when it is acquired. A material whose file failed to load, names an unknown
// template or needs a permutation the package lacks renders as the loud magenta error material, with a logged error.
class MaterialSystem {
public:
    struct Desc {
        rhi::Device* device = nullptr;
        ShaderLibrary* shaders = nullptr;
        ResourceManager* resources = nullptr;
        rhi::ResourceSetLayoutHandle frameLayout;      // set 0
        rhi::ResourceSetLayoutHandle reflectionLayout; // set 3 (the per-draw reflection cube)
        rhi::ResourceSetLayoutHandle skinnedLayout;    // set 3 of skinned pipelines: bone palette + reflection cube
        rhi::ResourceSetLayoutHandle passLayout;       // set 1: the forward passes' inputs (depth, SSAO, grab; M8 plan D9, D10)
    };

    // Draw-time view of a material; plain data, safe to read from job threads while nothing mutates the system.
    struct DrawState {
        rhi::ResourceSetHandle set;         // set 2
        uint16_t pipelineSlot = 0;          // see pipeline(); also the sort-key pipeline field
        uint16_t sortId = 0;                // dense id for sort keys
        QueuePass pass = QueuePass::Opaque; // Opaque, AlphaTest or Transparent
        // LOD cross-fade (M5 plan D7): the pipeline of a fading draw. Opaque materials use their alpha-test twin (the
        // ALPHA_TEST permutation with the cutoff uniform at −1, so only the dither discards); alpha-tested ones their own
        // pipeline; blended ones never fade (same as pipelineSlot).
        uint16_t fadePipelineSlot = 0;
        QueuePass fadePass = QueuePass::Opaque;
        // Refraction off (M8 plan D10): a refractive material's alpha-blended twin among the transparents; for every other
        // material the same as pipelineSlot and pass.
        uint16_t blendTwinSlot = 0;
        QueuePass blendTwinPass = QueuePass::Opaque;
    };

    core::Result<void> init(const Desc& desc);
    void shutdown();

    // Ref-counted and de-duplicated by path; textures load asynchronously (per-slot placeholders until Ready).
    MaterialHandle acquire(const std::string& path, const assets::MaterialFile& file);
    // Reads and parses a data-relative .fmat synchronously (loading time only), then acquire().
    MaterialHandle load(const std::string& path);
    void addRef(MaterialHandle material);
    void release(MaterialHandle material);

    void update(); // once per frame: swaps placeholders for maps that finished loading, and hot-reloaded maps (M9 plan D9)
    // Hot reload (dev; M9 plan D9): reads a loaded material's file again into the same handle (maps, parameters, blend
    // mode, flags; new pipeline slots, uniforms and set, the old ones released deferred). False if the material isn't
    // loaded or the file doesn't parse (it keeps its old state).
    bool reload(const std::string& path);

    // (Re)creates every pipeline for a target layout (startup, MSAA or format change; never per frame).
    core::Result<void> setTarget(const rhi::AttachmentLayout& target);
    void setAnisotropy(float anisotropy); // settings change: recreates the sampler and every material set

    // `mirrored`: the object transform has a negative determinant, which turns front faces clockwise.
    bool drawState(MaterialHandle material, ObjectVariant variant, bool mirrored, DrawState& out) const;

    // Shadow casting (M4 plan D7, D10): how a material draws into shadow maps. Plain data, safe on job threads.
    struct CasterState {
        rhi::ResourceSetHandle set; // set 2: alpha-tested casters sample the albedo
        bool casts = false;         // opaque or alpha-tested with castShadows (the error material casts like an opaque one)
        bool alphaTest = false;
        uint16_t sortId = 0;
    };
    bool casterState(MaterialHandle material, CasterState& out) const;
    rhi::ResourceSetLayoutHandle materialLayout() const { return m_materialLayout; } // set 2, for the shadow pipelines
    rhi::PipelineHandle pipeline(uint16_t slot) const { return slot < m_pipelines.size() ? m_pipelines[slot].pipeline : rhi::PipelineHandle{}; }
    // Depth prepass (M8 plan D9): the depth-only twin of an opaque pipeline slot (the prepass template: the same object
    // variant, culling and winding; its FADE variant for the LOD fade twin of an opaque material, so a fading object stays
    // in the depth with the twin's dither), or none for any other blend mode.
    rhi::PipelineHandle prepassPipeline(uint16_t slot) const;

    // Live editing (dev panel, M2 plan D13): applies the parameter block of `params` (colors, gloss, emission,
    // normal strength, alpha cutoff). Recreates the material's uniform buffer and set; frames in flight keep the old ones.
    void setParams(MaterialHandle material, const assets::MaterialFile& params);
    const assets::MaterialFile* file(MaterialHandle material) const;
    const std::string* path(MaterialHandle material) const;
    // The blend mode draws use: the error material is opaque whatever its file asked for.
    assets::BlendMode blendMode(MaterialHandle material) const;
    template <typename Fn>
    void forEach(Fn&& fn) { m_materials.forEach([&](MaterialHandle h, Material&) { fn(h); }); }

    uint32_t materialCount() const { return m_materials.liveCount(); }
    uint32_t pipelineCount() const { return uint32_t(m_pipelines.size()); }
    uint32_t waitingCount() const { return uint32_t(m_waiting.size()); }

    static constexpr uint32_t kMaxPipelines = 4096; // the sort key's 12-bit pipeline field

private:
    enum Slot : uint32_t { kAlbedo, kNormal, kSpecGloss, kEmission, kDetail, kSlotCount };
    enum class MapState : uint8_t { None, Loading, Ready, Failed };

    struct Material {
        std::string path;
        uint32_t refs = 0;
        assets::MaterialFile file;
        TextureAsset maps[kSlotCount];
        MapState mapState[kSlotCount] = {};
        bool error = false;           // magenta error material
        uint16_t pipelineSlots[kObjectVariants * 2] = {}; // [variant * 2 + mirrored]
        uint16_t fadeSlots[kObjectVariants * 2] = {};     // the same for fading LOD draws (M5 plan D7)
        uint16_t blendTwinSlots[kObjectVariants * 2] = {}; // refractive: the alpha-blended twin (refraction off; M8 plan D10)
        rhi::BufferHandle uniforms;
        rhi::ResourceSetHandle set;
        mutable rhi::TextureHandle boundTextures[kSlotCount]; // what the set was made with (a hot-reloaded map changes it)
    };
    struct PipelineEntry {
        uint64_t shaderKey = 0;
        assets::BlendMode blend = assets::BlendMode::Opaque;
        bool doubleSided = false;
        bool mirrored = false;
        ObjectVariant variant = ObjectVariant::Dynamic;
        std::string name; // debug name
        rhi::PipelineHandle pipeline;
    };
    struct AxisBits {
        uint8_t normalMap = 0, specGlossMap = 0, emissionMap = 0, alphaTest = 0, detailLayer = 0, object = 0;
    };

    static MaterialHandle acquireCallback(void* user, const std::string& path, const assets::MaterialFile& file);
    static void releaseCallback(void* user, MaterialHandle material);
    uint64_t shaderKey(const Material& m, ObjectVariant variant, bool fadeTwin = false) const;
    uint16_t pipelineSlot(const Material& m, ObjectVariant variant, bool mirrored, bool fadeTwin = false, bool blendTwin = false);
    rhi::PipelineHandle createPipeline(const PipelineEntry& entry);
    rhi::BufferHandle createUniforms(const Material& m) const;
    rhi::ResourceSetHandle createSet(const Material& m) const;
    // Everything acquire() makes for a material but its pool slot and set: maps, pipeline slots, uniforms. False if the
    // uniforms couldn't be made (its maps are taken all the same and must be released).
    bool build(Material& m, const std::string& path, const assets::MaterialFile& file);
    rhi::TextureHandle textureFor(const Material& m, uint32_t slot) const;
    void releaseMaps(Material& m);

    rhi::Device* m_device = nullptr;
    ShaderLibrary* m_shaders = nullptr;
    ResourceManager* m_resources = nullptr;
    AxisBits m_axes;
    rhi::ResourceSetLayoutHandle m_frameLayout;
    rhi::ResourceSetLayoutHandle m_reflectionLayout;
    rhi::ResourceSetLayoutHandle m_skinnedLayout;
    rhi::ResourceSetLayoutHandle m_passLayout;
    rhi::ResourceSetLayoutHandle m_materialLayout;
    rhi::SamplerHandle m_sampler;
    float m_anisotropy = 4.0f;
    rhi::AttachmentLayout m_target;
    bool m_hasTarget = false;
    std::vector<PipelineEntry> m_pipelines;
    std::unordered_map<uint64_t, uint16_t> m_pipelineIndex;
    rhi::PipelineHandle m_prepass[kObjectVariants][2][2][2] = {}; // [variant][double-sided][mirrored][fade], rebuilt by setTarget
    core::HandlePool<Material, struct MaterialTag> m_materials{4096};
    std::unordered_map<std::string, uint32_t> m_byPath;
    std::vector<MaterialHandle> m_waiting; // materials with a map still loading
    uint32_t m_seenTextureRevision = 0;    // ResourceManager::textureRevision() the sets were last checked against
};

} // namespace renderer
