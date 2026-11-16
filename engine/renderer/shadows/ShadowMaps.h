#pragma once

#include "core/Result.h"
#include "renderer/core/ClipSpace.h"
#include "renderer/core/Lights.h"
#include "renderer/core/RenderQueue.h"
#include "renderer/core/Scene.h"
#include "renderer/core/ShaderTypes.h"
#include "renderer/shadows/Cascades.h"
#include "renderer/shadows/ShadowSettings.h"
#include "rhi/Device.h"

#include <span>
#include <vector>

namespace renderer {

class MaterialSystem;
class ResourceManager;
class ShaderLibrary;

// Dynamic shadows (ARCHITECTURE.md §4.14, §6 passes 1a/1b; M4 plan D2–D8): the cascade array and the spot atlas, the
// per-frame fit, the caster list, and the depth-only passes that render them. Receivers sample both textures through
// the frame set (set 0 b2/b3) and the FrameUniforms fields that fillFrame() writes.
class ShadowMaps {
public:
    struct Stats {
        uint32_t cascades = 0;
        uint32_t casters = 0;     // caster items this frame (submeshes, static groups or rows)
        uint32_t draws = 0;       // draw calls over all shadow passes
        uint32_t spotShadows = 0; // spot lights with an atlas tile
        uint32_t mapSize = 0;
    };

    core::Result<void> init(rhi::Device& device, ShaderLibrary& shaders, const MaterialSystem& materials, rhi::ResourceSetLayoutHandle paletteLayout,
                            rhi::ResourceSetHandle paletteSet);
    void shutdown();

    // Settings change (never per steady frame): recreates textures whose size or layer count changed, which bumps
    // revision() so the renderer rebuilds its frame set. New maps that don't fit leave the old ones and the previous shadow
    // settings in force, and that request isn't tried again until it changes (M9 plan D6).
    void configure(const ShadowSettings& settings);
    uint32_t revision() const { return m_revision; }
    rhi::TextureHandle cascadeTexture() const { return m_cascades; }
    rhi::TextureHandle spotAtlas() const { return m_spotAtlas; }
    rhi::SamplerHandle compareSampler() const { return m_sampler; }

    // Picks the spot lights that get atlas tiles (High) and allocates their pass uniforms, so a light only gets a slot
    // that will be rendered. `slots[i]` receives the slot of lights[i], or −1. Lights that held a tile last frame keep it
    // unless another one clearly outranks them (no popping between equal candidates).
    void assignSpotShadows(std::span<const LightDesc* const> lights, const glm::vec3& cameraPosition, float time, std::span<int8_t> slots);

    struct FrameInput {
        const ClipSpace* clipSpace = nullptr;
        glm::vec3 cameraPosition{0.0f};
        glm::vec3 cameraForward{0.0f, 0.0f, -1.0f};
        float fovY = 1.0f, aspect = 1.0f, zNear = 0.1f, zFar = 1000.0f;
        glm::vec3 sunDirection{0.0f, 1.0f, 0.0f}; // unit, toward the sun
        float time = 0.0f;
    };
    // Fits the cascades and the chosen spot shadows, and writes the receivers' FrameUniforms fields.
    void prepare(const FrameInput& in, FrameUniforms& fu);

    // Caster draws of every shadow view (M5 plan D8), queued by the renderer's queue-build jobs after culling: one DrawItem
    // per instanced run or per draw, with objectIndex = view (0–2 cascades, 3–6 spot tiles), lod = level, material = the
    // material set's bits (alpha-tested casters), pipeline = casterPipeline(), perDrawOrInstance = the run's rows-only
    // instance data in the vertex ring (kInstanced), a static row (kStaticRow) or a dynamic renderable (kDynamic).
    RenderQueue& casterQueue() { return m_queue; }
    static uint64_t casterKey(uint32_t view, uint32_t pipeline, uint16_t sortId, uint32_t meshRef, uint32_t submesh, uint32_t lod);
    // Casters draw without culling (single-sided planes cast from both sides, as in the bake), so winding doesn't matter.
    // Skinned casters (M6 plan D9) blend their transform from the bone palette (set 3, a dynamic offset per draw).
    static uint32_t casterPipeline(bool instanced, bool alphaTest, bool skinned = false) {
        return (skinned ? 4u : instanced ? 2u : 0u) | (alphaTest ? 1u : 0u);
    }
    void resetCasters() { m_queue.reset(); }
    void sortCasters(); // after the jobs filled the queue

    // Records the cascade passes and the spot atlas pass (before the scene pass) from the sorted caster queue. Textures
    // that weren't rendered since they were created get a clear-only pass, so receivers always sample an initialized image.
    void record(rhi::CommandList& cmd, const Scene& scene, const ResourceManager& resources, rhi::BufferHandle vertexRing,
                class GpuTimers& timers);

    const Stats& stats() const { return m_stats; }
    std::span<const CascadeFit> cascades() const { return {m_fits, m_cascadeCount}; }
    uint32_t spotCount() const { return m_spotCount; }
    const SpotShadowFit& spotFit(uint32_t k) const { return m_spots[k].fit; }
    bool staticCasters() const { return m_settings.staticCasters; }

private:
    struct Spot {
        SpotShadowFit fit;
        glm::mat4 viewProj{1.0f}; // backend clip space: the caster pass's, and the receivers' through the tile transform
        glm::vec3 position{0.0f}, direction{0.0f, -1.0f, 0.0f};
        float range = 0.0f;
        uint32_t slot = 0;
    };

    bool createTextures();
    rhi::PipelineHandle createPipeline(ShaderLibrary& shaders, const MaterialSystem& materials, uint32_t variant, bool alphaTest);
    // Draws the sorted caster items [begin, end) of one view; returns the end.
    uint32_t drawView(rhi::CommandList& cmd, const Scene& scene, const ResourceManager& resources, rhi::BufferHandle vertexRing,
                      uint32_t passOffset, uint32_t view, uint32_t begin);

    rhi::Device* m_device = nullptr;
    ShadowSettings m_settings;
    ShadowSettings m_refused; // the last request whose maps didn't fit (m_hasRefused): not tried again until it changes
    bool m_hasRefused = false;
    uint32_t m_mapSize = 0;
    uint32_t m_layers = 0;
    bool m_spotEnabled = false;
    uint32_t m_revision = 0;
    rhi::TextureHandle m_cascades; // D16 array, one layer per cascade (≥ 1)
    rhi::TextureHandle m_spotAtlas; // D16, 2 × 2 tiles (16² when spot shadows are off)
    bool m_cascadesNeedClear = true, m_spotNeedsClear = true;
    rhi::SamplerHandle m_sampler;
    rhi::ResourceSetLayoutHandle m_passLayout; // set 1: ShadowPassUniforms (dynamic offset)
    rhi::ResourceSetHandle m_passSet;
    rhi::PipelineHandle m_pipelines[6]; // [casterPipeline()]: dynamic, instanced, skinned × alpha test
    rhi::ResourceSetLayoutHandle m_paletteLayout; // set 3 of skinned casters (owned by Environment)
    rhi::ResourceSetHandle m_paletteSet;
    ClipSpace m_clipSpace{rhi::ClipSpaceConvention{}}; // the device's, set in init()
    float m_nearClipZ = 0.0f;
    float m_ndcDepthScale = 1.0f; // backend NDC z per unit of depth01

    CascadeFit m_fits[kMaxCascades];
    uint32_t m_cascadeCount = 0;
    uint32_t m_passOffsets[kMaxCascades] = {};
    Spot m_spots[kMaxSpotShadows];
    uint32_t m_spotCount = 0;
    uint32_t m_spotPassOffsets[kMaxSpotShadows] = {};
    uint32_t m_spotAtlasSize = 16, m_spotTileSize = 8; // the atlas holds 2 × 2 tiles of half its size
    const LightDesc* m_previousSpots[kMaxSpotShadows] = {}; // last frame's winners (ranking hysteresis; compared, never read)
    bool m_warnedWideSpot = false;
    std::vector<const LightDesc*> m_spotCandidates; // reserved once
    RenderQueue m_queue{65536};
    Stats m_stats;
};

} // namespace renderer
