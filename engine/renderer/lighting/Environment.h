#pragma once

#include "assets/MapFormat.h"
#include "core/Result.h"
#include "renderer/core/ShaderTypes.h"
#include "renderer/lighting/ProbeGrid.h"
#include "renderer/resources/ResourceManager.h"
#include "rhi/Device.h"

namespace renderer {

// A map's baked lighting on the GPU (M3 plan D4–D8, D12): lightmap and shadowmask arrays and the sky cube (frame set,
// set 0 b4–b6), the reflection cubes (one set-3 resource set per slot: 0 = the sky, 1–7 = local probes) and the probe
// grid. Textures load asynchronously; until each arrives a default stands in (black lightmap, white shadowmask, and a
// tiny cube of the hemispherical ambient for reflections). Without a map the defaults are all there is, so M1/M2 scenes
// render as before plus ambient reflections. HDR textures are BC6H, or RGBM where the device lacks BPTC.
class Environment {
public:
    struct Desc {
        rhi::Device* device = nullptr;
        ResourceManager* resources = nullptr;
    };

    core::Result<void> init(const Desc& desc);
    void shutdown();

    // Starts loading a map's lighting; replaces any previous map. `probes` may be empty (no grid).
    void load(const assets::MapManifest& manifest, assets::ProbeGridData probes);
    void unload();
    // The default reflection cube follows the hemispherical ambient (no map, or until the sky arrives).
    void setAmbient(const glm::vec3& sky, const glm::vec3& ground);
    // Once per frame after ResourceManager::finalizeUploads(): swaps in textures that finished loading. revision()
    // changes whenever a texture or set the frame binds was replaced.
    void update();

    uint32_t revision() const { return m_revision; }
    rhi::TextureHandle skyCube() const;
    rhi::TextureHandle lightmaps() const;
    rhi::TextureHandle shadowmask() const;
    rhi::SamplerHandle cubeSampler() const { return m_cubeSampler; }
    rhi::SamplerHandle lightmapSampler() const { return m_lightmapSampler; }
    rhi::ResourceSetLayoutHandle reflectionLayout() const { return m_reflectionLayout; }
    rhi::ResourceSetHandle reflectionSet(uint32_t slot) const { return slot < kReflectionSlots && m_sets[slot] ? m_sets[slot] : m_sets[0]; }
    // Set 3 of skinned pipelines (M6 plan D9): binding 0 = the bone palette (a dynamic offset into the frame's uniform
    // ring), binding 1 = the reflection cube of the slot. Shadow and debug skinned pipelines read only the palette.
    rhi::ResourceSetLayoutHandle skinnedLayout() const { return m_skinnedLayout; }
    rhi::ResourceSetHandle skinnedSet(uint32_t slot) const {
        return slot < kReflectionSlots && m_skinnedSets[slot] ? m_skinnedSets[slot] : m_skinnedSets[0];
    }
    rhi::ResourceSetLayoutHandle paletteLayout() const { return m_paletteLayout; }
    rhi::ResourceSetHandle paletteSet() const { return m_paletteSet; }

    // The local reflection probe whose box holds `point` (the smallest if several), else 0 (the sky).
    uint8_t selectReflection(const glm::vec3& point) const;
    const ProbeGrid& probes() const { return m_probes; }
    bool hasMap() const { return m_hasMap; }
    bool hasSky() const;   // the map's sky cube has arrived (the sky pass draws it)
    bool rgbm() const { return m_rgbm; }
    void fillFrame(FrameUniforms& fu) const; // RGBM flag, sky max mip, reflection slots
    uint32_t pendingCount() const;          // map textures still loading
    uint32_t lightmapPages() const { return m_lightmapPages; }
    uint32_t reflectionProbeCount() const { return m_probeCount; }

private:
    struct Slot {
        TextureAsset asset;
        bool resolved = true; // Ready or Failed (nothing more will change)
    };
    void request(Slot& slot, const std::string& path);
    rhi::TextureHandle ready(const Slot& slot) const;
    void releaseSlots();
    void rebuildSets();
    rhi::TextureHandle makeAmbientCube(const glm::vec3& sky, const glm::vec3& ground) const;

    rhi::Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;
    bool m_rgbm = false;
    rhi::SamplerHandle m_cubeSampler, m_lightmapSampler;
    rhi::ResourceSetLayoutHandle m_reflectionLayout;
    rhi::ResourceSetHandle m_sets[kReflectionSlots];
    rhi::ResourceSetLayoutHandle m_skinnedLayout, m_paletteLayout;
    rhi::ResourceSetHandle m_skinnedSets[kReflectionSlots];
    rhi::ResourceSetHandle m_paletteSet;
    rhi::TextureHandle m_defaultLightmap, m_defaultShadowmask, m_ambientCube;
    glm::vec3 m_ambientSky{-1.0f}, m_ambientGround{-1.0f};

    bool m_hasMap = false;
    Slot m_lightmaps, m_shadowmask, m_sky;
    Slot m_reflections[kReflectionSlots]; // [0] unused: slot 0 is the sky
    struct Box {
        glm::vec3 min{0.0f}, max{0.0f}, position{0.0f};
    } m_boxes[kReflectionSlots];
    uint32_t m_probeCount = 0;
    uint32_t m_lightmapPages = 0;
    ProbeGrid m_probes;
    uint32_t m_revision = 1;
    uint32_t m_seenTextureRevision = 0; // ResourceManager::textureRevision() the sets were made at
};

} // namespace renderer
