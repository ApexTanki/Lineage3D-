#pragma once

#include "core/Result.h"
#include "renderer/vfx/Sprites.h"
#include "rhi/Device.h"

#include <span>
#include <vector>

namespace renderer {

class ShaderLibrary;
class Visibility;

// Flipbook sprites (M6 plan D11). Every frame the sprites the camera sees are animated (flipbookFrame), sorted back to
// front and written into the vertex ring as 64-byte instances; consecutive sprites with the same atlas and blend mode
// become one instanced draw of the `sprite` template, after the scene's transparent draws (sprites sort among
// themselves, not with transparent meshes). Unlit, with fog. Nothing allocates per frame: the sorting arrays are sized
// to kCapacity, and an atlas's resource set is made once, when its texture first arrives.
class SpriteRenderer {
public:
    static constexpr uint32_t kCapacity = 1024; // game-placed sprites (fires, flashes); M7's particles have their own pools
    // The command bytes of the most draws the sprites can record (one per sprite, each its own batch): part of the
    // renderer's command-list bound (R48), by capacity, so sprites coming and going never grow a list.
    static constexpr size_t kMaxCommandBytes = size_t(kCapacity) * 160 + 4096;

    core::Result<void> init(rhi::Device& device, ShaderLibrary& shaders, ResourceManager& resources, rhi::ResourceSetLayoutHandle frameLayout);
    void shutdown();
    // Recreates the pipelines for a scene attachment layout (format or MSAA change; never per frame).
    bool setTarget(const rhi::AttachmentLayout& target);

    SpriteId add(const SpriteDesc& desc); // takes a reference on the atlas texture
    void update(SpriteId id, const SpriteDesc& desc);
    void remove(SpriteId id);
    const SpriteDesc* get(SpriteId id) const { return m_sprites.get(id); }
    uint32_t count() const { return m_sprites.liveCount(); }
    // A one-shot sprite past its last frame at `time` (it draws nothing; the game removes it).
    bool finished(SpriteId id, float time) const;

    // After visibility: culls, animates and sorts this frame's sprites into the vertex ring. Returns the sprites drawn.
    uint32_t prepare(const Visibility& visibility, const glm::vec3& eye, float time);
    // Inside the scene pass, after the transparent draws.
    void record(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const;
    uint32_t drawCount() const { return m_batchCount; }

private:
    struct Atlas {
        TextureAsset asset;
        rhi::TextureHandle texture; // the one its set was made for
        rhi::ResourceSetHandle set;
    };
    struct Visible {
        float depth;     // squared distance to the eye
        uint32_t sprite; // pool slot of the sprite this frame (into m_frame)
    };
    struct Batch {
        rhi::ResourceSetHandle set;
        uint32_t blend = 0;
        uint32_t first = 0, count = 0; // instances
    };

    rhi::ResourceSetHandle atlasSet(TextureAsset asset);

    rhi::Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;
    rhi::ResourceSetLayoutHandle m_frameLayout, m_atlasLayout;
    rhi::SamplerHandle m_sampler;
    rhi::ShaderHandle m_vertex, m_fragment;
    rhi::PipelineHandle m_pipelines[3]; // [SpriteBlend]
    core::HandlePool<SpriteDesc, SceneSpriteTag, SpriteId::kMaxGeneration, 10> m_sprites{kCapacity};
    std::vector<Atlas> m_atlases;
    std::vector<const SpriteDesc*> m_frame; // this frame's visible sprites
    std::vector<Visible> m_visible;
    std::vector<Batch> m_batches;
    uint32_t m_batchCount = 0;
    rhi::BufferSlice m_instances;
};

} // namespace renderer
