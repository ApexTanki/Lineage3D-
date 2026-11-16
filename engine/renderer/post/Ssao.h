#pragma once

#include "core/Result.h"
#include "rhi/Device.h"

#include <span>

namespace renderer {

class GpuTimers;
class ShaderLibrary;

// Pass 2b (ARCHITECTURE.md §6, DEV-1; M8 plan D9): ambient occlusion at half resolution from the prepass's linear depth,
// then a separable depth-aware blur. The result is bound at frame-set binding 7, where the lit shader multiplies the
// indirect light and the environment reflection by it (never direct light).
class Ssao {
public:
    core::Result<void> init(rhi::Device& device, ShaderLibrary& shaders, rhi::ResourceSetLayoutHandle frameLayout);
    void shutdown();

    // (Re)creates the two half-resolution targets and their sets when the linear depth target or the size changes;
    // `enabled` false drops them. Returns whether the result texture changed (the frame set must follow).
    core::Result<bool> prepare(rhi::Extent2D extent, rhi::TextureHandle linearDepth, uint32_t revision, bool enabled);

    // AO, blur across, blur down: the result ends in result().
    void record(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets, float radius,
                float intensity, float zFar, GpuTimers& timers) const;

    rhi::TextureHandle result() const { return m_ao[0]; }
    bool ready() const { return bool(m_ao[0]); }

private:
    void destroyTargets();

    rhi::Device* m_device = nullptr;
    rhi::ResourceSetLayoutHandle m_frameLayout;
    rhi::ResourceSetLayoutHandle m_inputLayout; // set 1: binding 1 the linear depth, binding 2 the AO to blur
    rhi::SamplerHandle m_pointClamp;
    rhi::PipelineHandle m_aoPipeline, m_blurPipeline;
    rhi::TextureHandle m_ao[2];                   // R8, half resolution: [0] AO and the final result, [1] the blur between
    rhi::ResourceSetHandle m_aoSet, m_blurSet[2]; // blur [0] reads m_ao[0] into m_ao[1], [1] reads m_ao[1] into m_ao[0]
    rhi::TextureHandle m_linearDepth;
    rhi::Extent2D m_extent;
    uint32_t m_revision = ~0u;
};

} // namespace renderer
