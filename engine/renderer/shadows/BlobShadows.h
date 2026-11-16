#pragma once

#include "core/Result.h"
#include "renderer/core/Scene.h"
#include "rhi/Device.h"

#include <span>

namespace renderer {

class ShaderLibrary;

// Blob shadows (ARCHITECTURE.md §6 pass 5b, DEV-3; M4 plan D9): a soft horizontal ellipse under every dynamic model
// flagged blobShadow, sized by its world box and placed at its bottom, all in one instanced draw after the sky.
// The Low tier's only dynamic shadow and a contact shadow under the cascades on Medium/High.
class BlobShadows {
public:
    core::Result<void> init(rhi::Device& device, ShaderLibrary& shaders, rhi::ResourceSetLayoutHandle frameLayout);
    void shutdown();
    // Rebuilds the pipeline for a new scene attachment layout (format or MSAA change; never per frame).
    bool setTarget(const rhi::AttachmentLayout& target);

    // Writes this frame's instances into the vertex ring; returns how many blobs will draw. Blobs the camera can't see
    // (`visibility`, M5 plan D9) are skipped.
    uint32_t prepare(const Scene& scene, float strength, const class Visibility* visibility = nullptr);
    // Inside the scene render pass, after the sky.
    void record(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const;

private:
    rhi::Device* m_device = nullptr;
    rhi::ShaderHandle m_vertex, m_fragment;
    rhi::ResourceSetLayoutHandle m_frameLayout;
    rhi::PipelineHandle m_pipeline;
    rhi::BufferSlice m_instances;
    uint32_t m_count = 0;
};

} // namespace renderer
