#pragma once

#include "core/Result.h"
#include "rhi/Device.h"

struct ImTextureData;

namespace renderer {

class ClipSpace;
class ShaderLibrary;

// Draws Dear ImGui through the RHI instead of ImGui's per-API backends, so the overlay goes through the same
// ClipSpace and binding paths as everything else (ARCHITECTURE.md §2.1, libraries). Implements ImGui 1.92's
// texture protocol (ImGuiBackendFlags_RendererHasTextures).
class ImGuiRenderer {
public:
    core::Result<void> init(rhi::Device& device, ShaderLibrary& shaders, const rhi::AttachmentLayout& target);
    void shutdown();

    // Records ImGui::GetDrawData() into `cmd`, inside an open render pass on the target.
    void render(rhi::CommandList& cmd, const ClipSpace& clipSpace, bool targetEncodesSrgb);

private:
    void updateTexture(ImTextureData* texture);

    rhi::Device* m_device = nullptr;
    rhi::PipelineHandle m_pipeline;
    rhi::SamplerHandle m_sampler;
    rhi::ResourceSetLayoutHandle m_setLayout;
    bool m_warnedRingFull = false;
};

} // namespace renderer
