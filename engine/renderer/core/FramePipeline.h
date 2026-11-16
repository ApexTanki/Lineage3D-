#pragma once

#include "core/Result.h"
#include "rhi/Device.h"

#include <span>

namespace renderer {

class ShaderLibrary;

// The frame's render targets and fixed passes (ARCHITECTURE.md §6; M1 subset): the scene pass renders into
// sceneColor (R11G11B10F, else RGBA16F) with sceneDepth, optionally multisampled and resolved when the pass ends; the
// present pass copies sceneColor to the backbuffer with the output encoding. Transitions come from the render-pass
// convention (targets rest in the shader-read state between passes), so no explicit barriers are recorded.
// Soft particles (M7 plan D8): the scene pass can split in two around pass 5c, which copies the depth as linear view
// depth into a single-sample target that pass B's particles sample (GL can't sample an attached depth buffer).
class FramePipeline {
public:
    enum class ScenePart : uint8_t {
        Whole,  // one pass: clear, draw everything, resolve
        Opaque, // pass A: clear; color and depth are stored for pass B (no resolve)
        Blended // pass B: loads color and depth, resolves at its end
    };

    core::Result<void> init(rhi::Device& device, ShaderLibrary& shaders, rhi::ResourceSetLayoutHandle frameLayout);
    void shutdown();

    // (Re)creates the targets when the backbuffer size, the MSAA count, `hdr` (the scene format: R11G11B10F/RGBA16F, or
    // RGBA8 sRGB on the LDR path, M8 plan D8) or `sampleDepth` (pass 5c copies the depth for soft particles, SSAO,
    // refraction, heat haze and P2) changes.
    // `layoutChanged` reports a new scene attachment layout (format or sample count), after which scene pipelines must be
    // rebuilt.
    core::Result<void> prepare(rhi::Extent2D extent, uint8_t msaa, bool hdr, bool sampleDepth, bool& layoutChanged);

    const rhi::AttachmentLayout& sceneLayout() const { return m_sceneLayout; }
    // `depthLoaded`: the depth prepass already filled the depth (M8 plan D9), so the pass loads it instead of clearing.
    // `storeDepth`: a whole pass keeps its depth for a depth copy after it (post effects that read depth).
    // `resolveOpaque`: pass A also resolves into sceneColor (keeping its samples for pass B) because a grab follows.
    rhi::RenderPassDesc scenePass(const float clearLinear[4], ScenePart part = ScenePart::Whole, bool depthLoaded = false,
                                  bool storeDepth = false, bool resolveOpaque = false) const;
    // Pass 2 (M8 plan D9): depth only into sceneDepth, cleared and stored for the scene pass and pass 5c.
    rhi::RenderPassDesc prepassPass() const;
    // Pass 8 (M8 plan D11): heat haze draws into the resolved scene target, loaded, with no depth attachment.
    rhi::RenderPassDesc hazePass() const;
    // Pass 5c's output (R32F, else R16F), while sampled depth is on: SSAO, refraction and post effects read it.
    rhi::TextureHandle linearDepth() const { return m_linearDepth; }
    // Pass 5c, between scene passes A and B: its own render pass into the linear depth target.
    void recordDepthCopy(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets, float zNear,
                         float zFar) const;
    // Once after the targets were (re)created, before the frame's scene pass: every scene pass variant, 5c and pass 8
    // without draws, so their render passes and framebuffers exist before the first frame that needs one.
    void recordSplitWarmup(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets, float zNear,
                           float zFar);
    bool softDepthAvailable() const { return bool(m_softSet); }
    // Set 1 of the soft particle pipelines (binding 1: the linear depth) and its set, valid while softDepthAvailable().
    rhi::ResourceSetLayoutHandle softLayout() const { return m_inputLayout; }
    rhi::ResourceSetHandle softSet() const { return m_softSet; }
    // Inside the scene render pass, after alpha test (pass 5, M3 plan D9): the sky cube on the far plane.
    void recordSky(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const;
    // Inside the backbuffer render pass: full-screen copy of sceneColor.
    // `frameOffsets`: the frame set's two dynamic offsets (FrameUniforms, LightArray).
    void recordPresent(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const;

    uint8_t msaa() const { return m_msaa; }
    // The resolved single-sample scene target the post chain reads, its format, and a counter that changes whenever the
    // targets are recreated (post sets that reference them must follow).
    rhi::TextureHandle sceneColor() const { return m_sceneColor; }
    rhi::Format sceneFormat() const { return m_sceneLayout.color[0]; }
    uint32_t revision() const { return m_revision; }
    bool hdr() const { return m_hdr; }
    rhi::Extent2D extent() const { return m_extent; }

private:
    void destroyTargets();
    bool createSkyPipeline();
    bool createSoftDepth(); // the linear depth target and its sets

    rhi::Device* m_device = nullptr;
    ShaderLibrary* m_shaders = nullptr;
    rhi::ResourceSetLayoutHandle m_inputLayout; // set 1, binding 1, one texture: the present, depth-copy and soft inputs
    rhi::PipelineHandle m_presentPipeline;
    rhi::PipelineHandle m_skyPipeline; // rebuilt with the scene layout
    rhi::ShaderHandle m_skyVertex, m_skyFragment;
    rhi::ResourceSetLayoutHandle m_frameLayout;
    rhi::SamplerHandle m_pointClamp;
    rhi::ResourceSetHandle m_presentSet;
    rhi::TextureHandle m_sceneColor;     // single-sample, sampled by the present pass
    rhi::TextureHandle m_sceneColorMsaa; // only with MSAA
    rhi::TextureHandle m_sceneDepth;     // sampled by pass 5c while soft depth is on
    rhi::TextureHandle m_linearDepth;    // pass 5c's output (R32F, else R16F), sampled by soft particles
    rhi::ResourceSetHandle m_depthCopySet, m_softSet;
    // Pass 5c's pipelines for a single-sample [0] and a multisampled [1] depth buffer: made at init and kept, so neither a
    // depth consumer toggling nor a resize compiles anything (M9 plan D3).
    rhi::PipelineHandle m_depthCopyPipelines[2];
    rhi::Format m_linearDepthFormat = rhi::Format::R16F;
    rhi::AttachmentLayout m_sceneLayout;
    rhi::Extent2D m_extent;
    uint8_t m_msaa = 0;
    bool m_sampledDepth = false;
    bool m_warnedSoftDepth = false;
    bool m_splitWarmupPending = false;
    bool m_hdr = true;
    uint32_t m_revision = 0;
};

} // namespace renderer
