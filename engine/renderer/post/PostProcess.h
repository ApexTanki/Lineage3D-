#pragma once

#include "core/Math.h"
#include "core/Result.h"
#include "renderer/post/PostSettings.h"
#include "rhi/Device.h"

#include <span>

namespace renderer {

class GpuTimers;
class ShaderLibrary;

// Bloom mips (M8 plan D6): from half resolution down, at most kMaxBloomMips, none smaller than 8 pixels.
inline constexpr uint32_t kMaxBloomMips = 5;
uint32_t bloomMipCount(rhi::Extent2D scene);
rhi::Extent2D bloomMipExtent(rhi::Extent2D scene, uint32_t mip);

// The post chain after the scene passes (ARCHITECTURE.md §6 passes 5d/8 grabs, 9b–9e; M8 plan D3–D7, D10–D12): the scene
// grab, the P2 effects, bloom, the tonemap and FXAA. Every target and resource set is made by prepare() when the scene
// targets, the size or the enabled features change; steady frames only record. Full-screen passes share present.vert,
// take their parameters as per-draw constants and their inputs from set 1 (bindings 1–4).
class PostProcess {
public:
    // What the P2 passes need from the frame (M8 plan D12).
    struct P2Frame {
        glm::mat4 previousViewProj{1.0f}; // motion blur: last frame's backend view-projection
        glm::vec2 sunUv{0.5f};            // god rays: the sun's position in screen UV
        float sunFade = 0.0f;             // god rays: 1 with the sun on screen, 0 well outside it or behind the camera
        float zFar = 1000.0f;
    };

    core::Result<void> init(rhi::Device& device, ShaderLibrary& shaders, rhi::ResourceSetLayoutHandle frameLayout);
    void shutdown();

    // `sceneColor` is the resolved scene target and `revision` changes whenever FramePipeline recreates it (or the
    // linear depth the P2 effects read).
    core::Result<void> prepare(rhi::Extent2D extent, rhi::TextureHandle sceneColor, rhi::TextureHandle linearDepth, uint32_t revision,
                               const PostSettings& settings);

    // Pass 9b (M8 plan D12): depth of field, then motion blur, into the post targets, and the god rays the tonemap adds.
    void recordP2(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets,
                  const PostSettings& settings, const P2Frame& frame, GpuTimers& timers) const;
    // Pass 9c, before the backbuffer pass: the bloom chain from the scene (or the P2 result), exposure applied in its
    // prefilter.
    void recordBloom(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets,
                     const PostSettings& settings, float exposure, GpuTimers& timers) const;
    // Pass 9d into ldrColor (its own render pass), when FXAA follows.
    void recordTonemapToLdr(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets,
                            const PostSettings& settings, GpuTimers& timers) const;
    // Inside the backbuffer render pass: the tonemap straight to the backbuffer (no FXAA).
    void recordTonemap(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets,
                       const PostSettings& settings) const;
    // Inside the backbuffer render pass: FXAA from ldrColor (HDR path) or from the scene target (LDR path).
    void recordFxaa(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets, bool hdr) const;
    // Passes 5d and 8 (M8 plan D10, D11): the resolved scene into the half-res grab, as its own render pass.
    void recordGrab(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const;
    // Once after the grab was (re)created, before anything samples it: an empty grab pass gives it the shader-read layout
    // (set 1 of pass B binds it even in frames that take no grab) and creates that render pass ahead of need (M8 plan D10).
    void recordWarmup(rhi::CommandList& cmd);

    bool bloomReady() const { return m_bloomCount > 0; }
    bool ldrReady() const { return bool(m_ldrColor); }
    bool p2Ready() const { return m_dofReady || m_motionReady || m_godRaysReady; }
    rhi::TextureHandle blackTexture() const { return m_black; }
    rhi::TextureHandle whiteTexture() const { return m_white; }
    rhi::SamplerHandle linearClamp() const { return m_linearClamp; }
    rhi::TextureHandle grab() const { return m_grab; } // the half-res scene grab (refraction, heat haze; M8 plan D10, D11)

private:
    void destroyTargets();
    // prepare()'s work once the key changed: pipelines for the HDR format, then the enabled features' targets and sets.
    core::Result<void> build(rhi::Extent2D extent, rhi::TextureHandle sceneColor, rhi::TextureHandle linearDepth);
    rhi::TextureHandle target(const char* name, rhi::Extent2D extent, rhi::Format format);
    rhi::PipelineHandle fullScreenPipeline(const char* name, const char* templateName, uint64_t key, rhi::Format target,
                                           bool additive, uint8_t perDrawBytes);
    rhi::ResourceSetHandle inputSet(const char* name, rhi::TextureHandle b1, rhi::TextureHandle b2 = {}, rhi::TextureHandle b3 = {},
                                    rhi::TextureHandle b4 = {});
    void fullScreen(rhi::CommandList& cmd, rhi::PipelineHandle pipeline, rhi::ResourceSetHandle frameSet,
                    std::span<const uint32_t> frameOffsets, rhi::ResourceSetHandle inputs, const void* perDraw, uint8_t bytes) const;
    void pass(rhi::CommandList& cmd, rhi::TextureHandle target, const char* name, rhi::LoadOp load) const;

    rhi::Device* m_device = nullptr;
    ShaderLibrary* m_shaders = nullptr;
    rhi::ResourceSetLayoutHandle m_frameLayout;
    rhi::ResourceSetLayoutHandle m_inputLayout; // set 1: bindings 1–4, textures
    rhi::SamplerHandle m_linearClamp;
    rhi::TextureHandle m_black, m_white; // 1×1, for inputs a pass doesn't have this frame (and SSAO off)

    // Pipelines that depend on target formats are rebuilt with them.
    rhi::Format m_hdrFormat = rhi::Format::Undefined;
    rhi::PipelineHandle m_bloomPrefilter, m_bloomDown, m_bloomUp, m_copy;
    rhi::PipelineHandle m_dofCoc, m_dofBlur, m_dofComposite, m_motionBlur, m_godMask, m_godBlur; // P2 (M8 plan D12)
    rhi::PipelineHandle m_tonemapBackbuffer, m_tonemapLdr, m_fxaa;

    rhi::TextureHandle m_bloom[kMaxBloomMips];
    uint32_t m_bloomCount = 0;
    rhi::ResourceSetHandle m_bloomDownSets[kMaxBloomMips]; // [0] reads the chain's color (prefilter), [i] reads mip i − 1
    rhi::ResourceSetHandle m_bloomUpSets[kMaxBloomMips];   // [i] reads mip i + 1 and is drawn into mip i
    rhi::TextureHandle m_ldrColor;
    rhi::TextureHandle m_grab; // half resolution, the HDR format; while refraction or heat haze is on
    rhi::ResourceSetHandle m_grabSet;
    bool m_grabWarmup = false;
    // P2 (M8 plan D12): two full-res HDR targets the effects ping-pong between, depth of field's half-res color + CoC and
    // its blur, the god rays' half-res mask and blur.
    rhi::TextureHandle m_postColor[2], m_dof[2], m_godRays[2];
    rhi::ResourceSetHandle m_dofCocSet, m_dofBlurSet, m_dofCompositeSet, m_motionSet, m_godMaskSet, m_godBlurSet;
    rhi::TextureHandle m_dofTarget, m_motionTarget; // where depth of field and motion blur write in this configuration
    bool m_dofReady = false, m_motionReady = false, m_godRaysReady = false;
    rhi::ResourceSetHandle m_tonemapSet, m_fxaaHdrSet, m_fxaaLdrSet;

    rhi::Extent2D m_extent;
    rhi::TextureHandle m_sceneColor, m_linearDepth;
    uint32_t m_revision = ~0u;
    bool m_hdr = false, m_wantBloom = false, m_wantLdr = false, m_wantGrab = false;
    bool m_wantDof = false, m_wantMotion = false, m_wantGodRays = false;
};

} // namespace renderer
