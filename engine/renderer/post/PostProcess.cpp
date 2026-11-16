#include "renderer/post/PostProcess.h"

#include "core/Log.h"
#include "renderer/core/GpuTimers.h"
#include "renderer/core/ShaderLibrary.h"
#include "renderer/core/ShaderTypes.h"

#include <algorithm>

namespace renderer {
namespace {

rhi::TextureHandle solid(rhi::Device& device, const char* name, uint8_t value) {
    rhi::TextureDesc td;
    td.debugName = name;
    td.format = rhi::Format::RGBA8Unorm;
    const rhi::TextureHandle t = device.createTexture(td);
    const uint8_t pixel[4] = {value, value, value, 255};
    rhi::TextureUpload u;
    u.width = u.height = 1;
    u.data = std::as_bytes(std::span(pixel));
    if (t && !device.uploadTexture(t, u).ok()) {
        device.destroy(t);
        return {};
    }
    return t;
}

rhi::Format hdrTargetFormat(const rhi::Caps& caps) {
    const rhi::FormatSupport& r11 = caps.formats[uint32_t(rhi::Format::R11G11B10F)];
    return r11.colorTarget && r11.blendable && r11.sampled ? rhi::Format::R11G11B10F : rhi::Format::RGBA16F;
}

rhi::Extent2D half(rhi::Extent2D e) {
    return {std::max(1u, e.width / 2), std::max(1u, e.height / 2)};
}

} // namespace

uint32_t bloomMipCount(rhi::Extent2D scene) {
    uint32_t count = 0;
    while (count < kMaxBloomMips) {
        const rhi::Extent2D e = bloomMipExtent(scene, count);
        if (std::min(e.width, e.height) < 8) break;
        ++count;
    }
    return count;
}

rhi::Extent2D bloomMipExtent(rhi::Extent2D scene, uint32_t mip) {
    return {std::max(1u, scene.width >> (mip + 1)), std::max(1u, scene.height >> (mip + 1))};
}

core::Result<void> PostProcess::init(rhi::Device& device, ShaderLibrary& shaders, rhi::ResourceSetLayoutHandle frameLayout) {
    m_device = &device;
    m_shaders = &shaders;
    m_frameLayout = frameLayout;
    rhi::ResourceSetLayoutDesc ld;
    ld.set = 1;
    ld.count = 4;
    for (uint8_t b = 0; b < 4; ++b) ld.bindings[b] = {uint8_t(1 + b), rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment};
    m_inputLayout = device.createResourceSetLayout(ld);
    rhi::SamplerDesc sd;
    sd.filter = rhi::Filter::Linear;
    sd.mip = rhi::MipMode::Nearest;
    sd.address = rhi::AddressMode::ClampToEdge;
    m_linearClamp = device.createSampler(sd);
    m_black = solid(device, "black texel", 0);
    m_white = solid(device, "white texel", 255);
    if (!m_inputLayout || !m_linearClamp || !m_black || !m_white) return core::Error::InitFailed;

    // Pipelines whose target is the backbuffer or an sRGB8 image don't depend on the scene format: made once.
    m_tonemapBackbuffer = fullScreenPipeline("tonemap", "tonemap", 0, device.backbufferFormat(), false, 16);
    m_tonemapLdr = fullScreenPipeline("tonemap (ldrColor)", "tonemap", 0, rhi::Format::RGBA8Srgb, false, 16);
    m_fxaa = fullScreenPipeline("fxaa", "fxaa", 0, device.backbufferFormat(), false, 0);
    if (!m_tonemapBackbuffer || !m_tonemapLdr || !m_fxaa) return core::Error::CompileFailed;
    return {};
}

void PostProcess::destroyTargets() {
    for (uint32_t i = 0; i < kMaxBloomMips; ++i) {
        m_device->destroy(m_bloom[i]);
        m_device->destroy(m_bloomDownSets[i]);
        m_device->destroy(m_bloomUpSets[i]);
        m_bloom[i] = {};
        m_bloomDownSets[i] = m_bloomUpSets[i] = {};
    }
    m_bloomCount = 0;
    for (rhi::TextureHandle* t : {&m_ldrColor, &m_grab, &m_postColor[0], &m_postColor[1], &m_dof[0], &m_dof[1], &m_godRays[0], &m_godRays[1]}) {
        m_device->destroy(*t);
        *t = {};
    }
    for (rhi::ResourceSetHandle* s : {&m_grabSet, &m_dofCocSet, &m_dofBlurSet, &m_dofCompositeSet, &m_motionSet, &m_godMaskSet, &m_godBlurSet,
                                      &m_tonemapSet, &m_fxaaHdrSet, &m_fxaaLdrSet}) {
        m_device->destroy(*s);
        *s = {};
    }
    m_dofTarget = m_motionTarget = {};
    m_dofReady = m_motionReady = m_godRaysReady = false;
    m_grabWarmup = false; // set again with the next grab
}

void PostProcess::shutdown() {
    if (!m_device) return;
    destroyTargets();
    for (rhi::PipelineHandle p : {m_bloomPrefilter, m_bloomDown, m_bloomUp, m_copy, m_dofCoc, m_dofBlur, m_dofComposite, m_motionBlur, m_godMask,
                                  m_godBlur, m_tonemapBackbuffer, m_tonemapLdr, m_fxaa})
        m_device->destroy(p);
    m_device->destroy(m_black);
    m_device->destroy(m_white);
    m_device->destroy(m_linearClamp);
    m_device->destroy(m_inputLayout);
    m_device = nullptr;
}

rhi::TextureHandle PostProcess::target(const char* name, rhi::Extent2D extent, rhi::Format format) {
    rhi::TextureDesc td;
    td.debugName = name;
    td.width = extent.width;
    td.height = extent.height;
    td.format = format;
    td.usage = rhi::TextureUsage::ColorTarget | rhi::TextureUsage::Sampled;
    return m_device->createTexture(td);
}

rhi::PipelineHandle PostProcess::fullScreenPipeline(const char* name, const char* templateName, uint64_t key, rhi::Format target,
                                                    bool additive, uint8_t perDrawBytes) {
    auto program = m_shaders->program(templateName, key);
    if (!program) return {};
    rhi::PipelineDesc pd;
    pd.debugName = name;
    pd.vertex = program.value().vertex;
    pd.fragment = program.value().fragment;
    pd.raster.cull = rhi::CullMode::None; // clip-space triangle: winding differs per backend (rule C8)
    pd.attachments.colorCount = 1;
    pd.attachments.color[0] = target;
    if (additive) {
        pd.blend[0].enable = true;
        pd.blend[0].srcColor = pd.blend[0].dstColor = rhi::BlendFactor::One;
        pd.blend[0].srcAlpha = rhi::BlendFactor::Zero;
        pd.blend[0].dstAlpha = rhi::BlendFactor::One;
    }
    pd.setLayouts[0] = m_frameLayout;
    pd.setLayouts[1] = m_inputLayout;
    pd.perDrawBytes = perDrawBytes;
    const rhi::PipelineHandle pipeline = m_device->createPipeline(pd);
    if (!pipeline) FL_LOG_ERROR("Pipeline %s could not be created", name);
    return pipeline;
}

rhi::ResourceSetHandle PostProcess::inputSet(const char* name, rhi::TextureHandle b1, rhi::TextureHandle b2, rhi::TextureHandle b3,
                                             rhi::TextureHandle b4) {
    // Every binding gets a texture (black where the pass has no input), so the set is complete on both backends.
    const rhi::ResourceWrite writes[4] = {{1, {}, b1 ? b1 : m_black, m_linearClamp},
                                          {2, {}, b2 ? b2 : m_black, m_linearClamp},
                                          {3, {}, b3 ? b3 : m_black, m_linearClamp},
                                          {4, {}, b4 ? b4 : m_black, m_linearClamp}};
    rhi::ResourceSetDesc sd;
    sd.debugName = name;
    sd.layout = m_inputLayout;
    sd.writes = writes;
    return m_device->createResourceSet(sd);
}

core::Result<void> PostProcess::prepare(rhi::Extent2D extent, rhi::TextureHandle sceneColor, rhi::TextureHandle linearDepth, uint32_t revision,
                                        const PostSettings& settings) {
    const bool wantBloom = settings.hdr && settings.bloom;
    const bool wantLdr = settings.hdr && settings.fxaa;
    const bool wantGrab = settings.hdr && (settings.refraction || settings.heatHaze);
    const bool wantDof = settings.hdr && settings.depthOfField && linearDepth;
    const bool wantMotion = settings.hdr && settings.motionBlur && linearDepth;
    const bool wantGodRays = settings.hdr && settings.godRays && linearDepth;
    if (revision == m_revision && sceneColor == m_sceneColor && linearDepth == m_linearDepth && extent.width == m_extent.width &&
        extent.height == m_extent.height && settings.hdr == m_hdr && wantBloom == m_wantBloom && wantLdr == m_wantLdr &&
        wantGrab == m_wantGrab && wantDof == m_wantDof && wantMotion == m_wantMotion && wantGodRays == m_wantGodRays)
        return {};
    destroyTargets(); // deferred: frames in flight keep the old images
    m_revision = revision;
    m_sceneColor = sceneColor;
    m_linearDepth = linearDepth;
    m_extent = extent;
    m_hdr = settings.hdr;
    m_wantBloom = wantBloom;
    m_wantLdr = wantLdr;
    m_wantGrab = wantGrab;
    m_wantDof = wantDof;
    m_wantMotion = wantMotion;
    m_wantGodRays = wantGodRays;
    // A failure (out of memory, a pipeline that doesn't compile) leaves nothing half-built and forgets the key, so the next
    // frame tries again instead of taking the early-out above with targets, sets or pipelines missing.
    if (core::Result<void> built = build(extent, sceneColor, linearDepth); !built) {
        destroyTargets();
        m_revision = ~0u;
        return built;
    }
    return {};
}

core::Result<void> PostProcess::build(rhi::Extent2D extent, rhi::TextureHandle sceneColor, rhi::TextureHandle linearDepth) {
    const rhi::Format hdr = hdrTargetFormat(m_device->caps());
    if (hdr != m_hdrFormat) {
        for (rhi::PipelineHandle p : {m_bloomPrefilter, m_bloomDown, m_bloomUp, m_copy, m_dofCoc, m_dofBlur, m_dofComposite, m_motionBlur,
                                      m_godMask, m_godBlur})
            m_device->destroy(p);
        auto bloomMode = m_shaders->axis("bloom", "MODE");
        auto dofMode = m_shaders->axis("dof", "MODE");
        auto godMode = m_shaders->axis("god_rays", "MODE");
        if (!bloomMode || !dofMode || !godMode) return core::Error::InitFailed;
        const uint32_t b = bloomMode.value().bitOffset, d = dofMode.value().bitOffset, g = godMode.value().bitOffset;
        m_bloomPrefilter = fullScreenPipeline("bloom prefilter", "bloom", 0ull << b, hdr, false, 16);
        m_bloomDown = fullScreenPipeline("bloom down", "bloom", 1ull << b, hdr, false, 0);
        m_bloomUp = fullScreenPipeline("bloom up", "bloom", 2ull << b, hdr, true, 16);
        m_copy = fullScreenPipeline("scene grab", "copy", 0, hdr, false, 0);
        m_dofCoc = fullScreenPipeline("depth of field (CoC)", "dof", 0ull << d, rhi::Format::RGBA16F, false, 16);
        m_dofBlur = fullScreenPipeline("depth of field (blur)", "dof", 1ull << d, rhi::Format::RGBA16F, false, 16);
        m_dofComposite = fullScreenPipeline("depth of field (composite)", "dof", 2ull << d, hdr, false, 16);
        m_motionBlur = fullScreenPipeline("motion blur", "motion_blur", 0, hdr, false, uint8_t(sizeof(PerDraw)));
        m_godMask = fullScreenPipeline("god rays (mask)", "god_rays", 0ull << g, hdr, false, 16);
        m_godBlur = fullScreenPipeline("god rays (blur)", "god_rays", 1ull << g, hdr, false, 16);
        for (rhi::PipelineHandle p : {m_bloomPrefilter, m_bloomDown, m_bloomUp, m_copy, m_dofCoc, m_dofBlur, m_dofComposite, m_motionBlur,
                                      m_godMask, m_godBlur})
            if (!p) return core::Error::CompileFailed;
        m_hdrFormat = hdr;
    }

    // P2 (M8 plan D12): depth of field, then motion blur, each reading what came before it; `current` is what bloom and
    // the tonemap read.
    rhi::TextureHandle current = sceneColor;
    if (m_wantDof) {
        m_postColor[0] = target("post color 0", extent, hdr);
        m_dof[0] = target("dof color + CoC", half(extent), rhi::Format::RGBA16F);
        m_dof[1] = target("dof blur", half(extent), rhi::Format::RGBA16F);
        if (!m_postColor[0] || !m_dof[0] || !m_dof[1]) return core::Error::OutOfMemory;
        m_dofCocSet = inputSet("dof CoC inputs", sceneColor, linearDepth);
        m_dofBlurSet = inputSet("dof blur input", m_dof[0]);
        m_dofCompositeSet = inputSet("dof composite inputs", sceneColor, linearDepth, m_dof[1]);
        if (!m_dofCocSet || !m_dofBlurSet || !m_dofCompositeSet) return core::Error::OutOfMemory;
        m_dofTarget = current = m_postColor[0];
        m_dofReady = true;
    }
    if (m_wantMotion) {
        const uint32_t slot = current == m_postColor[0] ? 1 : 0;
        m_postColor[slot] = target(slot ? "post color 1" : "post color 0", extent, hdr);
        if (!m_postColor[slot]) return core::Error::OutOfMemory;
        m_motionSet = inputSet("motion blur inputs", current, linearDepth);
        if (!m_motionSet) return core::Error::OutOfMemory;
        m_motionTarget = current = m_postColor[slot];
        m_motionReady = true;
    }
    if (m_wantGodRays) {
        m_godRays[0] = target("god rays mask", half(extent), hdr);
        m_godRays[1] = target("god rays", half(extent), hdr);
        if (!m_godRays[0] || !m_godRays[1]) return core::Error::OutOfMemory;
        m_godMaskSet = inputSet("god rays mask inputs", current, linearDepth);
        m_godBlurSet = inputSet("god rays blur input", m_godRays[0]);
        if (!m_godMaskSet || !m_godBlurSet) return core::Error::OutOfMemory;
        m_godRaysReady = true;
    }

    if (m_wantBloom) {
        const uint32_t count = bloomMipCount(extent);
        static constexpr const char* kNames[kMaxBloomMips] = {"bloom 0", "bloom 1", "bloom 2", "bloom 3", "bloom 4"};
        for (uint32_t i = 0; i < count; ++i) {
            m_bloom[i] = target(kNames[i], bloomMipExtent(extent, i), hdr);
            if (!m_bloom[i]) return core::Error::OutOfMemory;
        }
        for (uint32_t i = 0; i < count; ++i) {
            m_bloomDownSets[i] = inputSet(i == 0 ? "bloom prefilter input" : "bloom down input", i == 0 ? current : m_bloom[i - 1]);
            if (i + 1 < count) m_bloomUpSets[i] = inputSet("bloom up input", m_bloom[i + 1]);
            if (!m_bloomDownSets[i] || (i + 1 < count && !m_bloomUpSets[i])) return core::Error::OutOfMemory;
        }
        m_bloomCount = count;
    }
    if (m_wantLdr) {
        m_ldrColor = target("ldrColor", extent, rhi::Format::RGBA8Srgb);
        if (!m_ldrColor) return core::Error::OutOfMemory;
        m_fxaaHdrSet = inputSet("fxaa input (ldrColor)", m_ldrColor);
        if (!m_fxaaHdrSet) return core::Error::OutOfMemory;
    }
    if (m_wantGrab) {
        m_grab = target("grab", half(extent), hdr);
        if (!m_grab) return core::Error::OutOfMemory;
        m_grabSet = inputSet("grab input", sceneColor);
        if (!m_grabSet) return core::Error::OutOfMemory;
        m_grabWarmup = true;
    }
    m_tonemapSet = inputSet("tonemap inputs", current, m_bloomCount ? m_bloom[0] : rhi::TextureHandle{},
                            m_godRaysReady ? m_godRays[1] : rhi::TextureHandle{});
    m_fxaaLdrSet = inputSet("fxaa input (scene)", sceneColor);
    if (!m_tonemapSet || !m_fxaaLdrSet) return core::Error::OutOfMemory;
    return {};
}

void PostProcess::fullScreen(rhi::CommandList& cmd, rhi::PipelineHandle pipeline, rhi::ResourceSetHandle frameSet,
                             std::span<const uint32_t> frameOffsets, rhi::ResourceSetHandle inputs, const void* perDraw, uint8_t bytes) const {
    cmd.bindPipeline(pipeline);
    cmd.bindResourceSet(0, frameSet, frameOffsets);
    cmd.bindResourceSet(1, inputs);
    if (bytes) cmd.setPerDraw(perDraw, bytes);
    cmd.draw(3);
}

void PostProcess::pass(rhi::CommandList& cmd, rhi::TextureHandle target, const char* name, rhi::LoadOp load) const {
    rhi::RenderPassDesc rp;
    rp.debugName = name;
    rp.colorCount = 1;
    rp.color[0].texture = target;
    rp.color[0].load = load;
    rp.color[0].store = rhi::StoreOp::Store;
    cmd.beginRenderPass(rp);
}

void PostProcess::recordP2(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets,
                           const PostSettings& settings, const P2Frame& frame, GpuTimers& timers) const {
    if (m_dofReady) {
        const uint32_t timer = timers.beginPass(cmd, "Depth of field");
        const glm::vec4 params(settings.focusDistance, settings.focusRange, 8.0f, 0.0f); // 8 half-res texels at full blur
        pass(cmd, m_dof[0], "Depth of field", rhi::LoadOp::DontCare);
        fullScreen(cmd, m_dofCoc, frameSet, frameOffsets, m_dofCocSet, &params, uint8_t(sizeof(params)));
        cmd.endRenderPass();
        pass(cmd, m_dof[1], "Depth of field", rhi::LoadOp::DontCare);
        fullScreen(cmd, m_dofBlur, frameSet, frameOffsets, m_dofBlurSet, &params, uint8_t(sizeof(params)));
        cmd.endRenderPass();
        pass(cmd, m_dofTarget, "Depth of field", rhi::LoadOp::DontCare);
        fullScreen(cmd, m_dofComposite, frameSet, frameOffsets, m_dofCompositeSet, &params, uint8_t(sizeof(params)));
        cmd.endRenderPass();
        timers.endPass(cmd, timer);
    }
    if (m_motionReady) {
        const uint32_t timer = timers.beginPass(cmd, "Motion blur");
        PerDraw pd;
        const glm::mat4& m = frame.previousViewProj;
        for (int r = 0; r < 3; ++r) pd.model.rows[r] = glm::vec4(m[0][r], m[1][r], m[2][r], m[3][r]);
        pd.lightmapST = glm::vec4(m[0][3], m[1][3], m[2][3], m[3][3]);
        pd.sh[0] = glm::vec4(settings.motionBlurScale, 0.04f, 0.0f, 0.0f); // the longest blur: 4 % of the screen
        pass(cmd, m_motionTarget, "Motion blur", rhi::LoadOp::DontCare);
        fullScreen(cmd, m_motionBlur, frameSet, frameOffsets, m_motionSet, &pd, uint8_t(sizeof(pd)));
        cmd.endRenderPass();
        timers.endPass(cmd, timer);
    }
    if (m_godRaysReady) { // also with the sun off screen (fade 0: black), so the tonemap never reads an unwritten target
        const uint32_t timer = timers.beginPass(cmd, "God rays");
        const glm::vec4 params(frame.sunUv, frame.zFar, frame.sunFade);
        pass(cmd, m_godRays[0], "God rays", rhi::LoadOp::DontCare);
        fullScreen(cmd, m_godMask, frameSet, frameOffsets, m_godMaskSet, &params, uint8_t(sizeof(params)));
        cmd.endRenderPass();
        pass(cmd, m_godRays[1], "God rays", rhi::LoadOp::DontCare);
        fullScreen(cmd, m_godBlur, frameSet, frameOffsets, m_godBlurSet, &params, uint8_t(sizeof(params)));
        cmd.endRenderPass();
        timers.endPass(cmd, timer);
    }
}

void PostProcess::recordBloom(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets,
                              const PostSettings& settings, float exposure, GpuTimers& timers) const {
    if (!m_bloomCount) return;
    const uint32_t timer = timers.beginPass(cmd, "Bloom");
    // Down: the prefilter into mip 0, then each mip from the one above.
    const glm::vec4 prefilter(settings.bloomThreshold, settings.bloomThreshold * settings.bloomKnee, exposure, 0.0f);
    for (uint32_t i = 0; i < m_bloomCount; ++i) {
        pass(cmd, m_bloom[i], "Bloom", rhi::LoadOp::DontCare); // the triangle covers every texel
        if (i == 0) fullScreen(cmd, m_bloomPrefilter, frameSet, frameOffsets, m_bloomDownSets[0], &prefilter, uint8_t(sizeof(prefilter)));
        else fullScreen(cmd, m_bloomDown, frameSet, frameOffsets, m_bloomDownSets[i], nullptr, 0);
        cmd.endRenderPass();
    }
    // Up: each mip's tent added into the one above it, ending in mip 0.
    const glm::vec4 tent(1.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t i = m_bloomCount - 1; i-- > 0;) {
        pass(cmd, m_bloom[i], "Bloom", rhi::LoadOp::Load);
        fullScreen(cmd, m_bloomUp, frameSet, frameOffsets, m_bloomUpSets[i], &tent, uint8_t(sizeof(tent)));
        cmd.endRenderPass();
    }
    timers.endPass(cmd, timer);
}

void PostProcess::recordTonemapToLdr(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets,
                                     const PostSettings& settings, GpuTimers& timers) const {
    if (!m_ldrColor) return;
    const uint32_t timer = timers.beginPass(cmd, "Tonemap");
    pass(cmd, m_ldrColor, "Tonemap", rhi::LoadOp::DontCare);
    const glm::vec4 params(m_bloomCount ? settings.bloomIntensity : 0.0f, m_godRaysReady ? settings.godRaysIntensity : 0.0f, 1.0f,
                           0.0f); // z: the sRGB8 target encodes
    fullScreen(cmd, m_tonemapLdr, frameSet, frameOffsets, m_tonemapSet, &params, uint8_t(sizeof(params)));
    cmd.endRenderPass();
    timers.endPass(cmd, timer);
}

void PostProcess::recordTonemap(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets,
                                const PostSettings& settings) const {
    const glm::vec4 params(m_bloomCount ? settings.bloomIntensity : 0.0f, m_godRaysReady ? settings.godRaysIntensity : 0.0f, 0.0f, 0.0f);
    fullScreen(cmd, m_tonemapBackbuffer, frameSet, frameOffsets, m_tonemapSet, &params, uint8_t(sizeof(params)));
}

void PostProcess::recordGrab(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const {
    if (!m_grab) return;
    pass(cmd, m_grab, "Grab", rhi::LoadOp::DontCare); // the triangle covers every texel
    fullScreen(cmd, m_copy, frameSet, frameOffsets, m_grabSet, nullptr, 0);
    cmd.endRenderPass();
}

void PostProcess::recordWarmup(rhi::CommandList& cmd) {
    if (!m_grabWarmup || !m_grab) return;
    m_grabWarmup = false;
    pass(cmd, m_grab, "Grab", rhi::LoadOp::DontCare); // the real grab pass's own load op, so this also creates its render pass
    cmd.endRenderPass();
}

void PostProcess::recordFxaa(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets, bool hdr) const {
    const rhi::ResourceSetHandle input = hdr ? m_fxaaHdrSet : m_fxaaLdrSet;
    if (!input) return;
    fullScreen(cmd, m_fxaa, frameSet, frameOffsets, input, nullptr, 0);
}

} // namespace renderer
