#include "renderer/core/FramePipeline.h"

#include "core/Log.h"
#include "core/Math.h"
#include "renderer/core/ShaderLibrary.h"

#include <algorithm>

namespace renderer {

core::Result<void> FramePipeline::init(rhi::Device& device, ShaderLibrary& shaders, rhi::ResourceSetLayoutHandle frameLayout) {
    m_device = &device;
    m_shaders = &shaders;
    rhi::ResourceSetLayoutDesc ld;
    ld.set = 1;
    ld.count = 1;
    ld.bindings[0] = {1, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment};
    m_inputLayout = device.createResourceSetLayout(ld);

    rhi::SamplerDesc sd;
    sd.filter = rhi::Filter::Nearest; // 1:1 texel copy
    sd.mip = rhi::MipMode::Nearest;
    sd.address = rhi::AddressMode::ClampToEdge;
    m_pointClamp = device.createSampler(sd);

    auto program = shaders.program("present", 0);
    if (!program || !m_inputLayout || !m_pointClamp) return core::Error::InitFailed;
    rhi::PipelineDesc pd;
    pd.debugName = "present";
    pd.vertex = program.value().vertex;
    pd.fragment = program.value().fragment;
    pd.raster.cull = rhi::CullMode::None; // clip-space triangle: winding differs per backend (rule C8)
    pd.attachments.colorCount = 1;
    pd.attachments.color[0] = device.backbufferFormat();
    pd.setLayouts[0] = frameLayout;
    pd.setLayouts[1] = m_inputLayout;
    m_presentPipeline = device.createPipeline(pd);
    auto sky = shaders.program("sky", 0);
    if (!sky) return core::Error::InitFailed;
    m_skyVertex = sky.value().vertex;
    m_skyFragment = sky.value().fragment;
    m_frameLayout = frameLayout;

    // Pass 5c writes linear view depth: R32F where it renders and filters (the P2 passes sample it through the post chain's
    // linear sampler; linear filtering of R32F is optional on Vulkan), else R16F, whose filtering every device has.
    const rhi::FormatSupport& r32 = device.caps().formats[uint32_t(rhi::Format::R32F)];
    m_linearDepthFormat = r32.colorTarget && r32.sampled && r32.filterable ? rhi::Format::R32F : rhi::Format::R16F;
    auto msaaAxis = shaders.axis("depth_copy", "MSAA");
    if (!msaaAxis) return core::Error::InitFailed;
    for (uint32_t multisampled = 0; multisampled < 2; ++multisampled) {
        auto copy = shaders.program("depth_copy", multisampled ? 1ull << msaaAxis.value().bitOffset : 0ull);
        if (!copy) return core::Error::InitFailed;
        rhi::PipelineDesc cd;
        cd.debugName = multisampled ? "depth copy (msaa)" : "depth copy";
        cd.vertex = copy.value().vertex;
        cd.fragment = copy.value().fragment;
        cd.raster.cull = rhi::CullMode::None; // clip-space triangle: winding differs per backend (rule C8)
        cd.attachments.colorCount = 1;
        cd.attachments.color[0] = m_linearDepthFormat;
        cd.setLayouts[0] = frameLayout;
        cd.setLayouts[1] = m_inputLayout;
        cd.perDrawBytes = 16; // near and far planes
        m_depthCopyPipelines[multisampled] = device.createPipeline(cd);
        if (!m_depthCopyPipelines[multisampled]) return core::Error::CompileFailed;
    }
    return m_presentPipeline ? core::Result<void>() : core::Result<void>(core::Error::CompileFailed);
}

bool FramePipeline::createSkyPipeline() {
    rhi::PipelineDesc pd;
    pd.debugName = "sky";
    pd.vertex = m_skyVertex;
    pd.fragment = m_skyFragment;
    pd.raster.cull = rhi::CullMode::None; // clip-space triangle: winding differs per backend (rule C8)
    pd.depth = {true, false, rhi::CompareOp::LessOrEqual};
    pd.attachments = m_sceneLayout;
    pd.setLayouts[0] = m_frameLayout;
    m_device->destroy(m_skyPipeline);
    m_skyPipeline = m_device->createPipeline(pd);
    return bool(m_skyPipeline);
}

void FramePipeline::recordSky(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const {
    cmd.bindPipeline(m_skyPipeline);
    cmd.bindResourceSet(0, frameSet, frameOffsets);
    cmd.draw(3);
}

void FramePipeline::destroyTargets() {
    m_device->destroy(m_presentSet);
    m_device->destroy(m_depthCopySet);
    m_device->destroy(m_softSet);
    m_device->destroy(m_sceneColor);
    m_device->destroy(m_sceneColorMsaa);
    m_device->destroy(m_sceneDepth);
    m_device->destroy(m_linearDepth);
    m_presentSet = m_depthCopySet = m_softSet = {};
    m_sceneColor = m_sceneColorMsaa = m_sceneDepth = m_linearDepth = {};
}

void FramePipeline::shutdown() {
    if (!m_device) return;
    destroyTargets();
    for (rhi::PipelineHandle& p : m_depthCopyPipelines) {
        m_device->destroy(p);
        p = {};
    }
    m_device->destroy(m_skyPipeline);
    m_device->destroy(m_presentPipeline);
    m_device->destroy(m_pointClamp);
    m_device->destroy(m_inputLayout);
    m_device = nullptr;
}

core::Result<void> FramePipeline::prepare(rhi::Extent2D extent, uint8_t msaa, bool hdr, bool sampleDepth, bool& layoutChanged) {
    layoutChanged = false;
    const rhi::Caps& caps = m_device->caps();
    uint8_t samples = 1;
    while (uint32_t(samples) * 2 <= std::min<uint32_t>(msaa, caps.maxMsaaSamples) && samples < 4) samples = uint8_t(samples * 2);
    // Pass 5c reads the depth buffer (soft particles, SSAO, refraction, heat haze, P2): a device whose depth format can't
    // be sampled goes without them.
    if (sampleDepth && !caps.formats[uint32_t(caps.depthFormat)].sampled) {
        if (!m_warnedSoftDepth) FL_LOG_WARN("The depth format can't be sampled here: no soft particles, SSAO, refraction, haze or P2");
        m_warnedSoftDepth = true;
        sampleDepth = false;
    }
    if (m_sceneColor && extent.width == m_extent.width && extent.height == m_extent.height && samples == m_msaa &&
        sampleDepth == m_sampledDepth && hdr == m_hdr)
        return {};

    // HDR: R11G11B10F where it renders, blends and samples, else RGBA16F (R9). LDR (M8 plan D8): sRGB8, so blending
    // stays linear while the forward shaders tonemap.
    const rhi::FormatSupport& r11 = caps.formats[uint32_t(rhi::Format::R11G11B10F)];
    const rhi::Format colorFormat = !hdr ? rhi::Format::RGBA8Srgb
                                  : r11.colorTarget && r11.blendable && r11.sampled ? rhi::Format::R11G11B10F
                                                                                     : rhi::Format::RGBA16F;
    rhi::AttachmentLayout layout;
    layout.colorCount = 1;
    layout.color[0] = colorFormat;
    layout.depth = caps.depthFormat;
    layout.samples = samples;
    layoutChanged = !(layout.color[0] == m_sceneLayout.color[0] && layout.depth == m_sceneLayout.depth &&
                      layout.samples == m_sceneLayout.samples && m_sceneLayout.colorCount == 1);

    destroyTargets(); // deferred: frames in flight keep using the old images
    rhi::TextureDesc td;
    td.width = extent.width;
    td.height = extent.height;
    td.debugName = "sceneColor";
    td.format = colorFormat;
    td.usage = rhi::TextureUsage::ColorTarget | rhi::TextureUsage::Sampled;
    m_sceneColor = m_device->createTexture(td);
    if (samples > 1) {
        td.debugName = "sceneColorMsaa";
        td.samples = samples;
        td.usage = rhi::TextureUsage::ColorTarget;
        m_sceneColorMsaa = m_device->createTexture(td);
    }
    td.debugName = "sceneDepth";
    td.format = caps.depthFormat;
    td.samples = samples;
    td.usage = sampleDepth ? rhi::TextureUsage::DepthTarget | rhi::TextureUsage::Sampled : rhi::TextureUsage::DepthTarget;
    m_sceneDepth = m_device->createTexture(td);
    // Out of memory (or over the render-target budget, M9 plan D6): nothing half-made stays, and the key is forgotten, so
    // the next prepare builds every target again, whatever it asks for (the renderer may retry with the previous settings).
    auto fail = [&] {
        destroyTargets();
        m_msaa = 0;
        return core::Error::OutOfMemory;
    };
    if (!m_sceneColor || !m_sceneDepth || (samples > 1 && !m_sceneColorMsaa)) {
        FL_LOG_ERROR("Frame targets %ux%u (%ux MSAA) could not be created", extent.width, extent.height, samples);
        return fail();
    }
    const rhi::ResourceWrite write{1, {}, m_sceneColor, m_pointClamp};
    rhi::ResourceSetDesc sd;
    sd.debugName = "present inputs";
    sd.layout = m_inputLayout;
    sd.writes = {&write, 1};
    m_presentSet = m_device->createResourceSet(sd);
    if (!m_presentSet) return fail();
    m_sceneLayout = layout;
    m_extent = extent;
    m_msaa = samples;
    m_sampledDepth = sampleDepth;
    m_hdr = hdr;
    ++m_revision;
    if ((layoutChanged || !m_skyPipeline) && !createSkyPipeline()) {
        FL_LOG_ERROR("The sky pipeline could not be created");
        fail();
        return core::Error::CompileFailed;
    }
    if (sampleDepth && !createSoftDepth()) {
        // M9 plan D6: like any target of the request, a linear depth that doesn't fit refuses the settings (the renderer
        // keeps the previous ones) instead of turning its consumers off unseen. Nothing half-made stays.
        FL_LOG_ERROR("The linear depth target (soft particles, SSAO, refraction, haze, P2) could not be created");
        m_splitWarmupPending = false;
        return fail(); // destroys every target, the linear depth's sets included
    }
    return {};
}

bool FramePipeline::createSoftDepth() {
    rhi::TextureDesc td;
    td.debugName = "linearDepth";
    td.width = m_extent.width;
    td.height = m_extent.height;
    td.format = m_linearDepthFormat;
    td.usage = rhi::TextureUsage::ColorTarget | rhi::TextureUsage::Sampled;
    m_linearDepth = m_device->createTexture(td);
    if (!m_linearDepth) return false;
    const rhi::ResourceWrite depthInput{1, {}, m_sceneDepth, m_pointClamp};
    rhi::ResourceSetDesc sd;
    sd.debugName = "depth copy input";
    sd.layout = m_inputLayout;
    sd.writes = {&depthInput, 1};
    m_depthCopySet = m_device->createResourceSet(sd);
    const rhi::ResourceWrite softInput{1, {}, m_linearDepth, m_pointClamp};
    sd.debugName = "soft particle depth";
    sd.writes = {&softInput, 1};
    m_softSet = m_device->createResourceSet(sd);
    if (!m_depthCopySet || !m_softSet) return false;
    m_splitWarmupPending = true; // new targets: their render passes are made ahead of need (R60)
    return true;
}

void FramePipeline::recordSplitWarmup(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets,
                                      float zNear, float zFar) {
    if (!m_splitWarmupPending || !m_softSet) return;
    m_splitWarmupPending = false;
    // Every scene pass a frame may record with these targets, empty, so each render pass and framebuffer exists before
    // the first frame that needs it, however long after warm-up that comes (M7 plan D8; M8 plan D9–D11): pass A with
    // and without the prepass depth and the grab's resolve, the whole pass with and without the prepass depth, keeping
    // its depth for a post copy or not (a view that splits from its first frame on never records the plain one until the
    // camera turns), pass 5c, pass B, and pass 8 on the resolved target. The first pass clears, so every later one loads
    // defined images.
    const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (bool loaded : {false, true})
        for (bool resolve : {false, true}) {
            if (resolve && m_msaa == 1) continue; // single-sample pass A writes the resolved target itself
            cmd.beginRenderPass(scenePass(black, ScenePart::Opaque, loaded, false, resolve));
            cmd.endRenderPass();
        }
    for (bool loaded : {false, true})
        for (bool storeDepth : {false, true}) {
            cmd.beginRenderPass(scenePass(black, ScenePart::Whole, loaded, storeDepth));
            cmd.endRenderPass();
        }
    recordDepthCopy(cmd, frameSet, frameOffsets, zNear, zFar);
    cmd.beginRenderPass(scenePass(black, ScenePart::Blended));
    cmd.endRenderPass();
    cmd.beginRenderPass(hazePass());
    cmd.endRenderPass(); // the frame's own scene pass clears everything again
}

rhi::RenderPassDesc FramePipeline::scenePass(const float clearLinear[4], ScenePart part, bool depthLoaded, bool storeDepth,
                                             bool resolveOpaque) const {
    rhi::RenderPassDesc pass;
    pass.debugName = part == ScenePart::Opaque ? "Scene (opaque)" : part == ScenePart::Blended ? "Scene (blended)" : "Scene";
    pass.colorCount = 1;
    rhi::ColorAttachment& c = pass.color[0];
    c.load = part == ScenePart::Blended ? rhi::LoadOp::Load : rhi::LoadOp::Clear;
    std::copy(clearLinear, clearLinear + 4, c.clear);
    if (m_msaa > 1) {
        c.texture = m_sceneColorMsaa;
        if (part == ScenePart::Opaque) {
            c.store = rhi::StoreOp::Store; // pass B continues on the samples
            if (resolveOpaque) c.resolveTo = m_sceneColor; // and the grab reads the resolved opaque world (M8 plan D10)
        } else {
            c.store = rhi::StoreOp::DontCare; // only the resolve is kept
            c.resolveTo = m_sceneColor;
        }
    } else {
        c.texture = m_sceneColor;
        c.store = rhi::StoreOp::Store;
    }
    pass.useDepth = true;
    pass.depth.texture = m_sceneDepth;
    pass.depth.load = part == ScenePart::Blended || depthLoaded ? rhi::LoadOp::Load : rhi::LoadOp::Clear;
    pass.depth.store = part == ScenePart::Opaque || storeDepth ? rhi::StoreOp::Store : rhi::StoreOp::DontCare;
    pass.depth.clearDepth = 1.0f;
    return pass;
}

rhi::RenderPassDesc FramePipeline::hazePass() const {
    rhi::RenderPassDesc pass;
    pass.debugName = "Heat haze";
    pass.colorCount = 1;
    pass.color[0].texture = m_sceneColor;
    pass.color[0].load = rhi::LoadOp::Load;
    pass.color[0].store = rhi::StoreOp::Store;
    return pass;
}

rhi::RenderPassDesc FramePipeline::prepassPass() const {
    rhi::RenderPassDesc pass;
    pass.debugName = "Depth prepass";
    pass.colorCount = 0;
    pass.useDepth = true;
    pass.depth.texture = m_sceneDepth;
    pass.depth.load = rhi::LoadOp::Clear;
    pass.depth.store = rhi::StoreOp::Store; // the scene pass loads it; pass 5c copies it for SSAO
    pass.depth.clearDepth = 1.0f;
    return pass;
}

void FramePipeline::recordDepthCopy(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets,
                                    float zNear, float zFar) const {
    const rhi::PipelineHandle pipeline = m_depthCopyPipelines[m_msaa > 1 ? 1 : 0];
    if (!pipeline || !m_softSet) return;
    rhi::RenderPassDesc pass;
    pass.debugName = "Depth copy";
    pass.colorCount = 1;
    pass.color[0].texture = m_linearDepth;
    pass.color[0].load = rhi::LoadOp::DontCare; // the triangle covers every texel
    pass.color[0].store = rhi::StoreOp::Store;
    cmd.beginRenderPass(pass);
    cmd.bindPipeline(pipeline);
    cmd.bindResourceSet(0, frameSet, frameOffsets);
    cmd.bindResourceSet(1, m_depthCopySet);
    const glm::vec4 planes(zNear, zFar, 0.0f, 0.0f);
    cmd.setPerDraw(&planes, uint8_t(sizeof(planes)));
    cmd.draw(3);
    cmd.endRenderPass();
}

void FramePipeline::recordPresent(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const {
    cmd.bindPipeline(m_presentPipeline);
    cmd.bindResourceSet(0, frameSet, frameOffsets);
    cmd.bindResourceSet(1, m_presentSet);
    cmd.draw(3);
}

} // namespace renderer
