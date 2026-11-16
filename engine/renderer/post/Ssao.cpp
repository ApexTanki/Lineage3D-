#include "renderer/post/Ssao.h"

#include "core/Log.h"
#include "core/Math.h"
#include "renderer/core/GpuTimers.h"
#include "renderer/core/ShaderLibrary.h"

#include <algorithm>

namespace renderer {

core::Result<void> Ssao::init(rhi::Device& device, ShaderLibrary& shaders, rhi::ResourceSetLayoutHandle frameLayout) {
    m_device = &device;
    m_frameLayout = frameLayout;
    rhi::ResourceSetLayoutDesc ld;
    ld.set = 1;
    ld.count = 2;
    ld.bindings[0] = {1, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment};
    ld.bindings[1] = {2, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment};
    m_inputLayout = device.createResourceSetLayout(ld);
    rhi::SamplerDesc sd;
    sd.filter = rhi::Filter::Nearest; // texelFetch everywhere; the lit shader upsamples with its own linear sampler
    sd.mip = rhi::MipMode::Nearest;
    sd.address = rhi::AddressMode::ClampToEdge;
    m_pointClamp = device.createSampler(sd);
    auto mode = shaders.axis("ssao", "MODE");
    if (!m_inputLayout || !m_pointClamp || !mode) return core::Error::InitFailed;
    for (uint32_t m = 0; m < 2; ++m) {
        auto program = shaders.program("ssao", uint64_t(m) << mode.value().bitOffset);
        if (!program) return core::Error::InitFailed;
        rhi::PipelineDesc pd;
        pd.debugName = m == 0 ? "ssao" : "ssao blur";
        pd.vertex = program.value().vertex;
        pd.fragment = program.value().fragment;
        pd.raster.cull = rhi::CullMode::None; // clip-space triangle: winding differs per backend (rule C8)
        pd.attachments.colorCount = 1;
        pd.attachments.color[0] = rhi::Format::R8Unorm;
        pd.setLayouts[0] = frameLayout;
        pd.setLayouts[1] = m_inputLayout;
        pd.perDrawBytes = 16;
        (m == 0 ? m_aoPipeline : m_blurPipeline) = device.createPipeline(pd);
    }
    return m_aoPipeline && m_blurPipeline ? core::Result<void>() : core::Result<void>(core::Error::CompileFailed);
}

void Ssao::destroyTargets() {
    for (rhi::TextureHandle& t : m_ao) {
        m_device->destroy(t);
        t = {};
    }
    m_device->destroy(m_aoSet);
    m_device->destroy(m_blurSet[0]);
    m_device->destroy(m_blurSet[1]);
    m_aoSet = m_blurSet[0] = m_blurSet[1] = {};
}

void Ssao::shutdown() {
    if (!m_device) return;
    destroyTargets();
    m_device->destroy(m_aoPipeline);
    m_device->destroy(m_blurPipeline);
    m_device->destroy(m_pointClamp);
    m_device->destroy(m_inputLayout);
    m_device = nullptr;
}

core::Result<bool> Ssao::prepare(rhi::Extent2D extent, rhi::TextureHandle linearDepth, uint32_t revision, bool enabled) {
    if (!enabled || !linearDepth) {
        if (!m_ao[0]) return false;
        destroyTargets();
        m_revision = ~0u;
        return true;
    }
    if (m_ao[0] && revision == m_revision && linearDepth == m_linearDepth && extent.width == m_extent.width &&
        extent.height == m_extent.height)
        return false;
    destroyTargets(); // deferred: frames in flight keep the old images
    m_revision = revision;
    m_linearDepth = linearDepth;
    m_extent = extent;
    // Out of memory: nothing half-built stays (record() and ready() look at m_ao[0]) and the key is forgotten, so the next
    // frame tries again.
    auto fail = [&] {
        destroyTargets();
        m_revision = ~0u;
        return core::Error::OutOfMemory;
    };
    for (uint32_t i = 0; i < 2; ++i) {
        rhi::TextureDesc td;
        td.debugName = i == 0 ? "ssao" : "ssao blur";
        td.width = std::max(1u, extent.width / 2);
        td.height = std::max(1u, extent.height / 2);
        td.format = rhi::Format::R8Unorm;
        td.usage = rhi::TextureUsage::ColorTarget | rhi::TextureUsage::Sampled;
        m_ao[i] = m_device->createTexture(td);
        if (!m_ao[i]) return fail();
    }
    auto makeSet = [&](const char* name, rhi::TextureHandle ao) {
        const rhi::ResourceWrite writes[2] = {{1, {}, linearDepth, m_pointClamp}, {2, {}, ao, m_pointClamp}};
        rhi::ResourceSetDesc sd;
        sd.debugName = name;
        sd.layout = m_inputLayout;
        sd.writes = writes;
        return m_device->createResourceSet(sd);
    };
    m_aoSet = makeSet("ssao inputs", m_ao[1]); // binding 2 is unused by the AO variant: any texture that isn't its target
    m_blurSet[0] = makeSet("ssao blur across", m_ao[0]);
    m_blurSet[1] = makeSet("ssao blur down", m_ao[1]);
    if (!m_aoSet || !m_blurSet[0] || !m_blurSet[1]) return fail();
    return true;
}

void Ssao::record(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets, float radius,
                  float intensity, float zFar, GpuTimers& timers) const {
    if (!m_ao[0]) return;
    const uint32_t timer = timers.beginPass(cmd, "SSAO");
    auto pass = [&](rhi::TextureHandle target, rhi::PipelineHandle pipeline, rhi::ResourceSetHandle inputs, const glm::vec4& params) {
        rhi::RenderPassDesc rp;
        rp.debugName = "SSAO";
        rp.colorCount = 1;
        rp.color[0].texture = target;
        rp.color[0].load = rhi::LoadOp::DontCare; // the triangle covers every texel
        rp.color[0].store = rhi::StoreOp::Store;
        cmd.beginRenderPass(rp);
        cmd.bindPipeline(pipeline);
        cmd.bindResourceSet(0, frameSet, frameOffsets);
        cmd.bindResourceSet(1, inputs);
        cmd.setPerDraw(&params, uint8_t(sizeof(params)));
        cmd.draw(3);
        cmd.endRenderPass();
    };
    pass(m_ao[0], m_aoPipeline, m_aoSet, glm::vec4(radius, intensity, zFar, 0.0f));
    pass(m_ao[1], m_blurPipeline, m_blurSet[0], glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    pass(m_ao[0], m_blurPipeline, m_blurSet[1], glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
    timers.endPass(cmd, timer);
}

} // namespace renderer
