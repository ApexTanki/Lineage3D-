#include "renderer/shadows/BlobShadows.h"
#include "renderer/visibility/Visibility.h"

#include "core/Log.h"
#include "renderer/core/ShaderLibrary.h"
#include "renderer/core/ShaderTypes.h"

#include <algorithm>
#include <cstring>

namespace renderer {
namespace {

constexpr float kLift = 0.02f;    // above the model's lowest point, so the quad wins the depth test against the ground
constexpr float kSpread = 1.6f;   // the ellipse reaches past the model's footprint (blob.frag: full strength up to 1/1.6)

} // namespace

core::Result<void> BlobShadows::init(rhi::Device& device, ShaderLibrary& shaders, rhi::ResourceSetLayoutHandle frameLayout) {
    m_device = &device;
    m_frameLayout = frameLayout;
    auto program = shaders.program("blob", 0);
    if (!program) return program.error();
    m_vertex = program.value().vertex;
    m_fragment = program.value().fragment;
    return {};
}

void BlobShadows::shutdown() {
    if (!m_device) return;
    m_device->destroy(m_pipeline);
    m_device = nullptr;
}

bool BlobShadows::setTarget(const rhi::AttachmentLayout& target) {
    rhi::PipelineDesc pd;
    pd.debugName = "blob shadows";
    pd.vertex = m_vertex;
    pd.fragment = m_fragment;
    pd.vertexLayout.streams[0] = {uint16_t(sizeof(BlobInstance)), true};
    pd.vertexLayout.streamCount = 1;
    pd.vertexLayout.attributes[0] = {8, 0, rhi::Format::RGBA32F, 0};
    pd.vertexLayout.attributes[1] = {9, 0, rhi::Format::RGBA32F, 16};
    pd.vertexLayout.attributeCount = 2;
    pd.raster.cull = rhi::CullMode::None;
    // Toward the camera, on top of the 2 cm lift: far away (≈ 250 m) the depth buffer no longer resolves 2 cm, and a
    // constant bias in depth-buffer units grows with distance exactly as that precision shrinks.
    pd.raster.depthBiasConstant = -4.0f;
    pd.raster.depthBiasSlope = -1.5f;
    pd.depth = {true, false, rhi::CompareOp::LessOrEqual};
    // Multiply: color = src × dst; the target's alpha stays.
    pd.blend[0] = {true, rhi::BlendFactor::DstColor, rhi::BlendFactor::Zero, rhi::BlendOp::Add,
                   rhi::BlendFactor::Zero, rhi::BlendFactor::One, rhi::BlendOp::Add, 0xF};
    pd.attachments = target;
    pd.setLayouts[0] = m_frameLayout;
    m_device->destroy(m_pipeline);
    m_pipeline = m_device->createPipeline(pd);
    if (!m_pipeline) FL_LOG_ERROR("The blob shadow pipeline could not be created");
    return bool(m_pipeline);
}

uint32_t BlobShadows::prepare(const Scene& scene, float strength, const Visibility* visibility) {
    m_count = 0;
    const auto blobs = scene.blobs();
    if (blobs.empty() || strength <= 0.0f) return 0;
    const rhi::TransientRing::Allocation a =
        m_device->transient(rhi::TransientKind::Vertex).allocate(uint32_t(blobs.size() * sizeof(BlobInstance)));
    if (!a.cpu) return 0;
    auto* out = static_cast<BlobInstance*>(a.cpu);
    const auto dynamics = scene.dynamics();
    for (const Scene::Blob& b : blobs) {
        if (b.dynamicCount == 0 || b.firstDynamic + b.dynamicCount > dynamics.size()) continue;
        // The ellipse turns with the model: its first renderable's X axis on the ground (a tank's hull), or its Z axis
        // when X points straight up.
        const Scene::Dynamic& main = dynamics[b.firstDynamic];
        glm::vec2 ax(main.boxHalfAxes[0].x, main.boxHalfAxes[0].z);
        if (glm::dot(ax, ax) < 1e-10f) ax = glm::vec2(main.boxHalfAxes[2].z, -main.boxHalfAxes[2].x);
        ax = glm::dot(ax, ax) > 1e-10f ? glm::normalize(ax) : glm::vec2(1.0f, 0.0f);
        const glm::vec2 az(-ax.y, ax.x);
        // Its extent covers every corner of every renderable's box (hull and turret alike), measured along those axes.
        glm::vec2 lo(INFINITY), hi(-INFINITY);
        float bottom = INFINITY;
        for (uint32_t d = b.firstDynamic; d < b.firstDynamic + b.dynamicCount; ++d) {
            const Scene::Dynamic& r = dynamics[d];
            for (int corner = 0; corner < 8; ++corner) {
                const glm::vec3 p = r.boxCenter + r.boxHalfAxes[0] * ((corner & 1) ? 1.0f : -1.0f) +
                                    r.boxHalfAxes[1] * ((corner & 2) ? 1.0f : -1.0f) + r.boxHalfAxes[2] * ((corner & 4) ? 1.0f : -1.0f);
                const glm::vec2 q(glm::dot(glm::vec2(p.x, p.z), ax), glm::dot(glm::vec2(p.x, p.z), az));
                lo = glm::min(lo, q);
                hi = glm::max(hi, q);
                bottom = std::min(bottom, p.y);
            }
        }
        const glm::vec2 center = ax * ((lo.x + hi.x) * 0.5f) + az * ((lo.y + hi.y) * 0.5f);
        const float radius = glm::length(hi - lo) * 0.5f * kSpread;
        if (visibility && !visibility->cameraSees(glm::vec4(center.x, bottom, center.y, radius))) continue;
        BlobInstance instance; // built here and written once: ring memory is write-only (ARCHITECTURE.md §4.8)
        instance.centerRadius = glm::vec4(center.x, bottom + kLift, center.y, (hi.x - lo.x) * 0.5f * kSpread);
        instance.params = glm::vec4(std::min(strength, 1.0f), (hi.y - lo.y) * 0.5f * kSpread, ax.x, ax.y);
        std::memcpy(out + m_count, &instance, sizeof(instance));
        ++m_count;
    }
    m_instances = rhi::BufferSlice{a.gpu.buffer, a.gpu.offset, m_count * uint32_t(sizeof(BlobInstance))};
    return m_count;
}

void BlobShadows::record(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const {
    if (m_count == 0 || !m_pipeline) return;
    cmd.bindPipeline(m_pipeline);
    cmd.bindResourceSet(0, frameSet, frameOffsets);
    cmd.bindVertexStreams(0, {&m_instances, 1});
    cmd.draw(6, m_count);
}

} // namespace renderer
