#include "renderer/vfx/SpriteRenderer.h"

#include "core/Log.h"
#include "core/Profile.h"
#include "renderer/core/ShaderLibrary.h"
#include "renderer/core/ShaderTypes.h"
#include "renderer/vfx/Flipbook.h"
#include "renderer/visibility/Visibility.h"

#include <algorithm>
#include <cstring>

namespace renderer {

core::Result<void> SpriteRenderer::init(rhi::Device& device, ShaderLibrary& shaders, ResourceManager& resources,
                                        rhi::ResourceSetLayoutHandle frameLayout) {
    m_device = &device;
    m_resources = &resources;
    m_frameLayout = frameLayout;
    auto program = shaders.program("sprite", 0);
    if (!program) return program.error();
    m_vertex = program.value().vertex;
    m_fragment = program.value().fragment;
    // Set 2, binding 1: the atlas (binding 0 of set 2 is a uniform-buffer slot on GL; BindingModel.h).
    rhi::ResourceSetLayoutDesc ld;
    ld.set = 2;
    ld.count = 1;
    ld.bindings[0] = {1, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment};
    m_atlasLayout = device.createResourceSetLayout(ld);
    rhi::SamplerDesc sd;
    sd.filter = rhi::Filter::Linear;
    sd.mip = rhi::MipMode::Linear;
    sd.address = rhi::AddressMode::ClampToEdge;
    m_sampler = device.createSampler(sd);
    if (!m_atlasLayout || !m_sampler) return core::Error::InitFailed;
    m_frame.resize(kCapacity);
    m_visible.resize(kCapacity);
    m_batches.resize(kCapacity);
    m_atlases.reserve(256); // atlases seen over a session: a new one later than that would allocate when it first draws
    return {};
}

void SpriteRenderer::shutdown() {
    if (!m_device) return;
    std::vector<SpriteId> ids;
    m_sprites.forEach([&](SpriteId id, SpriteDesc&) { ids.push_back(id); });
    for (SpriteId id : ids) remove(id);
    for (Atlas& a : m_atlases) m_device->destroy(a.set);
    m_atlases.clear();
    for (rhi::PipelineHandle& p : m_pipelines) m_device->destroy(p);
    m_device->destroy(m_sampler);
    m_device->destroy(m_atlasLayout);
    m_device = nullptr;
}

bool SpriteRenderer::setTarget(const rhi::AttachmentLayout& target) {
    using F = rhi::BlendFactor;
    static constexpr const char* kNames[3] = {"sprites (alpha)", "sprites (additive)", "sprites (premultiplied)"};
    const rhi::BlendState blends[3] = {
        {true, F::SrcAlpha, F::OneMinusSrcAlpha, rhi::BlendOp::Add, F::One, F::OneMinusSrcAlpha, rhi::BlendOp::Add, 0xF},
        {true, F::SrcAlpha, F::One, rhi::BlendOp::Add, F::Zero, F::One, rhi::BlendOp::Add, 0xF},
        {true, F::One, F::OneMinusSrcAlpha, rhi::BlendOp::Add, F::One, F::OneMinusSrcAlpha, rhi::BlendOp::Add, 0xF},
    };
    bool ok = true;
    for (uint32_t b = 0; b < 3; ++b) {
        rhi::PipelineDesc pd;
        pd.debugName = kNames[b];
        pd.vertex = m_vertex;
        pd.fragment = m_fragment;
        pd.vertexLayout.streams[0] = {uint16_t(sizeof(SpriteInstance)), true};
        pd.vertexLayout.streamCount = 1;
        for (uint8_t k = 0; k < 4; ++k) pd.vertexLayout.attributes[k] = {uint8_t(8 + k), 0, rhi::Format::RGBA32F, uint16_t(16 * k)};
        pd.vertexLayout.attributeCount = 4;
        pd.raster.cull = rhi::CullMode::None;
        pd.depth = {true, false, rhi::CompareOp::LessOrEqual}; // tested against the opaque world, never written
        pd.blend[0] = blends[b];
        pd.attachments = target;
        pd.setLayouts[0] = m_frameLayout;
        pd.setLayouts[2] = m_atlasLayout;
        m_device->destroy(m_pipelines[b]);
        m_pipelines[b] = m_device->createPipeline(pd);
        if (!m_pipelines[b]) {
            FL_LOG_ERROR("The %s pipeline could not be created", kNames[b]);
            ok = false;
        }
    }
    return ok;
}

SpriteId SpriteRenderer::add(const SpriteDesc& desc) {
    const SpriteId id = m_sprites.create(desc);
    if (!id) {
        FL_LOG_ERROR("Sprite pool exhausted (%u sprites)", kCapacity);
        return {};
    }
    m_resources->addRef(desc.atlas);
    return id;
}

void SpriteRenderer::update(SpriteId id, const SpriteDesc& desc) {
    SpriteDesc* s = m_sprites.get(id);
    if (!s) return;
    if (!(s->atlas == desc.atlas)) { // a new atlas: swap the references (a loading-time change)
        m_resources->addRef(desc.atlas);
        if (s->atlas) m_resources->release(s->atlas);
    }
    *s = desc;
}

void SpriteRenderer::remove(SpriteId id) {
    const SpriteDesc* s = m_sprites.get(id);
    if (!s) return;
    if (s->atlas) m_resources->release(s->atlas);
    m_sprites.destroy(id);
}

bool SpriteRenderer::finished(SpriteId id, float time) const {
    const SpriteDesc* s = m_sprites.get(id);
    return s && !s->loop && flipbookFrame(time - s->startTime, s->fps, s->firstFrame, s->frameCount, false).finished;
}

rhi::ResourceSetHandle SpriteRenderer::atlasSet(TextureAsset asset) {
    const rhi::TextureHandle texture = m_resources->texture(asset);
    if (!texture) return {}; // still loading (or failed): the sprite waits
    for (Atlas& a : m_atlases)
        if (a.asset == asset) {
            if (a.texture == texture) return a.set;
            m_device->destroy(a.set); // the texture was replaced (a reload): its set follows
            a.set = {};
            a.texture = texture;
        }
    const rhi::ResourceWrite write{1, {}, texture, m_sampler};
    rhi::ResourceSetDesc sd;
    sd.debugName = "sprite atlas";
    sd.layout = m_atlasLayout;
    sd.writes = {&write, 1};
    const rhi::ResourceSetHandle set = m_device->createResourceSet(sd);
    if (!set) return {};
    for (Atlas& a : m_atlases)
        if (a.asset == asset) return a.set = set;
    // A new atlas arrived (loading time): first drop the entries of textures that were released since (their handles are
    // stale), so sets and entries don't pile up as atlases come and go (M7 review).
    for (size_t i = 0; i < m_atlases.size();) {
        if (m_resources->state(m_atlases[i].asset) != LoadState::Failed) {
            ++i;
            continue;
        }
        m_device->destroy(m_atlases[i].set);
        m_atlases[i] = m_atlases.back();
        m_atlases.pop_back();
    }
    m_atlases.push_back({asset, texture, set});
    return set;
}

uint32_t SpriteRenderer::prepare(const Visibility& visibility, const glm::vec3& eye, float time) {
    FL_ZONE("SpriteRenderer::prepare");
    m_batchCount = 0;
    uint32_t visible = 0;
    m_sprites.forEach([&](SpriteId, SpriteDesc& s) {
        const float radius = 0.5f * glm::length(s.size);
        if (!visibility.cameraSees(glm::vec4(s.position, radius))) return;
        if (!s.loop && flipbookFrame(time - s.startTime, s.fps, s.firstFrame, s.frameCount, false).finished) return;
        if (time < s.startTime && !s.loop) return; // not started yet
        const glm::vec3 d = s.position - eye;
        m_frame[visible] = &s;
        m_visible[visible] = {glm::dot(d, d), visible};
        ++visible;
    });
    if (visible == 0) return 0;
    // Back to front; equal depths keep the pool order, so coincident sprites never swap from frame to frame.
    std::sort(m_visible.begin(), m_visible.begin() + visible,
              [](const Visible& a, const Visible& b) { return a.depth > b.depth || (a.depth == b.depth && a.sprite < b.sprite); });
    rhi::TransientRing& ring = m_device->transient(rhi::TransientKind::Vertex);
    const rhi::TransientRing::Allocation a = ring.allocate(visible * uint32_t(sizeof(SpriteInstance)));
    if (!a.cpu) return 0;
    auto* out = static_cast<SpriteInstance*>(a.cpu);
    uint32_t written = 0;
    for (uint32_t i = 0; i < visible; ++i) {
        const SpriteDesc& s = *m_frame[m_visible[i].sprite];
        const rhi::ResourceSetHandle set = atlasSet(s.atlas);
        if (!set) continue;
        const FlipbookFrame f = flipbookFrame(time - s.startTime, s.fps, s.firstFrame, s.frameCount, s.loop);
        SpriteInstance instance; // built here and written once: ring memory is write-only (ARCHITECTURE.md §4.8)
        instance.centerRotation = glm::vec4(s.position, s.rotation);
        instance.sizeFacing = glm::vec4(s.size, s.facing == SpriteFacing::AxisY ? 1.0f : 0.0f, float(uint32_t(s.blend)));
        instance.uvRect = flipbookRect(f.frame, s.columns, s.rows);
        instance.color = s.color;
        std::memcpy(out + written, &instance, sizeof(instance));
        const uint32_t blend = uint32_t(s.blend);
        if (m_batchCount > 0 && m_batches[m_batchCount - 1].set == set && m_batches[m_batchCount - 1].blend == blend) {
            ++m_batches[m_batchCount - 1].count;
        } else {
            m_batches[m_batchCount++] = {set, blend, written, 1};
        }
        ++written;
    }
    m_instances = a.gpu;
    return written;
}

void SpriteRenderer::record(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const {
    uint32_t boundBlend = ~0u;
    for (uint32_t b = 0; b < m_batchCount; ++b) {
        const Batch& batch = m_batches[b];
        const rhi::PipelineHandle pipeline = m_pipelines[batch.blend];
        if (!pipeline) continue;
        if (batch.blend != boundBlend) {
            cmd.bindPipeline(pipeline);
            cmd.bindResourceSet(0, frameSet, frameOffsets);
            boundBlend = batch.blend;
        }
        cmd.bindResourceSet(2, batch.set);
        const rhi::BufferSlice instances{m_instances.buffer, m_instances.offset + batch.first * uint32_t(sizeof(SpriteInstance)),
                                         batch.count * uint32_t(sizeof(SpriteInstance))};
        cmd.bindVertexStreams(0, {&instances, 1});
        cmd.draw(6, batch.count);
    }
}

} // namespace renderer
