#include "renderer/particles/ParticleRenderer.h"

#include "core/Log.h"
#include "core/Profile.h"
#include "platform/JobSystem.h"
#include "platform/Timer.h"
#include "renderer/core/RenderQueue.h"
#include "renderer/core/ShaderLibrary.h"
#include "renderer/core/ShaderTypes.h"
#include "renderer/core/SortKey.h"
#include "renderer/lighting/ProbeGrid.h"
#include "renderer/visibility/Visibility.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace renderer {

static_assert(uint32_t(DebugView::Overdraw) == 18, "particle.frag tests FL_DEBUG_OVERDRAW 18");

namespace {

constexpr float kPi = 3.14159265358979f;

uint8_t unorm8(float v) {
    return uint8_t(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// Rotations grow without bound with spin; half floats keep them precise only near zero.
float wrapAngle(float radians) {
    return std::remainder(radians, 2.0f * kPi);
}

} // namespace

uint16_t packHalf(float value) {
    // F. Giesen's float_to_half_fast3_rtne (public domain), clamping instead of producing infinities: a size or speed
    // beyond the half range must not turn a quad into a screen-filling triangle.
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    x &= 0x7FFFFFFFu;
    if (x > 0x7F800000u) return 0;                          // NaN
    if (x >= 0x477FE000u) return uint16_t(sign | 0x7BFFu);  // ≥ 65504: the largest finite half
    if (x < 0x38800000u) {                                  // below 2^−14: a subnormal half or zero
        float f;
        std::memcpy(&f, &x, sizeof(f));
        f += 0.5f; // aligns the 10 mantissa bits at the bottom; float addition rounds to nearest even
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return uint16_t(sign | (u - 0x3F000000u));
    }
    const uint32_t odd = (x >> 13) & 1u;
    x += 0xC8000FFFu + odd; // rebias the exponent (127 → 15) and round to nearest even
    return uint16_t(sign | (x >> 13));
}

ParticleFrame particleFrame(const assets::ParticleEmitterDesc& d, float age, float normalizedAge, float frameOffset) {
    ParticleFrame out;
    if (d.frameCount <= 1) return out;
    const float count = float(d.frameCount);
    float f = (d.overLife ? normalizedAge * count : age * d.fps) + frameOffset * count;
    // Over life, a random start wraps around; at a frame rate, a looping flipbook wraps and a one-shot holds its last.
    const bool wraps = d.overLife ? d.randomStart : d.flipbookLoop;
    if (wraps) {
        f = std::fmod(f, count);
        if (f < 0.0f) f += count;
        out.frame = std::floor(f);
        out.blend = d.interpolate ? f - out.frame : 0.0f; // the shader's next frame wraps too
    } else {
        f = std::clamp(f, 0.0f, count - 1.0f);
        out.frame = std::floor(f);
        out.blend = d.interpolate && out.frame < count - 1.0f ? f - out.frame : 0.0f;
    }
    if (!(out.frame < count)) { // fmod can round up to `count`
        out.frame = count - 1.0f;
        out.blend = 0.0f;
    }
    return out;
}

core::Result<void> ParticleRenderer::init(rhi::Device& device, ShaderLibrary& shaders, ResourceManager& resources,
                                          rhi::ResourceSetLayoutHandle frameLayout, rhi::ResourceSetLayoutHandle softLayout) {
    m_device = &device;
    m_resources = &resources;
    m_frameLayout = frameLayout;
    m_softLayout = softLayout;
    auto soft = shaders.axis("particle", "SOFT");
    if (!soft) return soft.error();
    for (uint32_t s = 0; s < 2; ++s) {
        auto program = shaders.program("particle", uint64_t(s) << soft.value().bitOffset);
        if (!program) return program.error();
        m_vertex[s] = program.value().vertex;
        m_fragment[s] = program.value().fragment;
    }
    // Heat haze (M8 plan D11): the DISTORT variant and its set 1 (binding 1 the linear depth, binding 2 the grab).
    auto distort = shaders.axis("particle", "DISTORT");
    if (!distort) return distort.error();
    auto haze = shaders.program("particle", 1ull << distort.value().bitOffset);
    if (!haze) return haze.error();
    m_hazeVertex = haze.value().vertex;
    m_hazeFragment = haze.value().fragment;
    rhi::ResourceSetLayoutDesc hl;
    hl.set = 1;
    hl.count = 2;
    hl.bindings[0] = {1, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment};
    hl.bindings[1] = {2, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment};
    m_hazeLayout = device.createResourceSetLayout(hl);
    if (!m_hazeLayout) return core::Error::InitFailed;
    m_hazeDraws.resize(ParticleSystem::kMaxEmitters);
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
    const rhi::ResourceWrite white{1, {}, resources.whiteTexture(), m_sampler};
    rhi::ResourceSetDesc setDesc;
    setDesc.debugName = "particle atlas (white)";
    setDesc.layout = m_atlasLayout;
    setDesc.writes = {&white, 1};
    m_whiteSet = device.createResourceSet(setDesc);
    if (!m_whiteSet) return core::Error::InitFailed;
    m_results.resize(ParticleSystem::kMaxEmitters);
    m_draws.resize(ParticleSystem::kMaxEmitters);
    m_sortKeys.resize(ParticleSystem::kArenaCapacity);
    m_atlases.reserve(64); // atlases seen over a session: a new one later than that would allocate when it first draws
    return {};
}

void ParticleRenderer::shutdown() {
    if (!m_device) return;
    for (Atlas& a : m_atlases) m_device->destroy(a.set);
    m_atlases.clear();
    for (auto& blend : m_pipelines)
        for (rhi::PipelineHandle& p : blend) m_device->destroy(p);
    m_device->destroy(m_overdraw);
    m_device->destroy(m_hazePipeline);
    m_device->destroy(m_hazeSet);
    m_device->destroy(m_hazeLayout);
    m_device->destroy(m_whiteSet);
    m_device->destroy(m_sampler);
    m_device->destroy(m_atlasLayout);
    m_device = nullptr;
}

bool ParticleRenderer::setHazeInputs(rhi::TextureHandle linearDepth, rhi::TextureHandle grab) {
    if (!linearDepth || !grab) { // heat haze can't run: no set (FrameInput::haze stays off)
        m_device->destroy(m_hazeSet);
        m_hazeSet = {};
        m_hazeInputs[0] = m_hazeInputs[1] = {};
        return true;
    }
    if (m_hazeSet && linearDepth == m_hazeInputs[0] && grab == m_hazeInputs[1]) return true;
    const rhi::ResourceWrite writes[2] = {{1, {}, linearDepth, m_sampler}, {2, {}, grab, m_sampler}};
    rhi::ResourceSetDesc sd;
    sd.debugName = "heat haze inputs";
    sd.layout = m_hazeLayout;
    sd.writes = writes;
    const rhi::ResourceSetHandle set = m_device->createResourceSet(sd);
    if (!set) return false;
    m_device->destroy(m_hazeSet); // deferred: frames in flight keep the old set
    m_hazeSet = set;
    m_hazeInputs[0] = linearDepth;
    m_hazeInputs[1] = grab;
    return true;
}

void ParticleRenderer::recordHaze(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const {
    if (!m_hazeCount || !m_hazePipeline || !m_hazeSet) return;
    cmd.bindPipeline(m_hazePipeline);
    cmd.bindResourceSet(0, frameSet, frameOffsets);
    cmd.bindResourceSet(1, m_hazeSet);
    for (uint32_t i = 0; i < m_hazeCount; ++i) {
        const Draw& d = m_hazeDraws[i];
        cmd.bindResourceSet(2, d.atlas);
        cmd.setPerDraw(&d.constants, uint8_t(sizeof(d.constants)));
        cmd.bindVertexStreams(0, {&d.instances, 1});
        cmd.draw(6, d.count);
    }
}

bool ParticleRenderer::setTarget(const rhi::AttachmentLayout& target) {
    using F = rhi::BlendFactor;
    static constexpr const char* kNames[3][2] = {{"particles (alpha)", "particles (alpha, soft)"},
                                                 {"particles (additive)", "particles (additive, soft)"},
                                                 {"particles (premultiplied)", "particles (premultiplied, soft)"}};
    const rhi::BlendState blends[3] = {
        {true, F::SrcAlpha, F::OneMinusSrcAlpha, rhi::BlendOp::Add, F::One, F::OneMinusSrcAlpha, rhi::BlendOp::Add, 0xF},
        {true, F::SrcAlpha, F::One, rhi::BlendOp::Add, F::Zero, F::One, rhi::BlendOp::Add, 0xF},
        {true, F::One, F::OneMinusSrcAlpha, rhi::BlendOp::Add, F::One, F::OneMinusSrcAlpha, rhi::BlendOp::Add, 0xF},
    };
    auto describe = [&](uint32_t soft) {
        rhi::PipelineDesc pd;
        pd.vertex = m_vertex[soft];
        pd.fragment = m_fragment[soft];
        pd.vertexLayout.streams[0] = {uint16_t(sizeof(ParticleInstance)), true};
        pd.vertexLayout.streamCount = 1;
        pd.vertexLayout.attributes[0] = {8, 0, rhi::Format::RGB32F, uint16_t(offsetof(ParticleInstance, position))};
        pd.vertexLayout.attributes[1] = {9, 0, rhi::Format::RGBA16F, uint16_t(offsetof(ParticleInstance, sizeRotation))};
        pd.vertexLayout.attributes[2] = {10, 0, rhi::Format::RGBA8Unorm, uint16_t(offsetof(ParticleInstance, color))};
        pd.vertexLayout.attributes[3] = {11, 0, rhi::Format::RGBA16F, uint16_t(offsetof(ParticleInstance, axisFrame))};
        pd.vertexLayout.attributeCount = 4;
        pd.raster.cull = rhi::CullMode::None;
        pd.depth = {true, false, rhi::CompareOp::LessOrEqual}; // tested against the opaque world, never written
        pd.attachments = target;
        pd.setLayouts[0] = m_frameLayout;
        if (soft) pd.setLayouts[1] = m_softLayout;
        pd.setLayouts[2] = m_atlasLayout;
        pd.perDrawBytes = uint8_t(sizeof(ParticleDrawConstants));
        return pd;
    };
    bool ok = true;
    for (uint32_t b = 0; b < 3; ++b)
        for (uint32_t s = 0; s < 2; ++s) {
            rhi::PipelineDesc pd = describe(s);
            pd.debugName = kNames[b][s];
            pd.blend[0] = blends[b];
            m_device->destroy(m_pipelines[b][s]);
            m_pipelines[b][s] = m_device->createPipeline(pd);
            if (!m_pipelines[b][s]) {
                FL_LOG_ERROR("The %s pipeline could not be created", kNames[b][s]);
                ok = false;
            }
        }
    // The overdraw view (M5 plan D10): every quad adds its constant, no depth test, like the debug overdraw pass.
    rhi::PipelineDesc pd = describe(0);
    pd.debugName = "particles (overdraw)";
    pd.depth = {false, false, rhi::CompareOp::Always};
    pd.blend[0] = {true, F::One, F::One, rhi::BlendOp::Add, F::Zero, F::One, rhi::BlendOp::Add, 0xF};
    m_device->destroy(m_overdraw);
    m_overdraw = m_device->createPipeline(pd);
    if (!m_overdraw) {
        FL_LOG_ERROR("The particle overdraw pipeline could not be created");
        ok = false;
    }
    // Heat haze (pass 8, M8 plan D11): into the resolved single-sample scene target, alpha-blended, with no depth buffer
    // (the shader's depth fade is the occlusion test).
    rhi::PipelineDesc haze = describe(0);
    haze.debugName = "particles (heat haze)";
    haze.vertex = m_hazeVertex;
    haze.fragment = m_hazeFragment;
    haze.depth = {false, false, rhi::CompareOp::Always};
    haze.blend[0] = blends[0];
    haze.attachments = {};
    haze.attachments.colorCount = 1;
    haze.attachments.color[0] = target.color[0];
    haze.setLayouts[1] = m_hazeLayout;
    m_device->destroy(m_hazePipeline);
    m_hazePipeline = m_device->createPipeline(haze);
    if (!m_hazePipeline) {
        FL_LOG_ERROR("The heat haze pipeline could not be created");
        ok = false;
    }
    return ok;
}

rhi::ResourceSetHandle ParticleRenderer::atlasSet(TextureAsset asset) {
    const rhi::TextureHandle texture = m_resources->texture(asset);
    if (!texture) return {}; // still loading (or failed): the emitter waits
    for (Atlas& a : m_atlases)
        if (a.asset == asset) {
            if (a.texture == texture) return a.set;
            m_device->destroy(a.set); // the texture was replaced (a reload): its set follows
            a.set = {};
            a.texture = texture;
        }
    const rhi::ResourceWrite write{1, {}, texture, m_sampler};
    rhi::ResourceSetDesc sd;
    sd.debugName = "particle atlas";
    sd.layout = m_atlasLayout;
    sd.writes = {&write, 1};
    const rhi::ResourceSetHandle set = m_device->createResourceSet(sd);
    if (!set) return {};
    for (Atlas& a : m_atlases)
        if (a.asset == asset) return a.set = set;
    // A new atlas arrived (loading time): first drop the entries of textures that were released since (their handles are
    // stale), so sets and entries don't pile up as presets come and go.
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

uint32_t ParticleRenderer::prepare(platform::JobSystem& jobs, const FrameInput& in) {
    FL_ZONE("ParticleRenderer::prepare");
    const uint64_t start = platform::nowTicks();
    m_drawCount = 0;
    m_hazeCount = 0;
    m_stats = Stats{};
    const std::span<const uint32_t> active = in.system->activeEmitters();
    JobContext context{this, &in, active};
    jobs.parallelFor(uint32_t(active.size()), 2, prepareRange, &context);
    // Draws keep the active order, so equal depths always queue the same way.
    for (uint32_t k = 0; k < active.size(); ++k) {
        Draw& r = m_results[k];
        if (r.count == 0) {
            if (r.emitter == ~0u) ++m_stats.dropped; // the vertex ring was full
            continue;
        }
        const ParticleSystem::Emitter& e = in.system->emitter(r.emitter);
        const bool distort = r.blend == uint32_t(assets::ParticleBlend::Distort);
        if (distort && !in.haze) { // heat haze off: distort emitters aren't drawn (M8 plan D11)
            ++m_stats.hazeSkipped;
            continue;
        }
        r.atlas = e.texture ? atlasSet(e.texture) : m_whiteSet;
        if (!r.atlas) continue; // its atlas is still loading (or failed, which the resource manager logged)
        if (distort) m_hazeDraws[m_hazeCount++] = r; // pass 8, after everything they bend
        else m_draws[m_drawCount++] = r;
        m_stats.instances += r.count;
        m_stats.softDraws += r.soft && !distort ? 1u : 0u;
        m_stats.coverage += r.coverage;
    }
    // Haze draws blend over one another back to front; equal depths order by emitter slot (std::sort never allocates).
    std::sort(m_hazeDraws.begin(), m_hazeDraws.begin() + m_hazeCount,
              [](const Draw& a, const Draw& b) { return a.depth > b.depth || (a.depth == b.depth && a.emitter < b.emitter); });
    m_stats.draws = m_drawCount;
    m_stats.hazeDraws = m_hazeCount;
    if (m_stats.dropped && !m_warnedDrops) {
        FL_LOG_WARN("Vertex ring full: %u particle emitters not drawn this frame", m_stats.dropped);
        m_warnedDrops = true;
    }
    m_stats.prepareMs = platform::ticksToMilliseconds(platform::nowTicks() - start);
    return m_drawCount;
}

void ParticleRenderer::prepareRange(void* context, uint32_t begin, uint32_t end, uint32_t) {
    const JobContext& c = *static_cast<const JobContext*>(context);
    ParticleRenderer& self = *c.self;
    const FrameInput& in = *c.in;
    const ParticleSystem& system = *in.system;
    const ParticleArena& a = system.arena();
    rhi::TransientRing& ring = self.m_device->transient(rhi::TransientKind::Vertex);
    // Screen coverage of a quad of area A at distance d: A / ((2 d tan(fov/2))² · aspect) screens.
    const float coverageScale = 1.0f / (4.0f * in.tanHalfFovY * in.tanHalfFovY * in.aspect);
    for (uint32_t k = begin; k < end; ++k) {
        Draw& out = self.m_results[k];
        out.count = 0;
        out.emitter = 0;
        const ParticleSystem::Emitter& e = system.emitter(c.active[k]);
        const assets::ParticleEmitterDesc& d = *e.desc;
        const glm::vec3 center = (e.boundsMin + e.boundsMax) * 0.5f;
        if (!in.visibility->cameraSees(glm::vec4(center, glm::length(e.boundsMax - center)))) continue;
        const rhi::TransientRing::Allocation alloc = ring.allocate(e.count * uint32_t(sizeof(ParticleInstance)));
        if (!alloc.cpu) {
            out.emitter = ~0u;
            continue;
        }
        // Alpha and premultiplied particles blend back to front; additive ones add in any order. Keys pack the
        // squared distance's bits (monotonic for positive floats), inverted so the farthest comes first, over the
        // particle's index, which breaks ties the same way every time.
        uint64_t* keys = self.m_sortKeys.data() + e.first;
        const bool sorted = d.blend != assets::ParticleBlend::Additive;
        for (uint32_t i = 0; i < e.count; ++i) {
            if (!sorted) {
                keys[i] = i;
                continue;
            }
            const uint32_t p = e.first + i;
            const float dx = a.px[p] - in.eye.x, dy = a.py[p] - in.eye.y, dz = a.pz[p] - in.eye.z;
            const float d2 = dx * dx + dy * dy + dz * dz;
            uint32_t bits;
            std::memcpy(&bits, &d2, sizeof(bits));
            const uint32_t farFirst = 0xFFFFFFFFu - bits;
            keys[i] = uint64_t(farFirst) << 32 | i;
        }
        if (sorted) std::sort(keys, keys + e.count);

        auto* dst = static_cast<ParticleInstance*>(alloc.cpu);
        const bool velocity = d.billboard == assets::ParticleBillboard::Velocity;
        float coverage = 0.0f;
        for (uint32_t i = 0; i < e.count; ++i) {
            const uint32_t p = e.first + uint32_t(keys[i] & 0xFFFFFFFFu);
            const float t = a.age[p] * a.invLife[p];
            const float size = d.size(t) * a.sizeScale[p];
            float height = size, vx = 0.0f, vy = 0.0f, vz = 0.0f;
            if (velocity) {
                vx = a.vx[p], vy = a.vy[p], vz = a.vz[p];
                height = size + std::sqrt(vx * vx + vy * vy + vz * vz) * d.stretch;
            }
            const glm::vec4 color = d.color(t);
            const ParticleFrame frame = particleFrame(d, a.age[p], t, a.frameOffset[p]);
            ParticleInstance instance; // built here and written once: ring memory is write-only (ARCHITECTURE.md §4.8)
            instance.position[0] = a.px[p];
            instance.position[1] = a.py[p];
            instance.position[2] = a.pz[p];
            instance.sizeRotation[0] = packHalf(size);
            instance.sizeRotation[1] = packHalf(height);
            instance.sizeRotation[2] = packHalf(wrapAngle(a.rotation[p]));
            instance.sizeRotation[3] = packHalf(frame.blend);
            instance.color[0] = unorm8(color.r);
            instance.color[1] = unorm8(color.g);
            instance.color[2] = unorm8(color.b);
            instance.color[3] = unorm8(color.a);
            instance.axisFrame[0] = packHalf(vx);
            instance.axisFrame[1] = packHalf(vy);
            instance.axisFrame[2] = packHalf(vz);
            instance.axisFrame[3] = packHalf(frame.frame);
            std::memcpy(dst + i, &instance, sizeof(instance));
            const float dx = a.px[p] - in.eye.x, dy = a.py[p] - in.eye.y, dz = a.pz[p] - in.eye.z;
            const float d2 = std::max(dx * dx + dy * dy + dz * dz, 0.25f); // within half a meter a quad covers the screen
            coverage += std::min(1.0f, size * height * coverageScale / d2);
        }

        out.emitter = c.active[k];
        out.instances = alloc.gpu;
        out.count = e.count;
        out.depth = std::clamp(glm::length(center - in.eye) * in.invFar, 0.0f, 1.0f);
        out.blend = uint32_t(d.blend);
        out.soft = in.soft && d.soft > 0.0f;
        out.coverage = coverage;
        ParticleDrawConstants& pc = out.constants;
        pc.params = glm::vec4(float(uint32_t(d.billboard)), float(uint32_t(d.blend)), d.soft > 0.0f ? 1.0f / d.soft : 0.0f, d.intensity);
        pc.flipbook = glm::vec4(float(d.columns), float(d.rows), float(d.frameCount), d.interpolate ? 1.0f : 0.0f);
        pc.normalLit = glm::vec4(d.normal, d.lit ? 1.0f : 0.0f);
        pc.sun = glm::vec4(1.0f, d.strength, 0.0f, 0.0f); // y: the distort strength (M8 plan D11)
        // Distort draws have no depth buffer: their depth fade is the occlusion test, so it never switches off.
        if (d.blend == assets::ParticleBlend::Distort) pc.params.z = 1.0f / std::max(d.soft, 0.1f);
        if (d.lit) { // the probes at the cloud's center (M7 plan D9), or the hemispherical ambient without a grid
            if (in.probes && !in.probes->empty()) {
                in.probes->sample(center, pc.sh, pc.sun.x);
            } else {
                for (int ch = 0; ch < 3; ++ch) pc.sh[ch] = in.ambientSh[ch];
            }
        }
    }
}

uint32_t ParticleRenderer::queueDraws(RenderQueue& queue) const {
    if (m_drawCount == 0) return 0;
    const uint32_t slot = queue.reserve(m_drawCount);
    if (slot == RenderQueue::kFull) return 0; // counted by the queue
    for (uint32_t i = 0; i < m_drawCount; ++i) {
        DrawItem item;
        item.flags = DrawItem::kParticles;
        item.perDrawOrInstance = i;
        item.instanceCount = m_draws[i].count;
        item.objectIndex = m_draws[i].emitter;
        // Pipeline field 0xFFF: at an equal depth an emitter draws after the transparent meshes.
        queue.set(slot + i, sortkey::blended(QueuePass::Transparent, 1, m_draws[i].depth, 0xFFF, uint16_t(i)), item);
    }
    return m_drawCount;
}

void ParticleRenderer::record(rhi::CommandList& cmd, uint32_t draw, rhi::ResourceSetHandle frameSet,
                              std::span<const uint32_t> frameOffsets, rhi::ResourceSetHandle softSet, bool softPass) const {
    if (draw >= m_drawCount) return;
    const Draw& d = m_draws[draw];
    const bool soft = d.soft && softPass && softSet;
    const rhi::PipelineHandle pipeline = m_pipelines[d.blend][soft ? 1 : 0];
    if (!pipeline) return;
    cmd.bindPipeline(pipeline);
    cmd.bindResourceSet(0, frameSet, frameOffsets);
    if (soft) cmd.bindResourceSet(1, softSet);
    cmd.bindResourceSet(2, d.atlas);
    cmd.setPerDraw(&d.constants, uint8_t(sizeof(d.constants)));
    cmd.bindVertexStreams(0, {&d.instances, 1});
    cmd.draw(6, d.count);
}

void ParticleRenderer::recordOverdraw(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const {
    if (m_drawCount == 0 || !m_overdraw) return;
    cmd.bindPipeline(m_overdraw);
    cmd.bindResourceSet(0, frameSet, frameOffsets);
    for (uint32_t i = 0; i < m_drawCount; ++i) {
        const Draw& d = m_draws[i];
        cmd.bindResourceSet(2, d.atlas);
        cmd.setPerDraw(&d.constants, uint8_t(sizeof(d.constants)));
        cmd.bindVertexStreams(0, {&d.instances, 1});
        cmd.draw(6, d.count);
    }
}

} // namespace renderer
