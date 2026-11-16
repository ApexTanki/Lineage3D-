#include "renderer/shadows/ShadowMaps.h"

#include "core/Log.h"
#include "core/Profile.h"
#include "renderer/core/GpuTimers.h"
#include "renderer/core/ShaderLibrary.h"
#include "renderer/materials/MaterialSystem.h"
#include "renderer/resources/ResourceManager.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace renderer {
namespace {

// Rasterizer bias for every caster pipeline (D16 units; slope × the polygon's depth slope). The constant part varies
// per cascade through the pass uniforms instead, because pipeline bias is static.
constexpr float kSlopeBias = 2.0f;
constexpr float kConstantBias = 1.0f;
constexpr float kCascadeDepthOffsetTexels = 0.25f; // per-cascade constant offset, in texel sizes of depth
constexpr float kPenumbraMeters = 0.06f;           // Poisson radius in world units (clamped to 1–2.5 texels)

} // namespace

core::Result<void> ShadowMaps::init(rhi::Device& device, ShaderLibrary& shaders, const MaterialSystem& materials,
                                     rhi::ResourceSetLayoutHandle paletteLayout, rhi::ResourceSetHandle paletteSet) {
    m_device = &device;
    m_paletteLayout = paletteLayout;
    m_paletteSet = paletteSet;
    rhi::ResourceSetLayoutDesc ld;
    ld.set = 1;
    ld.count = 1;
    ld.bindings[0] = {0, rhi::BindingKind::UniformBufferDynamic, rhi::ShaderStageMask::Vertex};
    m_passLayout = device.createResourceSetLayout(ld);
    if (!m_passLayout) return core::Error::InitFailed;
    const rhi::ResourceWrite write{0, rhi::BufferSlice{device.transient(rhi::TransientKind::Uniform).buffer(), 0,
                                                       uint32_t(sizeof(ShadowPassUniforms))},
                                   {}, {}};
    rhi::ResourceSetDesc sd;
    sd.debugName = "shadow pass";
    sd.layout = m_passLayout;
    sd.writes = {&write, 1};
    m_passSet = device.createResourceSet(sd);

    // Hardware 2×2 PCF where depth formats filter linearly; nearest otherwise (the shader then blends 4 taps, P16).
    // fl_shadows.glsl samples the cascades with implicit LOD, which is exact only while the maps keep one level and this
    // sampler one filter without anisotropy (M9 performance pass).
    rhi::SamplerDesc smp;
    smp.filter = device.caps().linearDepthCompare ? rhi::Filter::Linear : rhi::Filter::Nearest;
    smp.mip = rhi::MipMode::Nearest;
    smp.address = rhi::AddressMode::ClampToEdge;
    smp.compare = rhi::CompareOp::LessOrEqual;
    m_sampler = device.createSampler(smp);
    if (!m_passSet || !m_sampler) return core::Error::InitFailed;

    for (uint32_t i = 0; i < 6; ++i) {
        m_pipelines[i] = createPipeline(shaders, materials, i >> 1, (i & 1) != 0);
        if (!m_pipelines[i]) return core::Error::CompileFailed;
    }
    m_clipSpace = ClipSpace(device.caps().clipSpace);
    m_nearClipZ = m_clipSpace.nearClipZ();
    m_ndcDepthScale = m_clipSpace.depthToNdc().x;
    m_spotCandidates.reserve(1024);
    m_settings = ShadowSettings::low(); // configure() decides the real textures on the first frame
    if (!createTextures()) return core::Error::OutOfMemory;
    return {};
}

void ShadowMaps::shutdown() {
    if (!m_device) return;
    for (rhi::PipelineHandle& p : m_pipelines) m_device->destroy(p);
    m_device->destroy(m_cascades);
    m_device->destroy(m_spotAtlas);
    m_device->destroy(m_passSet);
    m_device->destroy(m_passLayout);
    m_device->destroy(m_sampler);
    m_device = nullptr;
}

// variant: the OBJECT value (0 dynamic, 1 instanced, 2 skinned; shadow.perm.json order).
rhi::PipelineHandle ShadowMaps::createPipeline(ShaderLibrary& shaders, const MaterialSystem& materials, uint32_t variant,
                                               bool alphaTest) {
    auto alpha = shaders.axis("shadow", "ALPHA_TEST");
    auto object = shaders.axis("shadow", "OBJECT");
    if (!alpha || !object) return {};
    const bool instanced = variant == 1, skinned = variant == 2;
    const uint64_t key = (alphaTest ? 1ull << alpha.value().bitOffset : 0ull) | (uint64_t(variant) << object.value().bitOffset);
    auto program = shaders.program("shadow", key);
    if (!program) return {};
    rhi::PipelineDesc pd;
    pd.debugName = "shadow caster";
    pd.vertex = program.value().vertex;
    pd.fragment = program.value().fragment;
    if (!fixedVertexLayout(*program.value().reflection, pd.vertexLayout)) return {};
    // Shadow views have instance data of their own: the rows only (M5 plan D8), which the shader reads at offsets 0–47.
    if (instanced) pd.vertexLayout.streams[2].stride = uint16_t(sizeof(InstanceRows));
    // No culling: the baker treats every face as an occluder from both sides (a single-sided plane casts both ways), so
    // the cascades that replace the baked sun visibility on High must too. Winding (mirrored instances) then doesn't matter.
    pd.raster.cull = rhi::CullMode::None;
    pd.raster.depthBiasConstant = kConstantBias;
    pd.raster.depthBiasSlope = kSlopeBias;
    pd.depth = {true, true, rhi::CompareOp::LessOrEqual};
    pd.attachments.colorCount = 0;
    pd.attachments.depth = rhi::Format::D16;
    pd.attachments.samples = 1;
    pd.setLayouts[1] = m_passLayout;
    pd.setLayouts[2] = materials.materialLayout();
    if (skinned) pd.setLayouts[3] = m_paletteLayout;
    pd.perDrawBytes = uint8_t(sizeof(InstanceRows));
    const rhi::PipelineHandle pipeline = m_device->createPipeline(pd);
    if (!pipeline) FL_LOG_ERROR("A shadow caster pipeline could not be created");
    return pipeline;
}

// The maps for m_settings, made before the old ones go (M9 plan D6): if either doesn't fit (over the render-target
// budget, out of memory), the old maps and their sizes stay and false comes back.
bool ShadowMaps::createTextures() {
    const uint32_t mapSize = m_settings.cascades ? std::clamp<uint32_t>(m_settings.mapSize, 256u, m_device->caps().maxTextureSize) : 16u;
    const uint32_t layers = std::max<uint32_t>(m_settings.cascades, 1u);
    const bool spotEnabled = m_settings.spotShadows;
    const uint32_t spotAtlasSize = spotEnabled ? std::min<uint32_t>(kSpotAtlasSize, m_device->caps().maxTextureSize) : 16u;
    rhi::TextureDesc td;
    td.debugName = "shadowCascades";
    td.type = rhi::TextureType::Tex2DArray;
    td.format = rhi::Format::D16;
    td.width = td.height = mapSize;
    td.layers = uint16_t(layers);
    td.usage = rhi::TextureUsage::DepthTarget | rhi::TextureUsage::Sampled;
    const rhi::TextureHandle cascades = m_device->createTexture(td);
    td.debugName = "spotShadowAtlas";
    td.type = rhi::TextureType::Tex2D;
    td.layers = 1;
    td.width = td.height = spotAtlasSize;
    const rhi::TextureHandle spotAtlas = cascades ? m_device->createTexture(td) : rhi::TextureHandle{};
    if (!cascades || !spotAtlas) {
        m_device->destroy(cascades);
        FL_LOG_ERROR("Shadow maps (%u² × %u, spot atlas %u²) could not be created", mapSize, layers, spotAtlasSize);
        return false;
    }
    m_device->destroy(m_cascades); // deferred: frames in flight keep the old ones
    m_device->destroy(m_spotAtlas);
    m_cascades = cascades;
    m_spotAtlas = spotAtlas;
    m_mapSize = mapSize;
    m_layers = layers;
    m_spotEnabled = spotEnabled;
    m_spotAtlasSize = spotAtlasSize;
    m_spotTileSize = m_spotAtlasSize / 2;
    m_cascadesNeedClear = m_spotNeedsClear = true;
    ++m_revision;
    return true;
}

void ShadowMaps::configure(const ShadowSettings& settings) {
    const ShadowSettings old = m_settings;
    m_settings = settings;
    m_settings.cascades = uint8_t(std::min<uint32_t>(settings.cascades, kMaxCascades));
    const uint32_t size = m_settings.cascades ? std::clamp<uint32_t>(m_settings.mapSize, 256u, m_device->caps().maxTextureSize) : 16u;
    const bool texturesChange = size != m_mapSize || std::max<uint32_t>(m_settings.cascades, 1u) != m_layers ||
                                m_settings.spotShadows != m_spotEnabled;
    if (!texturesChange) {
        m_hasRefused = false;
        return;
    }
    // A request whose maps didn't fit isn't tried again every frame: the previous shadow settings stay in force until the
    // request changes (as for render targets, M9 plan D6).
    const bool refusedBefore = m_hasRefused && m_refused.cascades == settings.cascades && m_refused.mapSize == settings.mapSize &&
                               m_refused.spotShadows == settings.spotShadows;
    if (refusedBefore || !createTextures()) {
        if (!refusedBefore) FL_LOG_WARN("The shadow maps of the requested settings don't fit: the previous shadow settings stay in force");
        m_refused = settings;
        m_hasRefused = true;
        m_settings = old;
        return;
    }
    m_hasRefused = false;
}

void ShadowMaps::assignSpotShadows(std::span<const LightDesc* const> lights, const glm::vec3& cameraPosition, float time,
                                   std::span<int8_t> slots) {
    std::fill(slots.begin(), slots.end(), int8_t(-1));
    m_spotCount = 0;
    if (!m_settings.spotShadows || !m_spotEnabled) {
        std::fill(std::begin(m_previousSpots), std::end(m_previousSpots), nullptr);
        return;
    }
    m_spotCandidates.clear();
    for (const LightDesc* l : lights) {
        if (l->type != LightType::Spot || !l->castsShadows || l->range <= 0.0f) continue;
        if (l->outerConeDeg > kMaxSpotShadowConeDeg) {
            if (!m_warnedWideSpot) {
                FL_LOG_WARN("A shadow-casting spot light has a %.0f° outer cone; spot shadows cover at most %.0f°, so it casts none",
                            double(l->outerConeDeg), double(kMaxSpotShadowConeDeg));
                m_warnedWideSpot = true;
            }
            continue;
        }
        m_spotCandidates.push_back(l);
    }
    // Last frame's winners keep their tiles unless another light outranks them by more than 25 %, so two lights of nearly
    // equal rank (headlights passing each other) don't trade the shadow back and forth every frame.
    auto score = [&](const LightDesc* l) {
        const glm::vec3 c = l->color * l->intensity;
        float s = spotShadowScore(0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b, l->range, l->position, cameraPosition);
        if (std::find(std::begin(m_previousSpots), std::end(m_previousSpots), l) != std::end(m_previousSpots)) s *= 1.25f;
        return s;
    };
    const size_t n = std::min<size_t>(m_spotCandidates.size(), kMaxSpotShadows);
    std::partial_sort(m_spotCandidates.begin(), m_spotCandidates.begin() + ptrdiff_t(n), m_spotCandidates.end(),
                      [&](const LightDesc* a, const LightDesc* b) { return score(a) > score(b); });
    // The pass uniforms are allocated here, before the lights are uploaded: a light gets a slot only if its tile will
    // be rendered this frame.
    rhi::TransientRing& ring = m_device->transient(rhi::TransientKind::Uniform);
    for (size_t k = 0; k < n; ++k) {
        const LightDesc* l = m_spotCandidates[k];
        Spot& s = m_spots[k];
        s.position = l->position;
        const float len = glm::length(l->direction);
        s.direction = len > 1e-6f ? l->direction / len : glm::vec3(0.0f, -1.0f, 0.0f);
        s.range = l->range;
        s.slot = uint32_t(k);
        s.fit = fitSpotShadow(s.position, s.direction, l->outerConeDeg, l->range, m_spotTileSize);
        s.viewProj = m_clipSpace.fix(s.fit.projection) * s.fit.view;
        ShadowPassUniforms pass;
        pass.viewProj = s.viewProj;
        pass.params = glm::vec4(0.0f, 0.0f, time, m_nearClipZ);
        const rhi::TransientRing::Allocation a = ring.allocate(sizeof(pass));
        if (!a.cpu) break; // out of ring space: the remaining lights go unshadowed this frame
        std::memcpy(a.cpu, &pass, sizeof(pass)); // ring memory is write-only (ARCHITECTURE.md §4.8)
        m_spotPassOffsets[k] = a.gpu.offset;
        for (size_t i = 0; i < lights.size() && i < slots.size(); ++i)
            if (lights[i] == l) slots[i] = int8_t(k);
        m_spotCount = uint32_t(k + 1);
    }
    for (uint32_t k = 0; k < kMaxSpotShadows; ++k) m_previousSpots[k] = k < m_spotCount ? m_spotCandidates[k] : nullptr;
}

void ShadowMaps::prepare(const FrameInput& in, FrameUniforms& fu) {
    const ClipSpace& cs = *in.clipSpace;
    CascadeSetup setup;
    setup.cameraPosition = in.cameraPosition;
    setup.cameraForward = in.cameraForward;
    setup.fovY = in.fovY;
    setup.aspect = in.aspect;
    setup.zNear = in.zNear;
    setup.shadowDistance = std::min(m_settings.distance, in.zFar);
    setup.sunDirection = in.sunDirection;
    setup.cascades = m_settings.cascades;
    setup.mapSize = m_mapSize;
    m_cascadeCount = fitCascades(setup, {m_fits, kMaxCascades});
    if (in.sunDirection.y <= 0.0f) m_cascadeCount = 0; // the sun is down: no direct light to shadow

    const ShadowFilterMode filter = m_cascadeCount == 0                                     ? ShadowFilterMode::Off
                                    : m_settings.filter == ShadowFilter::Poisson16           ? ShadowFilterMode::Poisson16
                                    : m_device->caps().linearDepthCompare ? ShadowFilterMode::HardwareTap
                                                                          : ShadowFilterMode::FourNearest;
    fu.cascadeSplits = glm::vec4(0.0f, 0.0f, 0.0f, float(m_cascadeCount));
    fu.cascadeBands = glm::vec4(0.0f, 0.0f, 0.0f, float(uint32_t(filter)));
    fu.cascadeTexel = glm::vec4(0.0f, 0.0f, 0.0f, float(m_mapSize));
    fu.cascadeFilter = glm::vec4(0.0f, 0.0f, 0.0f, m_settings.staticCasters && m_cascadeCount ? 1.0f : 0.0f);
    fu.cameraForward = glm::vec4(in.cameraForward, 0.0f);
    rhi::TransientRing& ring = m_device->transient(rhi::TransientKind::Uniform);
    for (uint32_t c = 0; c < m_cascadeCount; ++c) {
        const CascadeFit& f = m_fits[c];
        const glm::mat4 viewProj = cs.fix(f.projection) * f.view;
        fu.cascadeMatrix[c] = cs.clipToUvDepth() * viewProj;
        fu.cascadeSplits[int(c)] = f.splitFar;
        fu.cascadeBands[int(c)] = f.band;
        fu.cascadeTexel[int(c)] = f.texelSize;
        fu.cascadeFilter[int(c)] = std::clamp(kPenumbraMeters / f.texelSize, 1.0f, 2.5f) / float(m_mapSize);
        ShadowPassUniforms pass;
        pass.viewProj = viewProj;
        pass.params = glm::vec4(1.0f, kCascadeDepthOffsetTexels * f.texelSize / f.depthRange * m_ndcDepthScale, in.time, m_nearClipZ);
        const rhi::TransientRing::Allocation a = ring.allocate(sizeof(pass));
        if (!a.cpu) { // out of ring space: no cascades this frame rather than stale data
            m_cascadeCount = 0;
            fu.cascadeSplits.w = 0.0f;
            fu.cascadeBands.w = 0.0f;
            break;
        }
        std::memcpy(a.cpu, &pass, sizeof(pass)); // ring memory is write-only (ARCHITECTURE.md §4.8)
        m_passOffsets[c] = a.gpu.offset;
    }

    // Spot tiles (chosen, and their pass uniforms written, by assignSpotShadows()).
    const float atlas = float(m_spotAtlasSize);
    const float tileUv = float(m_spotTileSize) / atlas;
    for (uint32_t k = 0; k < m_spotCount; ++k) {
        const Spot& s = m_spots[k];
        const glm::vec2 tileMin(float(k % 2) * tileUv, float(k / 2) * tileUv);
        glm::mat4 toTile(1.0f); // clip-space (u, v) of the light's frustum → this tile of the atlas
        toTile[0][0] = tileUv;
        toTile[1][1] = tileUv;
        toTile[3][0] = tileMin.x;
        toTile[3][1] = tileMin.y;
        fu.spotMatrix[k] = toTile * cs.clipToUvDepth() * s.viewProj;
        const float inset = 2.0f / atlas; // filter taps stay inside the tile
        fu.spotTile[k] = glm::vec4(tileMin + inset, tileMin + tileUv - inset);
        fu.spotTexel[int(k)] = s.fit.texelAngle;
    }
    m_stats.cascades = m_cascadeCount;
    m_stats.spotShadows = m_spotCount;
    m_stats.mapSize = m_mapSize;
}

uint64_t ShadowMaps::casterKey(uint32_t view, uint32_t pipeline, uint16_t sortId, uint32_t meshRef, uint32_t submesh, uint32_t lod) {
    // view:4 | pipeline:4 | material:16 | mesh:24 | level:2 | submesh:14 — one pass per view, then state order.
    return uint64_t(view & 0xFu) << 60 | uint64_t(pipeline & 0xFu) << 56 | uint64_t(sortId) << 40 |
           uint64_t(meshRef & 0xFFFFFFu) << 16 | uint64_t(lod & 3u) << 14 | uint64_t(submesh & 0x3FFFu);
}

void ShadowMaps::sortCasters() {
    FL_ZONE("ShadowMaps::sortCasters");
    m_queue.sort();
    m_stats.casters = m_queue.size();
    if (m_queue.dropped()) FL_LOG_WARN("Shadow caster queue full: %u caster draws dropped this frame", m_queue.dropped());
}

uint32_t ShadowMaps::drawView(rhi::CommandList& cmd, const Scene& scene, const ResourceManager& resources, rhi::BufferHandle vertexRing,
                              uint32_t passOffset, uint32_t view, uint32_t i) {
    uint32_t boundPipeline = ~0u, boundMesh = ~0u, boundMaterial = 0;
    bool skinBound = false; // stream 2 holds boundMesh's skin stream (instanced runs put their rows there)
    const GpuMesh* mesh = nullptr;
    const ModelAsset* model = nullptr;
    for (; i < m_queue.size() && (m_queue.sortedKey(i) >> 60) == view; ++i) {
        const DrawItem& c = m_queue.sortedItem(i);
        if (c.flags == 0) continue;
        if (c.mesh != boundMesh) {
            const Scene::MeshRef& ref = scene.meshRefs()[c.mesh];
            const ModelAsset* m = resources.model(ref.model);
            if (!m || m->state != LoadState::Ready) continue;
            model = m;
            mesh = &m->meshes[ref.mesh];
        }
        if (c.pipeline != boundPipeline) {
            cmd.bindPipeline(m_pipelines[c.pipeline]);
            if (boundPipeline == ~0u) cmd.bindResourceSet(1, m_passSet, {&passOffset, 1}); // every caster layout is the same
            boundPipeline = c.pipeline;
        }
        if ((c.pipeline & 1u) && c.material != boundMaterial) { // alpha-tested casters sample their material
            cmd.bindResourceSet(2, rhi::ResourceSetHandle{c.material});
            boundMaterial = c.material;
        }
        const bool skinned = (c.flags & DrawItem::kSkinned) != 0;
        if (skinned) { // the renderable's bone palette for this frame (M6 plan D9)
            const uint32_t palette = scene.dynamics()[c.perDrawOrInstance].paletteOffset;
            if (palette == Scene::kNoPalette) continue;
            cmd.bindResourceSet(3, m_paletteSet, {&palette, 1});
        }
        if (c.mesh != boundMesh || (skinned && !skinBound)) {
            const rhi::BufferSlice streams[3] = {mesh->positions, mesh->surface, mesh->skin};
            cmd.bindVertexStreams(0, {streams, size_t(skinned ? 3 : 2)});
            cmd.bindIndexBuffer(mesh->indices, mesh->indexType);
            skinBound = skinned;
            boundMesh = c.mesh;
        }
        const GpuSubmesh& sub = model->submeshes[mesh->firstSubmesh + c.submesh];
        const uint32_t level = std::min<uint32_t>(c.lod & 3u, mesh->lodCount - 1);
        if (c.flags & DrawItem::kInstanced) {
            const rhi::BufferSlice data{vertexRing, c.perDrawOrInstance, c.instanceCount * uint32_t(sizeof(InstanceRows))};
            cmd.bindVertexStreams(2, {&data, 1});
            skinBound = false;
            cmd.drawIndexed(sub.lodCount[level], c.instanceCount, sub.lodFirst[level]);
        } else {
            const InstanceRows& rows = (c.flags & DrawItem::kDynamic) ? scene.dynamics()[c.perDrawOrInstance].rows
                                                                       : scene.staticRows()[c.perDrawOrInstance];
            cmd.setPerDraw(&rows, uint8_t(sizeof(rows)));
            cmd.drawIndexed(sub.lodCount[level], 1, sub.lodFirst[level]);
        }
        ++m_stats.draws;
    }
    return i;
}

void ShadowMaps::record(rhi::CommandList& cmd, const Scene& scene, const ResourceManager& resources, rhi::BufferHandle vertexRing,
                        GpuTimers& timers) {
    FL_ZONE("ShadowMaps::record");
    m_stats.draws = 0;
    // The queue is sorted by view: cascades 0–2, then spot tiles 3–6. Items of a view are skipped past when it isn't drawn.
    uint32_t i = 0;
    auto skipTo = [&](uint32_t view) {
        while (i < m_queue.size() && (m_queue.sortedKey(i) >> 60) < view) ++i;
    };
    if (m_cascadeCount > 0 || m_cascadesNeedClear) {
        const uint32_t timer = timers.beginPass(cmd, "Shadow cascades");
        static constexpr const char* kNames[kMaxCascades] = {"Shadow cascade 0", "Shadow cascade 1", "Shadow cascade 2"};
        for (uint32_t c = 0; c < std::max(m_cascadeCount, m_cascadesNeedClear ? m_layers : 0u); ++c) {
            rhi::RenderPassDesc pass;
            pass.debugName = c < kMaxCascades ? kNames[c] : "Shadow cascade";
            pass.useDepth = true;
            pass.depth.texture = m_cascades;
            pass.depth.layer = uint16_t(c);
            pass.depth.load = rhi::LoadOp::Clear;
            pass.depth.store = rhi::StoreOp::Store;
            pass.depth.clearDepth = 1.0f;
            cmd.beginRenderPass(pass);
            if (c < m_cascadeCount) {
                skipTo(c);
                i = drawView(cmd, scene, resources, vertexRing, m_passOffsets[c], c, i);
            }
            cmd.endRenderPass();
        }
        m_cascadesNeedClear = false;
        timers.endPass(cmd, timer);
    }
    if (m_spotCount > 0 || m_spotNeedsClear) {
        const uint32_t timer = timers.beginPass(cmd, "Spot shadows");
        rhi::RenderPassDesc pass;
        pass.debugName = "Spot shadow atlas";
        pass.useDepth = true;
        pass.depth.texture = m_spotAtlas;
        pass.depth.load = rhi::LoadOp::Clear;
        pass.depth.store = rhi::StoreOp::Store;
        pass.depth.clearDepth = 1.0f;
        cmd.beginRenderPass(pass);
        for (uint32_t k = 0; k < m_spotCount; ++k) {
            const rhi::Rect tile{int32_t((k % 2) * m_spotTileSize), int32_t((k / 2) * m_spotTileSize), m_spotTileSize, m_spotTileSize};
            cmd.setViewport(tile);
            cmd.setScissor(tile);
            skipTo(kMaxCascades + k);
            i = drawView(cmd, scene, resources, vertexRing, m_spotPassOffsets[k], kMaxCascades + k, i);
        }
        cmd.endRenderPass();
        m_spotNeedsClear = false;
        timers.endPass(cmd, timer);
    }
}

} // namespace renderer
