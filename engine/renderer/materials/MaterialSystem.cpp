#include "renderer/materials/MaterialSystem.h"

#include "core/Log.h"
#include "platform/FileSystem.h"
#include "renderer/core/ShaderLibrary.h"
#include "renderer/core/ShaderTypes.h"

#include <cstddef>
#include <cstdio>

namespace renderer {
namespace {

constexpr uint16_t kNoPipeline = 0xFFFF;

template <typename H>
H fromBits(uint32_t bits) {
    H h;
    h.bits = bits;
    return h;
}

// The engine's fixed vertex streams (ARCHITECTURE.md §4.7, §7.2): 0 = positions (8 B), 1 = surface (20 B),
// 2 = per-instance data (InstanceData, 80 B), or the skin stream of skinned variants (8 B). Every shader input location
// has exactly one place.
struct FixedAttribute {
    uint8_t location;
    uint8_t stream;
    rhi::Format format;
    uint16_t offset;
};
constexpr FixedAttribute kFixedAttributes[] = {
    {0, 0, rhi::Format::RGBA16Sint, 0},   // position, int16 grid coordinates
    {1, 1, rhi::Format::RGBA8Snorm, 0},   // normal
    {2, 1, rhi::Format::RGBA8Snorm, 4},   // tangent, w = bitangent sign
    {3, 1, rhi::Format::RG32F, 8},        // uv0
    {4, 1, rhi::Format::RG16Unorm, 16},   // uv1 (lightmaps, M3)
    {8, 2, rhi::Format::RGBA32F, uint16_t(offsetof(InstanceData, rows))},
    {9, 2, rhi::Format::RGBA32F, uint16_t(offsetof(InstanceData, rows) + 16)},
    {10, 2, rhi::Format::RGBA32F, uint16_t(offsetof(InstanceData, rows) + 32)},
    {11, 2, rhi::Format::RGBA32F, uint16_t(offsetof(InstanceData, lightmapST))},
    {12, 2, rhi::Format::RGBA32Uint, uint16_t(offsetof(InstanceData, lights))},
    // Skinned variants (M6 plan D9) are never instanced, so stream 2 carries the skin stream instead (8 B/vertex).
    {5, 2, rhi::Format::RGBA8Uint, 0}, // joints
    {6, 2, rhi::Format::RGBA8Unorm, 4}, // weights
};

// The layout declares exactly the inputs the variant reads, each at its fixed place (M2 plan D3).
bool vertexLayoutForImpl(const shaderformat::Reflection& reflection, rhi::VertexLayout& out) {
    out = rhi::VertexLayout{};
    out.streams[0] = {8, false};
    out.streams[1] = {20, false};
    out.streams[2] = {uint16_t(sizeof(InstanceData)), true};
    out.streamCount = 2;
    for (const shaderformat::VertexInput& in : reflection.vertexInputs) {
        const FixedAttribute* fixed = nullptr;
        for (const FixedAttribute& a : kFixedAttributes)
            if (a.location == in.location) fixed = &a;
        if (!fixed || out.attributeCount >= rhi::kMaxVertexAttributes) return false;
        out.attributes[out.attributeCount++] = {fixed->location, fixed->stream, fixed->format, fixed->offset};
        if (fixed->stream == 2) out.streamCount = 3;
        if (in.location == 5 || in.location == 6) out.streams[2] = {8, false}; // the skin stream
    }
    return true;
}

rhi::SamplerHandle makeSampler(rhi::Device& device, float anisotropy) {
    rhi::SamplerDesc sd;
    sd.filter = rhi::Filter::Linear;
    sd.mip = rhi::MipMode::Linear;
    sd.address = rhi::AddressMode::Repeat;
    sd.maxAnisotropy = anisotropy; // clamped by the backend to caps.maxAnisotropy
    return device.createSampler(sd);
}

bool isBlended(assets::BlendMode mode) {
    return mode == assets::BlendMode::AlphaBlend || mode == assets::BlendMode::Additive || mode == assets::BlendMode::Premultiplied;
}

// Blending happens in the linear HDR target (ARCHITECTURE.md §4.3). Alpha factors only matter for targets with alpha.
rhi::BlendState blendStateFor(assets::BlendMode mode) {
    using F = rhi::BlendFactor;
    rhi::BlendState b;
    switch (mode) {
    case assets::BlendMode::AlphaBlend:
        b = {true, F::SrcAlpha, F::OneMinusSrcAlpha, rhi::BlendOp::Add, F::One, F::OneMinusSrcAlpha, rhi::BlendOp::Add, 0xF};
        break;
    case assets::BlendMode::Additive:
        b = {true, F::SrcAlpha, F::One, rhi::BlendOp::Add, F::Zero, F::One, rhi::BlendOp::Add, 0xF};
        break;
    case assets::BlendMode::Premultiplied:
        b = {true, F::One, F::OneMinusSrcAlpha, rhi::BlendOp::Add, F::One, F::OneMinusSrcAlpha, rhi::BlendOp::Add, 0xF};
        break;
    case assets::BlendMode::Opaque:
    case assets::BlendMode::AlphaTest:
    case assets::BlendMode::Refractive: break; // refractive surfaces replace what is behind them (M8 plan D10)
    }
    return b;
}

QueuePass passFor(assets::BlendMode mode) {
    if (mode == assets::BlendMode::AlphaTest) return QueuePass::AlphaTest;
    if (mode == assets::BlendMode::Refractive) return QueuePass::Refraction; // the start of scene pass B (M8 plan D10)
    return isBlended(mode) ? QueuePass::Transparent : QueuePass::Opaque;
}

} // namespace

bool fixedVertexLayout(const shaderformat::Reflection& reflection, rhi::VertexLayout& out) {
    return vertexLayoutForImpl(reflection, out);
}

core::Result<void> MaterialSystem::init(const Desc& desc) {
    m_device = desc.device;
    m_shaders = desc.shaders;
    m_resources = desc.resources;
    m_frameLayout = desc.frameLayout;
    m_reflectionLayout = desc.reflectionLayout;
    m_skinnedLayout = desc.skinnedLayout;
    m_passLayout = desc.passLayout;

    // Permutation key bits come from the package's axis table, by name (M2 plan D1). OBJECT's values are in
    // ObjectVariant order (lit.perm.json; a unit test checks the manifest).
    struct {
        const char* name;
        uint8_t* bit;
        uint32_t bits;
    } axes[] = {{"NORMAL_MAP", &m_axes.normalMap, 1},     {"SPEC_GLOSS_MAP", &m_axes.specGlossMap, 1},
                {"EMISSION_MAP", &m_axes.emissionMap, 1}, {"ALPHA_TEST", &m_axes.alphaTest, 1},
                {"DETAIL_LAYER", &m_axes.detailLayer, 1}, {"OBJECT", &m_axes.object, 3}};
    for (const auto& a : axes) {
        auto axis = m_shaders->axis("lit", a.name);
        if (!axis) return axis.error();
        if (axis.value().bitCount != a.bits) {
            FL_LOG_ERROR("lit.fsp: axis %s must take %u key bit(s)", a.name, a.bits);
            return core::Error::InvalidData;
        }
        *a.bit = axis.value().bitOffset;
    }

    rhi::ResourceSetLayoutDesc ld;
    ld.set = 2;
    ld.count = 1 + kSlotCount;
    ld.bindings[0] = {0, rhi::BindingKind::UniformBuffer, rhi::ShaderStageMask::All};
    for (uint8_t s = 0; s < kSlotCount; ++s) ld.bindings[1 + s] = {uint8_t(1 + s), rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment};
    m_materialLayout = m_device->createResourceSetLayout(ld);
    m_sampler = makeSampler(*m_device, m_anisotropy);
    if (!m_materialLayout || !m_sampler) return core::Error::InitFailed;
    m_resources->setMaterialCallbacks({acquireCallback, releaseCallback, this});
    return {};
}

void MaterialSystem::shutdown() {
    if (!m_device) return;
    m_resources->setMaterialCallbacks({});
    for (PipelineEntry& e : m_pipelines) m_device->destroy(e.pipeline);
    for (auto& variant : m_prepass)
        for (auto& sided : variant)
            for (auto& mirrored : sided)
                for (rhi::PipelineHandle& p : mirrored) {
                    m_device->destroy(p);
                    p = {};
                }
    m_pipelines.clear();
    m_pipelineIndex.clear();
    m_materials.forEach([&](auto, Material& m) {
        m_device->destroy(m.set);
        m_device->destroy(m.uniforms);
        releaseMaps(m);
    });
    m_device->destroy(m_sampler);
    m_device->destroy(m_materialLayout);
    m_device = nullptr;
}

MaterialHandle MaterialSystem::acquireCallback(void* user, const std::string& path, const assets::MaterialFile& file) {
    return static_cast<MaterialSystem*>(user)->acquire(path, file);
}

void MaterialSystem::releaseCallback(void* user, MaterialHandle material) {
    static_cast<MaterialSystem*>(user)->release(material);
}

MaterialHandle MaterialSystem::load(const std::string& path) {
    if (auto it = m_byPath.find(path); it != m_byPath.end()) {
        const MaterialHandle h = fromBits<MaterialHandle>(it->second);
        if (m_materials.get(h)) {
            addRef(h);
            return h;
        }
    }
    assets::MaterialFile file;
    std::string error;
    auto bytes = platform::readFile(m_resources->dataDir() + path);
    if (!bytes || !assets::parseMaterial(bytes.value(), file, &error)) {
        FL_LOG_ERROR("Material %s: %s", path.c_str(), bytes ? error.c_str() : "cannot read the file");
        file = assets::MaterialFile{};
        file.templateName.clear();
    }
    return acquire(path, file);
}

uint64_t MaterialSystem::shaderKey(const Material& m, ObjectVariant variant, bool fadeTwin) const {
    uint64_t key = uint64_t(variant) << m_axes.object;
    if (m.error) return key; // the error material: no maps, opaque, switches LODs without fading
    if (m.file.features & assets::kFeatureNormalMap) key |= 1ull << m_axes.normalMap;
    if (m.file.features & assets::kFeatureSpecGlossMap) key |= 1ull << m_axes.specGlossMap;
    if (m.file.features & assets::kFeatureEmissionMap) key |= 1ull << m_axes.emissionMap;
    if (m.file.features & assets::kFeatureDetailLayer) key |= 1ull << m_axes.detailLayer;
    if (m.file.blend == assets::BlendMode::AlphaTest || (fadeTwin && m.file.blend == assets::BlendMode::Opaque)) key |= 1ull << m_axes.alphaTest;
    return key;
}

uint16_t MaterialSystem::pipelineSlot(const Material& m, ObjectVariant variant, bool mirrored, bool fadeTwin, bool blendTwin) {
    const assets::BlendMode blend = m.error ? assets::BlendMode::Opaque : blendTwin ? assets::BlendMode::AlphaBlend : m.file.blend;
    const bool doubleSided = !m.error && m.file.doubleSided;
    const uint64_t shader = shaderKey(m, variant, fadeTwin);
    const uint64_t key = shader << 8 | uint64_t(blend) << 2 | uint64_t(doubleSided) << 1 | uint64_t(mirrored);
    if (auto it = m_pipelineIndex.find(key); it != m_pipelineIndex.end()) return it->second;
    if (m_pipelines.size() >= kMaxPipelines) {
        FL_LOG_ERROR("Material %s: the pipeline table is full (%u)", m.path.c_str(), kMaxPipelines);
        return kNoPipeline;
    }
    PipelineEntry e;
    e.shaderKey = shader;
    e.blend = blend;
    e.doubleSided = doubleSided;
    e.mirrored = mirrored;
    e.variant = variant;
    static constexpr const char* kVariantNames[] = {" dynamic", " instanced", " lightmap", " lightmap-instanced", " skinned"};
    char name[160];
    std::snprintf(name, sizeof(name), "lit %s%s%s%s%s%s/%s%s%s", m.error ? "error " : "",
                  (shader >> m_axes.normalMap) & 1 ? "N" : "", (shader >> m_axes.specGlossMap) & 1 ? "S" : "",
                  (shader >> m_axes.emissionMap) & 1 ? "E" : "", (shader >> m_axes.detailLayer) & 1 ? "D" : "",
                  (shader >> m_axes.alphaTest) & 1 ? "A" : "", assets::blendModeName(blend), kVariantNames[uint32_t(variant)],
                  mirrored ? " mirrored" : "");
    e.name = std::string(name) + (doubleSided ? " double-sided" : "");
    if (m_hasTarget) e.pipeline = createPipeline(e);
    const auto slot = uint16_t(m_pipelines.size());
    m_pipelines.push_back(std::move(e));
    m_pipelineIndex.emplace(key, slot);
    return slot;
}

rhi::PipelineHandle MaterialSystem::createPipeline(const PipelineEntry& e) {
    auto program = m_shaders->program("lit", e.shaderKey);
    if (!program) return {}; // logged by the library
    rhi::PipelineDesc pd;
    pd.debugName = e.name.c_str();
    pd.vertex = program.value().vertex;
    pd.fragment = program.value().fragment;
    if (!fixedVertexLayout(*program.value().reflection, pd.vertexLayout)) {
        FL_LOG_ERROR("Pipeline %s: a vertex input has no fixed stream location (ARCHITECTURE.md §4.7)", e.name.c_str());
        return {};
    }
    pd.raster.cull = e.doubleSided ? rhi::CullMode::None : rhi::CullMode::Back;
    pd.raster.frontFaceClockwise = e.mirrored;
    pd.raster.alphaToCoverage = e.blend == assets::BlendMode::AlphaTest && m_target.samples > 1;
    pd.depth = {true, !isBlended(e.blend), rhi::CompareOp::LessOrEqual};
    pd.blend[0] = blendStateFor(e.blend);
    pd.attachments = m_target;
    pd.setLayouts[0] = m_frameLayout;
    pd.setLayouts[1] = m_passLayout; // the forward passes' inputs: linear depth, SSAO, refraction grab (M8 plan D9, D10)
    pd.setLayouts[2] = m_materialLayout;
    pd.setLayouts[3] = e.variant == ObjectVariant::Skinned ? m_skinnedLayout : m_reflectionLayout;
    pd.perDrawBytes = sizeof(PerDraw);
    const rhi::PipelineHandle pipeline = m_device->createPipeline(pd);
    if (!pipeline) FL_LOG_ERROR("Pipeline %s could not be created", e.name.c_str());
    return pipeline;
}

MaterialHandle MaterialSystem::acquire(const std::string& path, const assets::MaterialFile& file) {
    if (auto it = m_byPath.find(path); it != m_byPath.end()) {
        const MaterialHandle h = fromBits<MaterialHandle>(it->second);
        if (Material* m = m_materials.get(h)) {
            ++m->refs;
            return h;
        }
    }
    Material m;
    const bool built = build(m, path, file);

    // Copied before the move: a failed create must still release what this material took.
    const rhi::BufferHandle uniforms = m.uniforms;
    TextureAsset maps[kSlotCount];
    for (uint32_t s = 0; s < kSlotCount; ++s) maps[s] = m.maps[s];
    const MaterialHandle h = built ? m_materials.create(std::move(m)) : MaterialHandle{};
    Material* stored = m_materials.get(h);
    if (!stored) {
        FL_LOG_ERROR("Material %s: out of material slots or GPU memory", path.c_str());
        m_device->destroy(uniforms);
        for (TextureAsset map : maps)
            if (map) m_resources->release(map);
        return {};
    }
    stored->set = createSet(*stored);
    for (MapState state : stored->mapState)
        if (state == MapState::Loading) {
            m_waiting.push_back(h);
            break;
        }
    m_byPath[path] = h.bits;
    return h;
}

bool MaterialSystem::build(Material& m, const std::string& path, const assets::MaterialFile& file) {
    m.path = path;
    m.refs = 1;
    m.file = file;
    m.error = file.templateName != "lit";
    if (m.error) {
        FL_LOG_ERROR("Material %s: %s; rendered as the error material", path.c_str(),
                     file.templateName.empty() ? "it failed to load" : ("unknown template '" + file.templateName + "'").c_str());
    } else {
        // Every permutation is compiled at build time, so a miss means a stale or broken package: never a silent fallback.
        for (ObjectVariant variant :
             {ObjectVariant::Dynamic, ObjectVariant::Instanced, ObjectVariant::Lightmap, ObjectVariant::LightmapInstanced, ObjectVariant::Skinned})
            if (!m_shaders->program("lit", shaderKey(m, variant)) || !m_shaders->program("lit", shaderKey(m, variant, true))) {
                FL_LOG_ERROR("Material %s: lit.fsp lacks its permutation; rendered as the error material", path.c_str());
                m.error = true;
                break;
            }
    }
    if (!m.error) {
        const std::string* paths[kSlotCount] = {&file.albedo, &file.normal, &file.specGloss, &file.emission, &file.detail};
        for (uint32_t s = 0; s < kSlotCount; ++s) {
            if (paths[s]->empty()) continue;
            m.maps[s] = m_resources->loadTexture(*paths[s]);
            const LoadState state = m_resources->state(m.maps[s]);
            m.mapState[s] = state == LoadState::Ready ? MapState::Ready : state == LoadState::Loading ? MapState::Loading : MapState::Failed;
        }
    }
    // Refractive surfaces draw one by one too: with refraction off they blend and must sort (M8 plan D10).
    const bool refractive = !m.error && file.blend == assets::BlendMode::Refractive;
    const bool blended = !m.error && (isBlended(file.blend) || refractive);
    for (uint32_t i = 0; i < kObjectVariants * 2; ++i) {
        // Blended statics are drawn one by one so they sort (M2 plan D6): they never need the instanced variants.
        auto variant = ObjectVariant(i / 2);
        if (blended && variant == ObjectVariant::Instanced) variant = ObjectVariant::Dynamic;
        if (blended && variant == ObjectVariant::LightmapInstanced) variant = ObjectVariant::Lightmap;
        m.pipelineSlots[i] = pipelineSlot(m, variant, (i & 1) != 0);
        // LOD cross-fade twin (M5 plan D7): only opaque materials need a pipeline of their own.
        m.fadeSlots[i] = !m.error && m.file.blend == assets::BlendMode::Opaque ? pipelineSlot(m, variant, (i & 1) != 0, true) : m.pipelineSlots[i];
        m.blendTwinSlots[i] = refractive ? pipelineSlot(m, variant, (i & 1) != 0, false, true) : m.pipelineSlots[i];
    }
    m.uniforms = createUniforms(m);
    return bool(m.uniforms);
}

bool MaterialSystem::reload(const std::string& path) {
    const auto it = m_byPath.find(path);
    if (it == m_byPath.end()) return false;
    const MaterialHandle h = fromBits<MaterialHandle>(it->second);
    Material* m = m_materials.get(h);
    if (!m) return false;
    assets::MaterialFile file;
    std::string error;
    auto bytes = platform::readFile(m_resources->dataDir() + path);
    if (!bytes || !assets::parseMaterial(bytes.value(), file, &error)) {
        FL_LOG_ERROR("Hot reload: material %s: %s; it keeps its old state", path.c_str(), bytes ? error.c_str() : "cannot read the file");
        return false;
    }
    Material fresh;
    if (!build(fresh, path, file)) {
        FL_LOG_ERROR("Hot reload: material %s: out of GPU memory; it keeps its old state", path.c_str());
        m_device->destroy(fresh.uniforms);
        releaseMaps(fresh);
        return false;
    }
    fresh.refs = m->refs;
    m_device->destroy(m->set); // deferred: frames in flight may still use them
    m_device->destroy(m->uniforms);
    releaseMaps(*m); // after build() took the maps it keeps, so a map in both stays loaded
    *m = std::move(fresh);
    m->set = createSet(*m);
    const bool loading = std::find(std::begin(m->mapState), std::end(m->mapState), MapState::Loading) != std::end(m->mapState);
    if (loading && std::find(m_waiting.begin(), m_waiting.end(), h) == m_waiting.end()) m_waiting.push_back(h);
    return true;
}

void MaterialSystem::releaseMaps(Material& m) {
    for (TextureAsset& map : m.maps) {
        if (map) m_resources->release(map);
        map = {};
    }
}

void MaterialSystem::addRef(MaterialHandle material) {
    if (Material* m = m_materials.get(material)) ++m->refs;
}

void MaterialSystem::release(MaterialHandle material) {
    Material* m = m_materials.get(material);
    if (!m || --m->refs > 0) return;
    m_device->destroy(m->set);
    m_device->destroy(m->uniforms);
    releaseMaps(*m);
    if (auto it = m_byPath.find(m->path); it != m_byPath.end() && it->second == material.bits) m_byPath.erase(it);
    for (size_t i = 0; i < m_waiting.size(); ++i)
        if (m_waiting[i] == material) {
            m_waiting[i] = m_waiting.back();
            m_waiting.pop_back();
            break;
        }
    m_materials.destroy(material);
}

rhi::BufferHandle MaterialSystem::createUniforms(const Material& m) const {
    MaterialUniforms u;
    if (m.error) {
        u.baseColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
    } else {
        const assets::MaterialFile& f = m.file;
        u.baseColor = f.baseColor;
        u.specularGloss = glm::vec4(f.specular, f.gloss);
        u.emission = glm::vec4(f.emissionColor, f.emissionIntensity);
        // The cutoff only matters to alpha-tested materials; −1 turns the alpha test off in the fade twin of an opaque one.
        u.uvAnim = glm::vec4(f.uvScroll, glm::radians(f.uvRotate), f.blend == assets::BlendMode::AlphaTest ? f.alphaCutoff : -1.0f);
        u.flipbook = glm::vec4(f.flipbook, 0.0f);
        u.detail = glm::vec4(f.detailScroll, f.detailScale, f.detailStrength);
        u.misc = glm::vec4(f.reflection, f.blend == assets::BlendMode::Additive ? 1.0f : 0.0f, f.normalStrength,
                           f.blend == assets::BlendMode::Premultiplied ? 1.0f : 0.0f);
        // Refractive (M8 plan D10): what passes through, and how far the normal map bends it (0 marks other materials).
        if (f.blend == assets::BlendMode::Refractive) u.refraction = glm::vec4(f.refractionTint, std::max(f.refractionStrength, 1e-4f));
    }
    rhi::BufferDesc bd;
    bd.debugName = m.path.c_str();
    bd.size = sizeof(u);
    bd.usage = rhi::BufferUsage::Uniform;
    bd.memory = rhi::MemoryKind::GpuOnly;
    bd.initialData = &u;
    return m_device->createBuffer(bd);
}

// Per-slot placeholders while a map loads (or after it failed): grey albedo, a flat normal, and black spec/gloss and
// emission so nothing glows or shines that shouldn't. Slots without a map get white, which no variant samples except
// albedo (white × baseColor).
rhi::TextureHandle MaterialSystem::textureFor(const Material& m, uint32_t slot) const {
    if (m.mapState[slot] == MapState::None) return m_resources->whiteTexture();
    if (m.mapState[slot] == MapState::Ready)
        if (const rhi::TextureHandle t = m_resources->texture(m.maps[slot])) return t;
    if (slot == kAlbedo) return m_resources->greyTexture();
    if (slot == kNormal) return m_resources->flatNormalTexture();
    return m_resources->blackTexture();
}

rhi::ResourceSetHandle MaterialSystem::createSet(const Material& m) const {
    rhi::ResourceWrite writes[1 + kSlotCount];
    writes[0] = {0, rhi::BufferSlice{m.uniforms, 0, uint32_t(sizeof(MaterialUniforms))}, {}, {}};
    for (uint32_t s = 0; s < kSlotCount; ++s) {
        m.boundTextures[s] = textureFor(m, s);
        writes[1 + s] = {uint8_t(1 + s), {}, m.boundTextures[s], m_sampler};
    }
    rhi::ResourceSetDesc sd;
    sd.debugName = m.path.c_str();
    sd.layout = m_materialLayout;
    sd.writes = writes;
    const rhi::ResourceSetHandle set = m_device->createResourceSet(sd);
    if (!set) FL_LOG_ERROR("Material %s: resource set creation failed", m.path.c_str());
    return set;
}

void MaterialSystem::update() {
    // Hot reload (M9 plan D9): a reloaded map has a new GPU texture, so the sets made with the old one are made again.
    if (m_resources->textureRevision() != m_seenTextureRevision) {
        m_seenTextureRevision = m_resources->textureRevision();
        m_materials.forEach([&](MaterialHandle, Material& m) {
            bool rebuild = false;
            for (uint32_t s = 0; s < kSlotCount; ++s) {
                // A map whose first load failed may have been fixed and reloaded: it replaces its placeholder now.
                if (m.mapState[s] == MapState::Failed && m_resources->state(m.maps[s]) == LoadState::Ready) m.mapState[s] = MapState::Ready;
                rebuild = rebuild || textureFor(m, s) != m.boundTextures[s];
            }
            if (rebuild) {
                const rhi::ResourceSetHandle old = m.set;
                m.set = createSet(m);
                m_device->destroy(old);
            }
        });
    }
    for (size_t i = 0; i < m_waiting.size();) {
        Material* m = m_materials.get(m_waiting[i]);
        bool changed = false, loading = false;
        for (uint32_t s = 0; m && s < kSlotCount; ++s) {
            if (m->mapState[s] != MapState::Loading) continue;
            const LoadState state = m_resources->state(m->maps[s]);
            if (state == LoadState::Loading) {
                loading = true;
                continue;
            }
            m->mapState[s] = state == LoadState::Ready ? MapState::Ready : MapState::Failed;
            if (state != LoadState::Ready) FL_LOG_ERROR("Material %s: map %u failed to load; keeping its placeholder", m->path.c_str(), s);
            changed = true;
        }
        if (m && changed) {
            const rhi::ResourceSetHandle old = m->set;
            m->set = createSet(*m);
            m_device->destroy(old); // deferred: frames in flight may still use it
        }
        if (m && loading) {
            ++i;
            continue;
        }
        m_waiting[i] = m_waiting.back();
        m_waiting.pop_back();
    }
}

core::Result<void> MaterialSystem::setTarget(const rhi::AttachmentLayout& target) {
    m_target = target;
    m_hasTarget = true;
    bool ok = true;
    for (PipelineEntry& e : m_pipelines) {
        m_device->destroy(e.pipeline);
        e.pipeline = createPipeline(e);
        ok = ok && e.pipeline;
    }
    // The depth prepass (M8 plan D9): depth only, at the scene's depth format and sample count. Sets 0 and 3 as the lit
    // pipelines have them (skinned variants read the bone palette there).
    static constexpr const char* kNames[2][kObjectVariants] = {
        {"prepass dynamic", "prepass instanced", "prepass lightmap", "prepass lightmap-instanced", "prepass skinned"},
        {"prepass dynamic (fade)", "prepass instanced (fade)", "prepass lightmap (fade)", "prepass lightmap-instanced (fade)",
         "prepass skinned (fade)"}};
    auto object = m_shaders->axis("prepass", "OBJECT");
    auto fadeAxis = m_shaders->axis("prepass", "FADE");
    for (uint32_t v = 0; v < kObjectVariants; ++v)
        for (uint32_t sided = 0; sided < 2; ++sided)
            for (uint32_t mirrored = 0; mirrored < 2; ++mirrored)
                for (uint32_t fade = 0; fade < 2; ++fade) {
                    rhi::PipelineHandle& p = m_prepass[v][sided][mirrored][fade];
                    m_device->destroy(p);
                    p = {};
                    if (!object || !fadeAxis) continue;
                    auto program = m_shaders->program("prepass", (uint64_t(v) << object.value().bitOffset) |
                                                                     (uint64_t(fade) << fadeAxis.value().bitOffset));
                    if (!program) continue;
                    rhi::PipelineDesc pd;
                    pd.debugName = kNames[fade][v];
                    pd.vertex = program.value().vertex;
                    pd.fragment = program.value().fragment;
                    if (!fixedVertexLayout(*program.value().reflection, pd.vertexLayout)) continue;
                    pd.raster.cull = sided ? rhi::CullMode::None : rhi::CullMode::Back;
                    pd.raster.frontFaceClockwise = mirrored != 0;
                    pd.depth = {true, true, rhi::CompareOp::LessOrEqual};
                    pd.attachments.colorCount = 0;
                    pd.attachments.depth = target.depth;
                    pd.attachments.samples = target.samples;
                    pd.setLayouts[0] = m_frameLayout;
                    pd.setLayouts[3] = ObjectVariant(v) == ObjectVariant::Skinned ? m_skinnedLayout : m_reflectionLayout;
                    pd.perDrawBytes = sizeof(PerDraw);
                    p = m_device->createPipeline(pd);
                }
    for (const auto& variant : m_prepass)
        for (const auto& sided : variant)
            for (const auto& mirrored : sided)
                for (rhi::PipelineHandle p : mirrored) ok = ok && p;
    return ok ? core::Result<void>() : core::Result<void>(core::Error::CompileFailed);
}

rhi::PipelineHandle MaterialSystem::prepassPipeline(uint16_t slot) const {
    if (slot >= m_pipelines.size()) return {};
    const PipelineEntry& e = m_pipelines[slot];
    if (e.blend != assets::BlendMode::Opaque) return {}; // alpha-tested, blended and refractive surfaces stay out
    // An opaque entry that tests alpha is a LOD fade twin (shaderKey sets the bit only for those): the FADE variant.
    const uint32_t fade = (e.shaderKey >> m_axes.alphaTest) & 1u;
    return m_prepass[uint32_t(e.variant)][e.doubleSided ? 1 : 0][e.mirrored ? 1 : 0][fade];
}

void MaterialSystem::setAnisotropy(float anisotropy) {
    if (anisotropy == m_anisotropy && m_sampler) return;
    const rhi::SamplerHandle old = m_sampler;
    m_sampler = makeSampler(*m_device, anisotropy);
    m_anisotropy = anisotropy;
    m_materials.forEach([&](MaterialHandle, Material& m) {
        const rhi::ResourceSetHandle previous = m.set;
        m.set = createSet(m);
        m_device->destroy(previous);
    });
    m_device->destroy(old);
}

bool MaterialSystem::drawState(MaterialHandle material, ObjectVariant variant, bool mirrored, DrawState& out) const {
    const Material* m = m_materials.get(material);
    if (!m || !m->set) return false;
    const uint16_t slot = m->pipelineSlots[uint32_t(variant) * 2 + (mirrored ? 1 : 0)];
    if (slot >= m_pipelines.size() || !m_pipelines[slot].pipeline) return false;
    out.set = m->set;
    out.pipelineSlot = slot;
    out.sortId = uint16_t(material.index());
    out.pass = m->error ? QueuePass::Opaque : passFor(m->file.blend);
    const uint16_t fade = m->fadeSlots[uint32_t(variant) * 2 + (mirrored ? 1 : 0)];
    const bool twin = fade != slot && fade < m_pipelines.size() && m_pipelines[fade].pipeline;
    out.fadePipelineSlot = twin ? fade : slot;
    out.fadePass = twin ? QueuePass::AlphaTest : out.pass;
    const uint16_t blendTwin = m->blendTwinSlots[uint32_t(variant) * 2 + (mirrored ? 1 : 0)];
    const bool hasBlendTwin = blendTwin != slot && blendTwin < m_pipelines.size() && m_pipelines[blendTwin].pipeline;
    out.blendTwinSlot = hasBlendTwin ? blendTwin : slot;
    out.blendTwinPass = hasBlendTwin ? QueuePass::Transparent : out.pass;
    return true;
}

bool MaterialSystem::casterState(MaterialHandle material, CasterState& out) const {
    const Material* m = m_materials.get(material);
    if (!m || !m->set) return false;
    out.set = m->set;
    out.casts = m->error || (m->file.castShadows && !isBlended(m->file.blend) && m->file.blend != assets::BlendMode::Refractive);
    out.alphaTest = !m->error && m->file.blend == assets::BlendMode::AlphaTest;
    out.sortId = uint16_t(material.index());
    return true;
}

void MaterialSystem::setParams(MaterialHandle material, const assets::MaterialFile& params) {
    Material* m = m_materials.get(material);
    if (!m || m->error) return;
    assets::MaterialFile& f = m->file;
    f.baseColor = params.baseColor;
    f.specular = params.specular;
    f.gloss = params.gloss;
    f.emissionColor = params.emissionColor;
    f.emissionIntensity = params.emissionIntensity;
    f.normalStrength = params.normalStrength;
    f.alphaCutoff = params.alphaCutoff;
    f.reflection = params.reflection;
    f.uvScroll = params.uvScroll;
    f.uvRotate = params.uvRotate;
    f.flipbook = params.flipbook;
    f.detailScale = params.detailScale;
    f.detailScroll = params.detailScroll;
    f.detailStrength = params.detailStrength;
    const rhi::BufferHandle uniforms = createUniforms(*m);
    if (!uniforms) {
        FL_LOG_ERROR("Material %s: the edited uniforms could not be created", m->path.c_str());
        return;
    }
    m_device->destroy(m->uniforms); // deferred: frames in flight keep reading the old buffer and set
    m->uniforms = uniforms;
    const rhi::ResourceSetHandle old = m->set;
    m->set = createSet(*m);
    m_device->destroy(old);
}

const assets::MaterialFile* MaterialSystem::file(MaterialHandle material) const {
    const Material* m = m_materials.get(material);
    return m ? &m->file : nullptr;
}

const std::string* MaterialSystem::path(MaterialHandle material) const {
    const Material* m = m_materials.get(material);
    return m ? &m->path : nullptr;
}

assets::BlendMode MaterialSystem::blendMode(MaterialHandle material) const {
    const Material* m = m_materials.get(material);
    return m && !m->error ? m->file.blend : assets::BlendMode::Opaque;
}

} // namespace renderer
