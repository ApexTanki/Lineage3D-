#include "renderer/lighting/Environment.h"

#include "core/Log.h"

#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <cmath>

namespace renderer {
namespace {

rhi::TextureHandle solidArray(rhi::Device& device, const char* name, uint8_t value) {
    rhi::TextureDesc td;
    td.debugName = name;
    td.type = rhi::TextureType::Tex2DArray;
    td.format = rhi::Format::RGBA8Unorm;
    const rhi::TextureHandle t = device.createTexture(td);
    const uint8_t pixel[4] = {value, value, value, value};
    rhi::TextureUpload u;
    u.width = u.height = 1;
    u.data = std::as_bytes(std::span(pixel));
    if (t && !device.uploadTexture(t, u).ok()) {
        device.destroy(t);
        return {};
    }
    return t;
}

void rgbmBytes(const glm::vec3& color, uint8_t out[4]) { // same encoding as fl_assetc (HdrCodec.h)
    const glm::vec3 c = glm::clamp(color, glm::vec3(0.0f), glm::vec3(8.0f));
    const float m = std::max(c.r, std::max(c.g, c.b)) / 8.0f;
    const int a = std::clamp(int(std::ceil(m * 255.0f)), 1, 255);
    const float scale = 255.0f / (float(a) / 255.0f * 8.0f);
    for (int i = 0; i < 3; ++i) out[i] = uint8_t(std::clamp(std::lround(c[i] * scale), 0l, 255l));
    out[3] = uint8_t(a);
}

} // namespace

core::Result<void> Environment::init(const Desc& desc) {
    m_device = desc.device;
    m_resources = desc.resources;
    m_rgbm = !m_device->caps().bptc;
    rhi::SamplerDesc sd;
    sd.filter = rhi::Filter::Linear;
    sd.mip = rhi::MipMode::Linear;
    sd.address = rhi::AddressMode::ClampToEdge;
    m_cubeSampler = m_device->createSampler(sd);
    sd.mip = rhi::MipMode::Nearest; // lightmaps have one mip
    m_lightmapSampler = m_device->createSampler(sd);
    rhi::ResourceSetLayoutDesc ld;
    ld.set = 3;
    ld.count = 1;
    ld.bindings[0] = {1, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment, rhi::TextureType::Cube};
    m_reflectionLayout = m_device->createResourceSetLayout(ld);
    // Skinned pipelines (M6 plan D9): the bone palette at binding 0, read by the vertex shader at a dynamic offset.
    ld.count = 2;
    ld.bindings[0] = {0, rhi::BindingKind::UniformBufferDynamic, rhi::ShaderStageMask::Vertex};
    ld.bindings[1] = {1, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment, rhi::TextureType::Cube};
    m_skinnedLayout = m_device->createResourceSetLayout(ld);
    ld.count = 1;
    m_paletteLayout = m_device->createResourceSetLayout(ld);
    if (m_paletteLayout) {
        const rhi::ResourceWrite palette{0, rhi::BufferSlice{m_device->transient(rhi::TransientKind::Uniform).buffer(), 0, kPaletteBytes}, {}, {}};
        rhi::ResourceSetDesc paletteDesc;
        paletteDesc.debugName = "palette";
        paletteDesc.layout = m_paletteLayout;
        paletteDesc.writes = {&palette, 1};
        m_paletteSet = m_device->createResourceSet(paletteDesc);
    }
    m_defaultLightmap = solidArray(*m_device, "DefaultLightmap", 0);     // no baked light (RGBM 0 decodes to 0 too)
    m_defaultShadowmask = solidArray(*m_device, "DefaultShadowmask", 255); // the sun is visible
    if (!m_cubeSampler || !m_lightmapSampler || !m_reflectionLayout || !m_skinnedLayout || !m_paletteSet || !m_defaultLightmap ||
        !m_defaultShadowmask)
        return core::Error::InitFailed;
    setAmbient(glm::vec3(0.3f), glm::vec3(0.1f));
    return m_ambientCube ? core::Result<void>() : core::Result<void>(core::Error::InitFailed);
}

void Environment::shutdown() {
    if (!m_device) return;
    releaseSlots();
    for (rhi::ResourceSetHandle& s : m_sets) {
        m_device->destroy(s);
        s = {};
    }
    for (rhi::ResourceSetHandle& s : m_skinnedSets) {
        m_device->destroy(s);
        s = {};
    }
    m_device->destroy(m_paletteSet);
    m_device->destroy(m_paletteLayout);
    m_device->destroy(m_skinnedLayout);
    m_device->destroy(m_ambientCube);
    m_device->destroy(m_defaultLightmap);
    m_device->destroy(m_defaultShadowmask);
    m_device->destroy(m_reflectionLayout);
    m_device->destroy(m_cubeSampler);
    m_device->destroy(m_lightmapSampler);
    m_device = nullptr;
}

rhi::TextureHandle Environment::makeAmbientCube(const glm::vec3& sky, const glm::vec3& ground) const {
    rhi::TextureDesc td;
    td.debugName = "AmbientCube";
    td.type = rhi::TextureType::Cube;
    td.format = m_rgbm ? rhi::Format::RGBA8Unorm : rhi::Format::RGBA16F;
    td.layers = 6;
    const rhi::TextureHandle t = m_device->createTexture(td);
    if (!t) return {};
    const glm::vec3 side = (sky + ground) * 0.5f;
    const glm::vec3 faces[6] = {side, side, sky, ground, side, side}; // +X −X +Y −Y +Z −Z
    for (uint32_t f = 0; f < 6; ++f) {
        rhi::TextureUpload u;
        u.layer = f;
        u.width = u.height = 1;
        uint8_t rgbm[4];
        uint16_t half[4];
        if (m_rgbm) {
            rgbmBytes(faces[f], rgbm);
            u.data = std::as_bytes(std::span(rgbm));
        } else {
            for (int c = 0; c < 3; ++c) half[c] = glm::packHalf1x16(faces[f][c]);
            half[3] = glm::packHalf1x16(1.0f);
            u.data = std::as_bytes(std::span(half));
        }
        if (!m_device->uploadTexture(t, u).ok()) {
            m_device->destroy(t);
            return {};
        }
    }
    return t;
}

void Environment::setAmbient(const glm::vec3& sky, const glm::vec3& ground) {
    if (sky == m_ambientSky && ground == m_ambientGround && m_ambientCube) return; // a settings change, never per frame
    const rhi::TextureHandle cube = makeAmbientCube(sky, ground);
    if (!cube) {
        FL_LOG_ERROR("The ambient reflection cube could not be created");
        return;
    }
    m_device->destroy(m_ambientCube); // deferred: frames in flight keep the old one
    m_ambientCube = cube;
    m_ambientSky = sky;
    m_ambientGround = ground;
    rebuildSets();
}

void Environment::request(Slot& slot, const std::string& path) {
    slot.asset = path.empty() ? TextureAsset{} : m_resources->loadTexture(path);
    slot.resolved = !slot.asset;
}

rhi::TextureHandle Environment::ready(const Slot& slot) const {
    return slot.asset ? m_resources->texture(slot.asset) : rhi::TextureHandle{};
}

void Environment::releaseSlots() {
    for (Slot* s : {&m_lightmaps, &m_shadowmask, &m_sky}) {
        if (s->asset) m_resources->release(s->asset);
        *s = Slot{};
    }
    for (Slot& s : m_reflections) {
        if (s.asset) m_resources->release(s.asset);
        s = Slot{};
    }
}

void Environment::load(const assets::MapManifest& manifest, assets::ProbeGridData probes) {
    if (!m_device) return; // shut down
    releaseSlots();
    m_hasMap = true;
    // The device decides which encoding is sampled (D5); both files hold the same light.
    request(m_lightmaps, m_rgbm ? manifest.lightmapsRgbm : manifest.lightmaps);
    request(m_shadowmask, manifest.shadowmask);
    request(m_sky, m_rgbm ? manifest.skyCubeRgbm : manifest.skyCube);
    m_probeCount = uint32_t(std::min<size_t>(manifest.reflectionProbes.size(), kReflectionSlots - 1));
    for (uint32_t i = 0; i < m_probeCount; ++i) {
        const assets::MapReflectionProbe& p = manifest.reflectionProbes[i];
        request(m_reflections[i + 1], m_rgbm ? p.cubeRgbm : p.cube);
        m_boxes[i + 1] = {p.boxMin, p.boxMax, p.position};
    }
    m_lightmapPages = manifest.lightmapPages;
    if (probes.probes.empty()) m_probes.clear();
    else m_probes.set(std::move(probes));
    rebuildSets();
}

void Environment::unload() {
    if (!m_device) return; // shut down
    releaseSlots();
    m_hasMap = false;
    m_probeCount = 0;
    m_lightmapPages = 0;
    m_probes.clear();
    rebuildSets();
}

void Environment::update() {
    bool changed = false;
    auto poll = [&](Slot& slot) {
        if (slot.resolved) return;
        const LoadState state = m_resources->state(slot.asset);
        if (state == LoadState::Loading) return;
        slot.resolved = true;
        changed = true;
        if (state == LoadState::Failed) FL_LOG_ERROR("A baked lighting texture failed to load; its default stays in place");
    };
    poll(m_lightmaps);
    poll(m_shadowmask);
    poll(m_sky);
    for (uint32_t i = 1; i <= m_probeCount; ++i) poll(m_reflections[i]);
    // A hot reload replaced a texture (M9 plan D9), perhaps one of the map's (lightmaps, shadowmask, sky, a reflection
    // cube): the reflection sets are made again, and revision() makes the renderer remake the frame set, so no set keeps
    // the old texture, which is destroyed once the frames in flight are done with it.
    if (m_resources->textureRevision() != m_seenTextureRevision) {
        m_seenTextureRevision = m_resources->textureRevision();
        changed = true;
    }
    if (changed) rebuildSets();
}

rhi::TextureHandle Environment::skyCube() const {
    const rhi::TextureHandle t = ready(m_sky);
    return t ? t : m_ambientCube;
}

rhi::TextureHandle Environment::lightmaps() const {
    const rhi::TextureHandle t = ready(m_lightmaps);
    return t ? t : m_defaultLightmap;
}

rhi::TextureHandle Environment::shadowmask() const {
    const rhi::TextureHandle t = ready(m_shadowmask);
    return t ? t : m_defaultShadowmask;
}

bool Environment::hasSky() const {
    return bool(ready(m_sky));
}

uint32_t Environment::pendingCount() const {
    uint32_t n = 0;
    for (const Slot* s : {&m_lightmaps, &m_shadowmask, &m_sky}) n += s->resolved ? 0u : 1u;
    for (uint32_t i = 1; i <= m_probeCount; ++i) n += m_reflections[i].resolved ? 0u : 1u;
    return n;
}

// Loading-time only (a texture arrived, a map changed): one set per reflection slot, the old ones deferred.
void Environment::rebuildSets() {
    for (uint32_t slot = 0; slot < kReflectionSlots; ++slot) {
        rhi::TextureHandle cube = slot == 0 ? skyCube() : ready(m_reflections[slot]);
        if (slot > 0 && slot > m_probeCount) cube = {};
        if (slot > 0 && !cube && slot <= m_probeCount) cube = skyCube(); // until the probe arrives, the sky stands in
        m_device->destroy(m_sets[slot]);
        m_device->destroy(m_skinnedSets[slot]);
        m_sets[slot] = m_skinnedSets[slot] = {};
        if (!cube) continue;
        const rhi::ResourceWrite write{1, {}, cube, m_cubeSampler};
        rhi::ResourceSetDesc sd;
        sd.debugName = "reflection";
        sd.layout = m_reflectionLayout;
        sd.writes = {&write, 1};
        m_sets[slot] = m_device->createResourceSet(sd);
        if (!m_sets[slot]) FL_LOG_ERROR("Reflection set %u could not be created", slot);
        // The skinned twin (M6 plan D9): the same cube plus the bone palette in the uniform ring.
        const rhi::ResourceWrite skinnedWrites[2] = {
            {0, rhi::BufferSlice{m_device->transient(rhi::TransientKind::Uniform).buffer(), 0, kPaletteBytes}, {}, {}}, write};
        sd.debugName = "reflection (skinned)";
        sd.layout = m_skinnedLayout;
        sd.writes = skinnedWrites;
        m_skinnedSets[slot] = m_device->createResourceSet(sd);
        if (!m_skinnedSets[slot]) FL_LOG_ERROR("Skinned reflection set %u could not be created", slot);
    }
    ++m_revision;
}

uint8_t Environment::selectReflection(const glm::vec3& point) const {
    uint8_t best = 0;
    float bestVolume = INFINITY;
    for (uint32_t i = 1; i <= m_probeCount; ++i) {
        const Box& b = m_boxes[i];
        if (point.x < b.min.x || point.y < b.min.y || point.z < b.min.z || point.x > b.max.x || point.y > b.max.y || point.z > b.max.z)
            continue;
        const glm::vec3 d = b.max - b.min;
        if (d.x * d.y * d.z < bestVolume) {
            bestVolume = d.x * d.y * d.z;
            best = uint8_t(i);
        }
    }
    return best;
}

void Environment::fillFrame(FrameUniforms& fu) const {
    fu.outputParams.w = m_rgbm ? 1.0f : 0.0f;
    const uint32_t skyMips = ready(m_sky) ? m_resources->textureMips(m_sky.asset) : 1u;
    fu.environment = glm::vec4(float(std::max(1u, skyMips) - 1), hasSky() ? 1.0f : 0.0f, 0.0f, 0.0f);
    for (uint32_t slot = 0; slot < kReflectionSlots; ++slot) {
        glm::vec4* p = &fu.reflectionProbes[slot * 3];
        const rhi::TextureHandle cube = slot == 0 ? rhi::TextureHandle{} : ready(m_reflections[slot]);
        if (slot == 0 || !cube || slot > m_probeCount) {
            // The sky (or a probe whose cube hasn't arrived, which binds the sky): no box projection.
            p[0] = glm::vec4(0.0f);
            p[1] = glm::vec4(0.0f, 0.0f, 0.0f, fu.environment.x);
            p[2] = glm::vec4(0.0f);
            continue;
        }
        const uint32_t mips = std::max(1u, m_resources->textureMips(m_reflections[slot].asset));
        p[0] = glm::vec4(m_boxes[slot].min, 1.0f);
        p[1] = glm::vec4(m_boxes[slot].max, float(mips - 1));
        p[2] = glm::vec4(m_boxes[slot].position, 0.0f);
    }
}

} // namespace renderer
