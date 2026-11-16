#include "world/RenderWorld.h"

#include "core/Log.h"
#include "platform/FileSystem.h"

namespace world {

core::Result<void> RenderWorld::loadMap(const std::string& mapDirectory) {
    unloadMap();
    renderer::ResourceManager& resources = m_renderer.resources();
    const std::string base = mapDirectory.empty() || mapDirectory.back() == '/' ? mapDirectory : mapDirectory + "/";
    auto bytes = platform::readFile(resources.dataDir() + base + "map.json");
    if (!bytes) {
        FL_LOG_ERROR("Map %s: cannot read map.json (is it baked? the build cooks maps/ into the data folder)", mapDirectory.c_str());
        return core::Error::NotFound;
    }
    std::string error;
    assets::MapManifest manifest;
    if (!assets::parseMapManifest(bytes.value(), manifest, &error)) {
        FL_LOG_ERROR("Map %s: %s", mapDirectory.c_str(), error.c_str());
        return core::Error::InvalidData;
    }
    assets::ProbeGridData probes;
    if (!manifest.probes.empty()) {
        auto probeBytes = platform::readFile(resources.dataDir() + manifest.probes);
        if (!probeBytes || !assets::readProbeGrid(probeBytes.value(), probes, &error)) {
            FL_LOG_ERROR("Map %s: probe grid: %s; objects fall back to the ambient", mapDirectory.c_str(),
                         probeBytes ? error.c_str() : "cannot read the file");
            probes = assets::ProbeGridData{};
        }
    }

    m_renderer.environment().load(manifest, std::move(probes));
    m_lightingBeforeMap = m_renderer.lighting(); // the map's sun, ambient, fog and sky last until it is unloaded
    renderer::LightingData lighting = m_renderer.lighting();
    lighting.sunDirection = manifest.sunDirection;
    lighting.sunColor = manifest.sunColor;
    lighting.ambientSky = manifest.ambientSky;
    lighting.ambientGround = manifest.ambientGround;
    lighting.fog = manifest.fog;
    lighting.skyIntensity = manifest.skyIntensity;
    lighting.sunDiscIntensity = manifest.sunDiscIntensity;
    lighting.sunDiscRadiusDeg = manifest.sunDiscRadiusDeg;
    lighting.exposure = manifest.exposure; // R8: a per-map value (M8)
    lighting.clearColor = manifest.fog.color; // only seen before the sky arrives
    m_renderer.setLighting(lighting);

    std::vector<renderer::MaterialHandle> overrides;
    std::vector<renderer::LightmapPlacement> lightmaps;
    for (const assets::MapInstance& inst : manifest.instances) {
        const renderer::ModelHandle model = resources.loadModel(inst.model);
        overrides.clear();
        for (const std::string& path : inst.materials)
            overrides.push_back(path.empty() ? renderer::MaterialHandle{} : m_renderer.materials().load(path));
        lightmaps.clear();
        for (const assets::MapLightmapRect& r : inst.lightmaps) lightmaps.push_back({r.page, r.scaleOffset});
        renderer::InstanceDesc d;
        d.model = model;
        d.transform = inst.transform;
        d.isStatic = inst.isStatic;
        d.occluder = inst.occluder;
        d.layer = inst.layer;
        d.materialOverrides = overrides;
        d.lightmaps = lightmaps;
        d.reflectionSlot = inst.reflectionProbe;
        const ModelId id = model ? m_renderer.scene().add(d) : ModelId{};
        if (!id) FL_LOG_ERROR("Map %s: instance of %s could not be added", mapDirectory.c_str(), inst.model.c_str());
        m_mapInstances.push_back(id);
        // The scene took its own references.
        if (model) resources.release(model);
        for (renderer::MaterialHandle m : overrides)
            if (m) m_renderer.materials().release(m);
    }
    m_map = std::move(manifest);
    m_hasMap = true;
    FL_LOG_INFO("Map %s: %zu instances, %u lightmap page(s), %zu reflection probe(s), %u probes", mapDirectory.c_str(),
                m_mapInstances.size(), m_map.lightmapPages, m_map.reflectionProbes.size(), m_renderer.environment().probes().count());
    return {};
}

void RenderWorld::unloadMap() {
    if (!m_hasMap) return;
    // After Renderer::shutdown() (a world destroyed late) the renderer has already released everything.
    if (m_renderer.initialized()) {
        for (ModelId id : m_mapInstances)
            if (id) m_renderer.scene().remove(id);
        m_renderer.environment().unload();
        m_renderer.setLighting(m_lightingBeforeMap);
    }
    m_mapInstances.clear();
    m_map = assets::MapManifest{};
    m_hasMap = false;
}

ModelId RenderWorld::add(const ModelDesc& desc) {
    renderer::InstanceDesc d;
    d.model = desc.model;
    d.transform = desc.transform;
    d.isStatic = desc.mobility == Mobility::Static;
    d.layer = desc.layer;
    d.materialOverrides = desc.materialOverrides;
    d.castsShadows = desc.castsShadows;
    d.blobShadow = desc.blobShadow;
    d.occluder = desc.occluder;
    d.animated = desc.animated;
    return m_renderer.scene().add(d);
}

anim::Instance* RenderWorld::animation(ModelId id) {
    return m_renderer.scene().animation(id);
}

anim::ClipRef RenderWorld::clip(ModelId id, renderer::AnimHandle library, std::string_view name) {
    return m_renderer.scene().clip(id, library, name);
}

int32_t RenderWorld::findNode(ModelId id, std::string_view name) const {
    return m_renderer.scene().findNode(id, name);
}

bool RenderWorld::setNodeLocal(ModelId id, uint32_t node, const anim::Transform& local) {
    anim::Instance* a = animation(id);
    if (!a || node >= a->nodeCount()) return false;
    a->setOverride(node, local);
    return true;
}

void RenderWorld::clearNodeLocal(ModelId id, uint32_t node) {
    if (anim::Instance* a = animation(id)) a->clearOverride(node);
}

bool RenderWorld::nodeWorld(ModelId id, uint32_t node, glm::mat4& out) const {
    return m_renderer.scene().nodeWorld(id, node, out);
}

renderer::SpriteId RenderWorld::add(const renderer::SpriteDesc& desc) {
    return m_renderer.sprites().add(desc);
}

void RenderWorld::update(renderer::SpriteId id, const renderer::SpriteDesc& desc) {
    m_renderer.sprites().update(id, desc);
}

void RenderWorld::remove(renderer::SpriteId id) {
    m_renderer.sprites().remove(id);
}

bool RenderWorld::spriteFinished(renderer::SpriteId id, float time) const {
    return m_renderer.sprites().finished(id, time);
}

renderer::EffectId RenderWorld::add(const renderer::EffectDesc& desc) {
    return m_renderer.particles().spawn(desc);
}

void RenderWorld::setTransform(renderer::EffectId id, const glm::vec3& position, const glm::quat& rotation) {
    m_renderer.particles().setTransform(id, position, rotation);
}

void RenderWorld::stop(renderer::EffectId id) {
    m_renderer.particles().stop(id);
}

void RenderWorld::remove(renderer::EffectId id) {
    m_renderer.particles().kill(id);
}

bool RenderWorld::alive(renderer::EffectId id) const {
    return m_renderer.particles().alive(id);
}

void RenderWorld::setTransform(ModelId id, const glm::mat4& transform) {
    m_renderer.scene().setTransform(id, transform);
}

void RenderWorld::remove(ModelId id) {
    m_renderer.scene().remove(id);
}

LightId RenderWorld::add(const LightDesc& desc) {
    return m_renderer.scene().addLight(desc);
}

void RenderWorld::update(LightId id, const LightDesc& desc) {
    m_renderer.scene().updateLight(id, desc);
}

void RenderWorld::remove(LightId id) {
    m_renderer.scene().removeLight(id);
}

void RenderWorld::setEnvironment(const Environment& environment) {
    m_renderer.setLighting(environment);
}

CameraId RenderWorld::add(const CameraDesc& desc) {
    const CameraId id = m_cameras.create(desc);
    if (!id) FL_LOG_ERROR("Camera pool exhausted");
    return id;
}

void RenderWorld::update(CameraId id, const CameraDesc& desc) {
    if (CameraDesc* c = m_cameras.get(id)) *c = desc;
}

void RenderWorld::remove(CameraId id) {
    if (m_cameras.get(id)) m_cameras.destroy(id);
}

core::Result<bool> RenderWorld::render(CameraId camera, const renderer::FrameSettings& settings, double timeSeconds,
                                       float deltaSeconds, std::vector<uint8_t>* screenshot, rhi::Extent2D* screenshotExtent) {
    const CameraDesc* c = m_cameras.get(camera);
    if (!c) return core::Error::NotFound;
    renderer::CameraData data;
    data.view = glm::inverse(c->worldFromView);
    data.position = glm::vec3(c->worldFromView[3]);
    data.fovYRadians = glm::radians(c->fovYDeg);
    data.zNear = c->zNear;
    data.zFar = c->zFar;
    return m_renderer.renderFrame(data, settings, timeSeconds, deltaSeconds, screenshot, screenshotExtent);
}

} // namespace world
