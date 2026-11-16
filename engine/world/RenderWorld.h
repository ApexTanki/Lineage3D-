#pragma once

#include "assets/MapFormat.h"
#include "core/Handle.h"
#include "renderer/core/Renderer.h"

#include <span>
#include <string>
#include <vector>

namespace world {

using ModelId = renderer::InstanceId;
using CameraId = core::Handle<struct CameraTag>;
using LightId = renderer::LightId;
using SpriteId = renderer::SpriteId;
using SpriteDesc = renderer::SpriteDesc;
using SpriteFacing = renderer::SpriteFacing;
using SpriteBlend = renderer::SpriteBlend;
using EffectId = renderer::EffectId;
using EffectDesc = renderer::EffectDesc;

// Local point and spot lights (M2 plan D4) and the environment: the one sun, the hemispherical ambient, the clear color,
// fog and the sky's display parameters (a map sets them from its package). They are the renderer's own plain structs;
// see renderer/core/Lights.h and Renderer.h.
using LightType = renderer::LightType;
using LightDesc = renderer::LightDesc;
using Environment = renderer::LightingData;

enum class Mobility : uint8_t { Static, Dynamic };

// ARCHITECTURE.md §5.6. Static models are grouped and instanced by the renderer; dynamic ones are drawn one by one
// with per-draw constants and may move every frame. Moving a static model (setTransform) rebuilds every instance group
// on the next frame, which allocates: it is a loading-time operation, so anything that moves belongs in Dynamic.
struct ModelDesc {
    renderer::ModelHandle model;
    std::span<const renderer::MaterialHandle> materialOverrides; // empty → the model's own materials
    glm::mat4 transform{1.0f};
    Mobility mobility = Mobility::Dynamic;
    uint8_t layer = 1;         // sort-key layer (0 = big occluders)
    bool castsShadows = true;  // into the shadow maps, where its materials cast (M4 plan D10)
    bool blobShadow = false;   // dynamic models: a soft grounding blob under it (M4 plan D9)
    bool occluder = false;     // static models: its mesh boxes hide what is behind them; solid box-like meshes only (M5 plan D4)
    // Dynamic models (M6 plan D6): animation(id) drives its nodes (clips, layers, node overrides). Models with a skin are
    // always animated.
    bool animated = false;
};

struct CameraDesc {
    glm::mat4 worldFromView{1.0f};
    float fovYDeg = 60.0f;
    float zNear = 0.2f;
    float zFar = 1000.0f;
};

// The interface the game uses: explicit add/update/remove of long-lived objects over a flat renderer scene
// (no scene graph): models, local lights, sprites, particle effects, cameras and baked maps.
class RenderWorld {
public:
    explicit RenderWorld(renderer::Renderer& renderer) : m_renderer(renderer) {}
    ~RenderWorld() { unloadMap(); }
    RenderWorld(const RenderWorld&) = delete;
    RenderWorld& operator=(const RenderWorld&) = delete;

    // Loads a baked map package (data-relative directory, e.g. "maps/m3_baked_map"; M3 plan D12): map.json and the probe
    // grid synchronously, then models, materials and the lightmap/sky/reflection textures asynchronously. Static
    // instances get their lightmap rects and reflection probe; dynamic ones are probe-lit. The map's sun, ambient, fog
    // and sky replace the current environment. Replaces a previously loaded map. Loading time only (allocates).
    core::Result<void> loadMap(const std::string& mapDirectory);
    void unloadMap();
    const assets::MapManifest* map() const { return m_hasMap ? &m_map : nullptr; }
    std::span<const ModelId> mapInstances() const { return m_mapInstances; } // in manifest order

    ModelId add(const ModelDesc& desc);
    void setTransform(ModelId id, const glm::mat4& transform);
    void remove(ModelId id);

    // Animation (ARCHITECTURE.md §5.6, M6 plan D6–D8). animation() is nullptr until the model is Ready (or for models that
    // aren't animated); then it stays valid until remove(). Clips come from a clip library
    // (Renderer::resources().loadAnimation("anims/x.fanim")) bound to the model by node names; keep the library loaded
    // while its clips play. Node overrides (turret yaw, gun pitch) replace a node's animated local transform.
    anim::Instance* animation(ModelId id);
    anim::ClipRef clip(ModelId id, renderer::AnimHandle library, std::string_view name);
    int32_t findNode(ModelId id, std::string_view name) const;
    bool setNodeLocal(ModelId id, uint32_t node, const anim::Transform& local);
    void clearNodeLocal(ModelId id, uint32_t node);
    bool nodeWorld(ModelId id, uint32_t node, glm::mat4& out) const; // attach points (muzzle → emitter), last frame's pose

    // Sprites (M6 plan D11): flipbook billboards (fires, smoke, flashes). A one-shot (loop = false) draws nothing once it has
    // played; spriteFinished() tells when to remove it.
    renderer::SpriteId add(const renderer::SpriteDesc& desc);
    void update(renderer::SpriteId id, const renderer::SpriteDesc& desc);
    void remove(renderer::SpriteId id);
    bool spriteFinished(renderer::SpriteId id, float time) const;

    // Particle effects (M7 plan D1): an effect plays a preset (Renderer::resources().loadPreset("particles/x.fpart")) at a
    // place; it waits while the preset loads. One-shots (explosions) free themselves once their particles and lights
    // are gone; looping ones (fires, exhaust) run until stop(), then fade out and free themselves. remove() ends one at
    // once. A moving effect (dust behind a tank) spawns along its path and passes `inherit` of its velocity on.
    renderer::EffectId add(const renderer::EffectDesc& desc);
    void setTransform(renderer::EffectId id, const glm::vec3& position, const glm::quat& rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    void stop(renderer::EffectId id);
    void remove(renderer::EffectId id);
    bool alive(renderer::EffectId id) const;

    LightId add(const LightDesc& desc);
    void update(LightId id, const LightDesc& desc);
    void remove(LightId id);
    void setEnvironment(const Environment& environment);

    CameraId add(const CameraDesc& desc);
    void update(CameraId id, const CameraDesc& desc);
    void remove(CameraId id);

    // Renders one frame from `camera`; false when nothing was rendered (minimized window).
    core::Result<bool> render(CameraId camera, const renderer::FrameSettings& settings, double timeSeconds, float deltaSeconds,
                              std::vector<uint8_t>* screenshot = nullptr, rhi::Extent2D* screenshotExtent = nullptr);

private:
    renderer::Renderer& m_renderer;
    core::HandlePool<CameraDesc, CameraTag> m_cameras{16};
    assets::MapManifest m_map;
    renderer::LightingData m_lightingBeforeMap; // restored by unloadMap
    bool m_hasMap = false;
    std::vector<ModelId> m_mapInstances;
};

} // namespace world
