#pragma once

#include "core/Result.h"
#include "renderer/core/ClipSpace.h"
#include "renderer/core/Fog.h"
#include "renderer/core/FramePipeline.h"
#include "renderer/core/GpuTimers.h"
#include "renderer/core/Lights.h"
#include "renderer/core/RenderQueue.h"
#include "renderer/core/Scene.h"
#include "renderer/core/ShaderLibrary.h"
#include "renderer/debug/DebugDraw.h"
#include "renderer/debug/ImGuiRenderer.h"
#include "renderer/lighting/Environment.h"
#include "renderer/materials/MaterialSystem.h"
#include "renderer/particles/ParticleRenderer.h"
#include "renderer/particles/ParticleSystem.h"
#include "renderer/post/PostProcess.h"
#include "renderer/post/PostSettings.h"
#include "renderer/post/Ssao.h"
#include "renderer/resources/ResourceManager.h"
#include "renderer/shadows/BlobShadows.h"
#include "renderer/shadows/ShadowMaps.h"
#include "renderer/shadows/ShadowSettings.h"
#include "renderer/visibility/Visibility.h"
#include "renderer/vfx/SpriteRenderer.h"
#include "rhi/Device.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace platform {
class JobSystem;
}

namespace renderer {

struct CameraData {
    glm::mat4 view{1.0f}; // world → view (right-handed, looking down −Z)
    glm::vec3 position{0.0f};
    float fovYRadians = 1.0471976f; // 60°
    float zNear = 0.2f;
    float zFar = 1000.0f;
};

// The environment's lighting (ARCHITECTURE.md §4.4): the one sun, a hemispherical ambient, fog and the sky's display
// parameters, in linear units. Local point and spot lights live in the Scene; a map's baked lighting (lightmaps,
// probes, sky and reflection cubes) lives in the Environment (renderer/lighting/Environment.h).
struct LightingData {
    glm::vec3 sunDirection{0.42f, 0.82f, 0.38f}; // toward the sun; normalized by the renderer
    glm::vec3 sunColor{1.55f, 1.45f, 1.30f};     // color × intensity
    glm::vec3 ambientSky{0.34f, 0.40f, 0.50f};   // also the fallback reflection cube and the SH of probe-less objects
    glm::vec3 ambientGround{0.16f, 0.14f, 0.12f};
    glm::vec3 clearColor{0.40f, 0.52f, 0.68f};   // background where no sky is loaded
    FogSettings fog;                             // M3 plan D10; off by default
    float skyIntensity = 1.0f;                   // multiplies the sky cube in the sky pass
    float sunDiscIntensity = 30.0f;              // sun disc radiance in the sky pass, × sunColor
    float sunDiscRadiusDeg = 0.6f;
    float exposure = 1.0f;                       // scene color × this before the tone curve (M8 plan D2; R8)
};

struct FrameSettings {
    uint8_t msaa = 1;          // 1, 2 or 4; clamped to the device
    float anisotropy = 4.0f;   // up to 16, clamped to the device
    bool instancing = true;    // false draws every static renderable on its own (measurement mode)
    uint8_t maxLocalLights = 4; // per object, 0–8 (tiers: Low 2, Medium 4, High 8)
    DebugView debugView = DebugView::Lit;
    uint64_t uploadBudgetBytes = 8ull << 20;
    ShadowSettings shadows = ShadowSettings::high(); // M4 plan D1
    // Visibility and LOD (M5 plan D3, D4, D6, D7, D10).
    bool culling = true;        // frustum culling of objects and lights, for the camera and every shadow view
    bool occlusion = true;      // software occlusion culling for the camera
    int8_t forceLod = -1;       // −1 automatic; 0–3 a fixed level (clamped per mesh)
    float lodBias = 1.0f;       // > 1 picks coarser levels sooner
    bool lodFade = true;        // dithered cross-fade between levels
    bool freezeCulling = false; // debug: culling, occlusion and LOD stay at the camera of the frame it was turned on
    float animationSpeed = 1.0f; // M6 plan D8: animated instances advance by frame delta × this (0 holds every pose)
    // Particles (M7 plan D4, D8, D10).
    bool particles = true;            // off: effects keep their clocks but spawn nothing, their lights go dark, nothing draws
    bool softParticles = true;        // fade against the scene depth; splits the scene pass while a soft emitter is visible
    float particleSpeed = 1.0f;       // effects advance by frame delta × this (0 holds every particle)
    float particleBudgetScale = 1.0f; // scales every emitter's particle budget (tiers; 0–1)
    float particleCoverageCap = 3.0f; // screens of particle area before spawn rates are scaled down
    PostSettings post = PostSettings::high(); // M8 plan D2: HDR or LDR, tone curve, bloom, FXAA, SSAO, refraction, P2
};

// GPU memory budgets (M9 plan D6), in bytes; 0 = no budget. Textures and meshes are asset budgets kept by the resource
// manager (a texture over budget loads with its largest mips skipped); render targets and buffers are kept by the device
// (an object over budget isn't made: a settings change whose targets don't fit leaves the previous settings in force).
struct MemoryBudgets {
    uint64_t textures = 512ull << 20;
    uint64_t meshes = 256ull << 20;
    uint64_t targets = 0;
    uint64_t buffers = 0;
};

struct FrameStats {
    uint32_t sceneDraws = 0;   // draw calls in the scene pass
    uint32_t instances = 0;    // objects drawn in the scene pass
    uint64_t triangles = 0;
    uint32_t queueItems = 0;
    uint32_t queueDropped = 0;
    uint32_t transparentDraws = 0;
    uint32_t refractionDraws = 0;  // water and glass that refracted this frame (M8 plan D10)
    uint32_t lightsInScene = 0;    // local lights in the scene
    uint32_t lightsVisible = 0;    // of those, the ones whose range reaches the camera frustum (M5 plan D9)
    uint32_t lightsUploaded = 0;   // in this frame's LightArray (≤ 256)
    uint32_t lightsDropped = 0;    // beyond 256: the farthest from the camera are left out
    uint32_t litObjects = 0;       // objects with at least one local light
    uint32_t maxObjectLights = 0;  // the longest per-object list
    uint64_t lightAssignments = 0; // sum of per-object list lengths
    uint32_t probeSamples = 0;     // dynamic objects lit by the probe grid this frame
    bool skyDrawn = false;
    uint32_t blobs = 0;            // blob shadows drawn (one instanced draw)
    Visibility::Stats visibility;  // culling, occlusion and LOD (M5 plan D11)
    // Animation (M6 plan D12): animated instances, skinned renderables, palettes written this frame, CPU times (the
    // evaluation of every pose and the palettes, both on the job workers).
    uint32_t animated = 0;
    uint32_t skinned = 0;
    uint32_t palettes = 0;
    uint32_t palettesDropped = 0; // the uniform ring was full: those skinned renderables weren't drawn
    uint32_t skinnedDraws = 0;  // in the scene pass
    double animationMs = 0.0;
    double paletteMs = 0.0;
    uint32_t sprites = 0;     // sprites drawn (M6 plan D11)
    uint32_t spriteDraws = 0; // their instanced draws
    ParticleSystem::Stats particles;          // effects, emitters, live particles, simulation time (M7 plan D13)
    ParticleRenderer::Stats particleDraws;    // emitters drawn, particles, coverage, instance writing time
    uint32_t particleSceneDraws = 0;          // particle draws recorded in the scene pass
    bool softParticlePass = false;            // the scene pass was split around pass 5c for soft particles
    uint32_t prepassDraws = 0;                // depth prepass draws (High with SSAO; M8 plan D9)
    bool settingsRefused = false;             // the requested settings' targets didn't fit: the previous ones are in force (M9 plan D6)
    double buildMs = 0.0;          // light gathering + queue build (light lists included) + sort
    double recordMs = 0.0;
    double submitMs = 0.0;
    // M9 plan D13: where a slow frame's time went (the benchmark breaks its worst frame down with these).
    double waitMs = 0.0;    // beginFrame: waiting for the frame slot's fence and the next swapchain image
    double presentMs = 0.0; // present
    double renderMs = 0.0;  // the whole renderFrame call
};

// Owns GPU-side state (resources, materials, targets, pipelines) and the Scene, and renders one frame at a time
// (ARCHITECTURE.md §5.4). It holds no gameplay state.
class Renderer {
public:
    struct Desc {
        rhi::Device* device = nullptr;
        platform::JobSystem* jobs = nullptr;
        std::string dataDir; // with a trailing separator
    };

    Renderer() = default;
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    core::Result<void> init(const Desc& desc);
    void shutdown();

    ResourceManager& resources() { return m_resources; }
    Environment& environment() { return m_environment; }
    MaterialSystem& materials() { return m_materials; }
    Scene& scene() { return *m_scene; }
    rhi::Device& device() { return *m_device; }
    const GpuTimers& timers() const { return m_timers; }
    const FrameStats& stats() const { return m_stats; }
    const ShadowMaps& shadows() const { return m_shadows; }
    SpriteRenderer& sprites() { return m_sprites; }
    ParticleSystem& particles() { return m_particles; }
    const Visibility& visibility() const { return m_visibility; }
    void setLighting(const LightingData& lighting) { m_lighting = lighting; }
    const LightingData& lighting() const { return m_lighting; }
    bool initialized() const { return m_initialized; }
    // M9 plan D6: lower budgets don't evict anything; they apply to what is made from then on. A budget that goes down
    // takes effect once the next frame has rendered (that frame swaps the outgoing settings' targets for the incoming
    // ones under the looser of the two); raised budgets, and any before the first frame, apply at once.
    void setMemoryBudgets(const MemoryBudgets& budgets);
    // Hot reload (dev; M9 plan D9) of one cooked file, by its data-relative path: a shader package ("shaders/x.fsp"), a
    // material (".fmat") or a texture (".ktx2"). Call between frames. False for files it doesn't reload (models, maps,
    // presets, the sky) and for reloads that failed (logged; the old state stays).
    bool reload(const std::string& path);
    // The settings the last frame ran with: the request, except where a settings change didn't fit (stats().settingsRefused).
    const FrameSettings& effectiveSettings() const { return m_effective; }

    // One frame: beginFrame → finalize uploads → scene update → frame lights → queue build with per-object light lists
    // and probe samples (jobs) → scene pass (opaque, alpha test, sky, transparent) → present and debug-UI passes → submit →
    // present. The debug
    // UI pass draws ImGui's current draw data, if any. Returns false when there was nothing to render to (minimized
    // window). With `screenshot`, the backbuffer is read back before presenting.
    core::Result<bool> renderFrame(const CameraData& camera, const FrameSettings& request, double timeSeconds,
                                   float deltaSeconds, std::vector<uint8_t>* screenshot = nullptr,
                                   rhi::Extent2D* screenshotExtent = nullptr);

private:
    struct BuildContext;
    struct alignas(64) WorkerStats { // one per job participant, so the queue build never shares counters
        uint64_t assignments = 0;
        uint32_t litObjects = 0;
        uint32_t maxLights = 0;
        uint32_t probeSamples = 0;
    };
    // Queue build, phase 1: light lists of every static row (spread over rows, so a few huge instance groups still use
    // every worker). Phase 2: queue items, light lists of dynamic renderables, instance data.
    static void staticLightsRange(void* context, uint32_t begin, uint32_t end, uint32_t worker);
    static void buildRange(void* context, uint32_t begin, uint32_t end, uint32_t worker);
    bool uploadLights(const CameraData& camera, float time, uint32_t& lightOffset);
    // `refraction`: water and glass refract this frame (else their blend twins draw among the transparents, M8 plan D10).
    void buildQueue(const CameraData& camera, const FrameSettings& settings, bool refraction);
    size_t commandBytesBound(const FrameSettings& settings) const;
    // `split`: soft particles or refractive surfaces are visible, so the scene pass ends before them, pass 5c copies the
    // depth and pass B continues (M7 plan D8); `grab`: water or glass is visible, so pass 5d grabs the resolved world
    // between the two (M8 plan D10).
    void recordScene(rhi::CommandList& cmd, const uint32_t frameOffsets[2], bool drawBlobs, DebugView view, const glm::mat4& viewProj,
                     bool split, bool grab, const CameraData& camera);
    void recordPrepass(rhi::CommandList& cmd, const uint32_t frameOffsets[2]); // pass 2, depth only (M8 plan D9)
    bool rebuildForwardSets(); // set 1 of the lit pipelines: linear depth, SSAO, refraction grab (M8 plan D9, D10)
    bool queueHasPass(QueuePass pass) const; // the sorted camera queue holds a draw of that pass
    bool rebuildFrameSet();

    rhi::Device* m_device = nullptr;
    platform::JobSystem* m_jobs = nullptr;
    std::unique_ptr<ShaderLibrary> m_shaders;
    ResourceManager m_resources;
    Environment m_environment;
    MaterialSystem m_materials;
    std::unique_ptr<Scene> m_scene;
    FramePipeline m_pipeline;
    ShadowMaps m_shadows;
    BlobShadows m_blobs;
    Visibility m_visibility;
    DebugDraw m_debugDraw; // wireframe and overdraw views (M5 plan D10)
    SpriteRenderer m_sprites; // flipbook sprites (M6 plan D11)
    ParticleSystem m_particles;           // effects and their simulation (M7 plan D1–D4)
    ParticleRenderer m_particleRenderer;  // their instances and draws (M7 plan D5–D10)
    PostProcess m_post;                   // bloom, tonemap, FXAA (M8 plan D3–D7)
    Ssao m_ssao;                          // depth prepass AO (M8 plan D9)
    uint32_t m_particleQueueItems = 0;    // this frame's emitter items in m_queue
    ImGuiRenderer m_imgui;
    GpuTimers m_timers;
    RenderQueue m_queue{65536};
    rhi::ResourceSetLayoutHandle m_frameLayout;
    rhi::ResourceSetHandle m_frameSet;
    rhi::ResourceSetLayoutHandle m_forwardLayout; // set 1 of the lit pipelines: bindings 1–4, textures
    rhi::ResourceSetHandle m_forwardSets[2];      // [0] scene pass A or a whole pass, [1] pass B (with the grab)
    rhi::TextureHandle m_forwardInputs[3];        // linear depth, SSAO, grab: the textures those sets were built with
    rhi::SamplerHandle m_pointClamp;              // the linear depth is read texel for texel
    uint32_t m_frameSetRevision = 0;       // Environment::revision() the frame set was built for
    uint32_t m_frameSetShadowRevision = 0; // ShadowMaps::revision() likewise
    LightingData m_lighting;
    glm::mat4 m_previousViewProj{1.0f}; // last frame's backend view-projection: camera motion blur (M8 plan D12)
    bool m_hasPreviousViewProj = false;
    FrameStats m_stats;
    float m_anisotropy = 4.0f;
    bool m_initialized = false;
    bool m_warnedLightOverflow = false;
    bool m_materialsNeedTarget = false; // set on a scene-layout change, cleared once every pipeline was rebuilt
    // M9 plan D6: the settings this frame runs with; the target-affecting settings last made to fit; a request whose
    // targets didn't fit, not tried again until the request or the window size changes.
    FrameSettings m_effective;
    FrameSettings m_targetsAccepted;
    FrameSettings m_targetsRefused;
    rhi::Extent2D m_refusedExtent;
    bool m_hasAcceptedTargets = false;
    bool m_hasRefusedTargets = false;
    // Budgets in force, and the ones asked for when a lowered budget waits for the next rendered frame.
    void applyBudgets(const MemoryBudgets& budgets);
    MemoryBudgets m_budgets;
    MemoryBudgets m_budgetsRequested;
    bool m_budgetsPending = false;
    uint64_t m_framesRendered = 0;

    // Per-frame working memory, sized up front (or when the scene grows), never per frame.
    std::vector<const LightDesc*> m_lightCandidates;
    std::vector<FrameLight> m_frameLights;
    std::vector<glm::uvec4> m_dynamicLightLists; // per dynamic renderable
    std::vector<glm::uvec4> m_staticLightLists;  // per static row drawn on its own (instancing off, blended)
    std::vector<int8_t> m_spotSlots;             // per frame light: spot shadow slot or −1
    std::vector<WorkerStats> m_workerStats;
    // Probe samples (M3 plan D6): SH irradiance, sun visibility and reflection slot of each dynamic renderable.
    struct ProbeSample {
        glm::vec4 sh[3];
        float sunVisibility;
        uint32_t reflectionSlot;
    };
    std::vector<ProbeSample> m_dynamicProbes;
    // The hemispherical ambient as SH: dynamic objects without a probe grid, and statics without lightmaps drawn on
    // their own (the same light their instanced groups get from the INSTANCED variant).
    ProbeSample m_ambientProbe{};
};

} // namespace renderer
