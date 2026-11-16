#include "renderer/core/Renderer.h"

#include "core/Log.h"
#include "core/Profile.h"
#include "platform/JobSystem.h"
#include "platform/Timer.h"
#include "renderer/core/Output.h"
#include "renderer/core/ShaderTypes.h"
#include "renderer/core/SortKey.h"
#include "renderer/post/ToneCurves.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace renderer {

struct Renderer::BuildContext {
    Renderer* self = nullptr;
    const Scene* scene = nullptr;
    glm::vec3 eye{0.0f};
    float invFar = 0.0f;
    bool instancing = true;
    bool refraction = true; // refractive materials refract (else their blended twins draw)
    uint32_t maxLights = 0;
    uint32_t staticItems = 0; // index space of phase 2: [0, staticItems) statics (groups or rows), then dynamics
    std::span<const FrameLight> lights;
    // Visibility of this frame (M5 plan D3, D6): view masks, camera LODs and shadow levels per row and dynamic renderable.
    std::span<const uint8_t> rowMasks, dynMasks;
    std::span<const Visibility::DrawLod> rowLods, dynLods;
    std::span<const uint8_t> rowShadowLevels, dynShadowLevels;
    uint8_t shadowBits = 0; // mask bits of the shadow views drawn this frame (bit v + 1 = view v)
};

namespace {

void countLights(uint64_t& assignments, uint32_t& litObjects, uint32_t& maxLights, const glm::uvec4& list) {
    const uint32_t n = lightListCount(list);
    assignments += n;
    litObjects += n > 0 ? 1u : 0u;
    maxLights = std::max(maxLights, n);
}

} // namespace

Renderer::~Renderer() {
    shutdown();
}

core::Result<void> Renderer::init(const Desc& desc) {
    m_device = desc.device;
    m_jobs = desc.jobs;
    m_shaders = std::make_unique<ShaderLibrary>(*m_device);
    for (const char* name : {"lit", "present", "imgui", "sky", "shadow", "blob", "debug", "sprite", "particle", "depth_copy", "tonemap",
                             "bloom", "fxaa", "prepass", "ssao", "copy", "dof", "motion_blur", "god_rays"})
        if (auto r = m_shaders->load(name, desc.dataDir + "shaders/" + name + ".fsp"); !r) return r;

    // Set 0: the frame's uniforms (b0) and local lights (b1), each at a new dynamic offset into the transient uniform
    // ring every frame, so no descriptor set is ever allocated per frame (ARCHITECTURE.md §4.6); plus the shadow maps
    // (b2 cascades, b3 spot atlas; M4 plan D11) and the baked lighting textures (b4 sky cube, b5 lightmaps, b6
    // shadowmask), whose set is rebuilt only when a map's textures arrive or the shadow settings change.
    rhi::ResourceSetLayoutDesc ld;
    ld.set = 0;
    ld.count = 7;
    ld.bindings[0] = {0, rhi::BindingKind::UniformBufferDynamic, rhi::ShaderStageMask::All};
    ld.bindings[1] = {1, rhi::BindingKind::UniformBufferDynamic, rhi::ShaderStageMask::Fragment};
    ld.bindings[2] = {2, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment, rhi::TextureType::Tex2DArray, true};
    ld.bindings[3] = {3, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment, rhi::TextureType::Tex2D, true};
    ld.bindings[4] = {4, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment, rhi::TextureType::Cube};
    ld.bindings[5] = {5, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment, rhi::TextureType::Tex2DArray};
    ld.bindings[6] = {6, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment, rhi::TextureType::Tex2DArray};
    m_frameLayout = m_device->createResourceSetLayout(ld);
    if (!m_frameLayout) return core::Error::InitFailed;
    // Set 1 of the lit pipelines: the forward passes' inputs (M8 plan D9, D10), within set 1's four GL texture units.
    rhi::ResourceSetLayoutDesc fl;
    fl.set = 1;
    fl.count = 4;
    for (uint8_t b = 0; b < 4; ++b) fl.bindings[b] = {uint8_t(1 + b), rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment};
    m_forwardLayout = m_device->createResourceSetLayout(fl);
    rhi::SamplerDesc pointDesc;
    pointDesc.filter = rhi::Filter::Nearest;
    pointDesc.mip = rhi::MipMode::Nearest;
    pointDesc.address = rhi::AddressMode::ClampToEdge;
    m_pointClamp = m_device->createSampler(pointDesc);
    if (!m_forwardLayout || !m_pointClamp) return core::Error::InitFailed;

    if (auto r = m_resources.init({m_device, desc.dataDir, 2}); !r) return r;
    if (auto r = m_environment.init({m_device, &m_resources}); !r) return r;
    if (auto r = m_materials.init(
            {m_device, m_shaders.get(), &m_resources, m_frameLayout, m_environment.reflectionLayout(), m_environment.skinnedLayout(),
             m_forwardLayout});
        !r)
        return r;
    if (auto r = m_shadows.init(*m_device, *m_shaders, m_materials, m_environment.paletteLayout(), m_environment.paletteSet()); !r) return r;
    // The post chain and SSAO before the first frame set: it binds their textures (a white texel while SSAO is off).
    if (auto r = m_post.init(*m_device, *m_shaders, m_frameLayout); !r) return r;
    if (auto r = m_ssao.init(*m_device, *m_shaders, m_frameLayout); !r) return r;
    if (!rebuildFrameSet()) return core::Error::InitFailed;
    m_scene = std::make_unique<Scene>(m_resources, m_materials);
    m_particles.init(m_resources, *m_scene);
    if (auto r = m_pipeline.init(*m_device, *m_shaders, m_frameLayout); !r) return r;
    if (auto r = m_particleRenderer.init(*m_device, *m_shaders, m_resources, m_frameLayout, m_pipeline.softLayout()); !r) return r;
    if (auto r = m_blobs.init(*m_device, *m_shaders, m_frameLayout); !r) return r;
    m_visibility.init(*m_jobs);
    if (auto r = m_debugDraw.init(*m_device, *m_shaders, m_environment.paletteLayout(), m_environment.paletteSet()); !r) return r;
    if (auto r = m_sprites.init(*m_device, *m_shaders, m_resources, m_frameLayout); !r) return r;
    rhi::AttachmentLayout backbuffer;
    backbuffer.colorCount = 1;
    backbuffer.color[0] = m_device->backbufferFormat();
    if (auto r = m_imgui.init(*m_device, *m_shaders, backbuffer); !r) return r;
    if (!m_timers.init(*m_device)) return core::Error::InitFailed;
    m_lightCandidates.reserve(1024); // the scene's light pool capacity: it never holds more
    m_frameLights.reserve(kMaxFrameLights);
    m_spotSlots.resize(kMaxFrameLights);
    m_workerStats.resize(m_jobs->participantCount());
    m_initialized = true;
    return {};
}

void Renderer::shutdown() {
    if (!m_initialized) return;
    m_device->waitIdle();
    m_particles.shutdown(); // releases its presets and removes its lights, so before the scene
    m_scene.reset();        // releases model and material references
    m_particleRenderer.shutdown();
    m_timers.shutdown();
    m_imgui.shutdown();
    m_blobs.shutdown();
    m_debugDraw.shutdown();
    m_sprites.shutdown(); // releases its atlas textures, so before the resource manager
    m_ssao.shutdown();
    m_post.shutdown();
    m_pipeline.shutdown();
    m_shadows.shutdown(); // its pipelines use the material set layout, so before the material system
    m_materials.shutdown();
    m_environment.shutdown(); // releases its texture references, so before the resource manager
    m_resources.shutdown();
    m_device->destroy(m_frameSet);
    m_device->destroy(m_forwardSets[0]);
    m_device->destroy(m_forwardSets[1]);
    m_device->destroy(m_forwardLayout);
    m_device->destroy(m_pointClamp);
    m_device->destroy(m_frameLayout);
    m_shaders.reset();
    m_initialized = false;
}

namespace {

// The settings whose change makes or drops render targets (M9 plan D6); the overdraw view drops the post chain's. Shadow
// maps keep their own fallback: maps that don't fit leave the previous shadow settings in force (ShadowMaps::configure).
bool sameTargets(const FrameSettings& a, const FrameSettings& b) {
    const PostSettings& p = a.post;
    const PostSettings& q = b.post;
    return a.msaa == b.msaa && a.particles == b.particles && a.softParticles == b.softParticles && p.hdr == q.hdr && p.bloom == q.bloom &&
           p.fxaa == q.fxaa && p.ssao == q.ssao && p.refraction == q.refraction && p.heatHaze == q.heatHaze &&
           p.depthOfField == q.depthOfField && p.motionBlur == q.motionBlur && p.godRays == q.godRays && p.ssr == q.ssr &&
           (a.debugView == DebugView::Overdraw) == (b.debugView == DebugView::Overdraw);
}

void copyTargets(const FrameSettings& from, FrameSettings& to) {
    if ((to.debugView == DebugView::Overdraw) != (from.debugView == DebugView::Overdraw)) to.debugView = from.debugView;
    to.msaa = from.msaa;
    to.particles = from.particles;
    to.softParticles = from.softParticles;
    to.post.hdr = from.post.hdr;
    to.post.bloom = from.post.bloom;
    to.post.fxaa = from.post.fxaa;
    to.post.ssao = from.post.ssao;
    to.post.refraction = from.post.refraction;
    to.post.heatHaze = from.post.heatHaze;
    to.post.depthOfField = from.post.depthOfField;
    to.post.motionBlur = from.post.motionBlur;
    to.post.godRays = from.post.godRays;
    to.post.ssr = from.post.ssr;
}

} // namespace

bool Renderer::reload(const std::string& path) {
    auto endsWith = [&](const char* suffix) {
        const size_t n = std::strlen(suffix);
        return path.size() > n && path.compare(path.size() - n, n, suffix) == 0;
    };
    if (path.rfind("shaders/", 0) == 0 && endsWith(".fsp")) {
        const std::string name = path.substr(8, path.size() - 8 - 4);
        auto result = m_shaders->reload(name, m_resources.dataDir() + path);
        if (!result) {
            FL_LOG_ERROR("Hot reload: %s failed (%s)", path.c_str(), core::errorName(result.error()));
            return false;
        }
        const ShaderLibrary::ReloadResult r = result.value();
        FL_LOG_INFO("Hot reload: %s, %u of %u variant(s) replaced", path.c_str(), r.replaced, r.variants);
        return r.replaced == r.variants; // a variant left with its old code makes the reload a failure (logged above)
    }
    if (endsWith(".fmat")) {
        const bool ok = m_materials.reload(path);
        if (ok) FL_LOG_INFO("Hot reload: %s", path.c_str());
        return ok;
    }
    if (endsWith(".ktx2")) {
        const bool ok = m_resources.reloadTexture(path);
        if (ok) FL_LOG_INFO("Hot reload: %s (loading)", path.c_str());
        return ok;
    }
    return false;
}

void Renderer::applyBudgets(const MemoryBudgets& budgets) {
    m_resources.setBudget(MemoryCategory::Textures, budgets.textures);
    m_resources.setBudget(MemoryCategory::Meshes, budgets.meshes);
    m_device->memory().setBudget(rhi::GpuMemoryCategory::RenderTargets, budgets.targets);
    m_device->memory().setBudget(rhi::GpuMemoryCategory::Buffers, budgets.buffers);
    m_budgets = budgets;
}

void Renderer::setMemoryBudgets(const MemoryBudgets& budgets) {
    // M9 plan D6: a budget that goes down waits until a frame has rendered. That frame still holds the outgoing settings'
    // targets while its systems make the incoming ones (shadow maps before the frame targets), so a lower budget at once
    // could refuse the very change that frees memory (a tier switch from High to Low at 1440p). Raised budgets, and every
    // budget before the first frame, apply at once.
    m_budgetsRequested = budgets;
    if (!m_framesRendered) {
        applyBudgets(budgets);
        m_budgetsPending = false;
        return;
    }
    auto looser = [](uint64_t a, uint64_t b) { return a == 0 || b == 0 ? 0 : std::max(a, b); }; // 0 = no budget
    MemoryBudgets now;
    now.textures = looser(m_budgets.textures, budgets.textures);
    now.meshes = looser(m_budgets.meshes, budgets.meshes);
    now.targets = looser(m_budgets.targets, budgets.targets);
    now.buffers = looser(m_budgets.buffers, budgets.buffers);
    applyBudgets(now);
    m_budgetsPending = true;
}

core::Result<bool> Renderer::renderFrame(const CameraData& camera, const FrameSettings& request, double timeSeconds,
                                         float deltaSeconds, std::vector<uint8_t>* screenshot, rhi::Extent2D* screenshotExtent) {
    FL_ZONE("Renderer::renderFrame");
    m_effective = request;
    const FrameSettings& settings = m_effective;
    const uint64_t frameStart = platform::nowTicks();
    auto begun = m_device->beginFrame();
    if (!begun) return begun.error();
    if (!begun.value()) return false;
    m_stats.waitMs = platform::ticksToMilliseconds(platform::nowTicks() - frameStart);
    m_timers.beginFrame(); // every path below ends with m_timers.endFrame(), so the timers stay on the device's slots
    const rhi::Extent2D extent = m_device->backbufferExtent();
    // M9 plan D6: a request whose render targets didn't fit runs on the settings last made to fit until it, or the window
    // size, changes. (Only target-making fields are swapped, none of which the code before prepareTargets reads.)
    if (m_hasRefusedTargets) {
        if (m_hasAcceptedTargets && sameTargets(request, m_targetsRefused) && extent.width == m_refusedExtent.width &&
            extent.height == m_refusedExtent.height)
            copyTargets(m_targetsAccepted, m_effective);
        else
            m_hasRefusedTargets = false;
    }

    m_resources.finalizeUploads(settings.uploadBudgetBytes);
    m_materials.update();
    if (settings.anisotropy != m_anisotropy) { // a settings change, never part of a steady frame
        m_anisotropy = settings.anisotropy;
        m_materials.setAnisotropy(m_anisotropy);
    }
    m_scene->update();
    // Animation (M6 plan D8): every pose on the job workers, so the animated renderables' bounds are current for culling.
    const uint64_t animStart = platform::nowTicks();
    m_scene->animate(*m_jobs, std::max(0.0f, deltaSeconds * settings.animationSpeed));
    m_stats.animationMs = platform::ticksToMilliseconds(platform::nowTicks() - animStart);
    m_stats.animated = m_scene->animatedCount();
    m_stats.skinned = m_scene->skinnedCount();
    m_environment.setAmbient(m_lighting.ambientSky, m_lighting.ambientGround); // no-op unless the ambient changed
    m_environment.update();
    m_shadows.configure(settings.shadows); // recreates the shadow maps only when their size or count changed

    // Once beginFrame() succeeded the frame must be submitted and presented, or the next one would wait forever on
    // this slot's fence; failures below still finish it with just a cleared backbuffer.
    auto abandon = [&](core::Error error) -> core::Result<bool> {
        rhi::CommandList* cmd = m_device->allocateCommandList();
        rhi::RenderPassDesc pass;
        pass.debugName = "Backbuffer (failed frame)";
        pass.toBackbuffer = true;
        pass.colorCount = 1;
        cmd->beginRenderPass(pass);
        cmd->endRenderPass();
        m_device->submit({&cmd, 1});
        m_device->present();
        m_timers.endFrame();
        return error;
    };
    bool layoutChanged = false;
    // Pass 5c copies the depth for every consumer that is on (M8 plan D4): soft particles, SSAO, refraction, haze and P2.
    const PostSettings& post = settings.post;
    auto prepareTargets = [&](bool& changed) -> core::Result<void> {
        const bool effects = post.hdr && settings.debugView != DebugView::Overdraw;
        const bool ao = effects && post.ssao;
        const bool depth = (settings.particles && settings.softParticles) || ao ||
                           (effects && (post.refraction || post.heatHaze || post.depthOfField || post.motionBlur || post.godRays || post.ssr));
        if (auto r = m_pipeline.prepare(extent, settings.msaa, post.hdr, depth, changed); !r) return r.error();
        if (auto r = m_post.prepare(extent, m_pipeline.sceneColor(), m_pipeline.linearDepth(), m_pipeline.revision(), post); !r) return r.error();
        if (auto r = m_ssao.prepare(extent, m_pipeline.linearDepth(), m_pipeline.revision(), ao); !r) return r.error();
        return {};
    };
    if (auto r = prepareTargets(layoutChanged); !r) {
        // M9 plan D6: the targets don't fit (over the render-target budget, or out of memory). A settings change goes back
        // to the settings last made to fit. When those don't fit either (the window grew) or none have fit yet (startup),
        // MSAA comes down a step at a time. The request isn't tried again until it or the window size changes.
        const bool settingsChange = m_hasAcceptedTargets && !sameTargets(settings, m_targetsAccepted);
        FL_LOG_WARN("The render targets of the requested settings don't fit at %ux%u (%s): %s", extent.width, extent.height,
                    core::errorName(r.error()), settingsChange ? "the previous settings stay in force" : "lowering MSAA");
        m_targetsRefused = request;
        m_refusedExtent = extent;
        m_hasRefusedTargets = true;
        core::Result<void> again = r;
        bool changed = false;
        if (settingsChange) {
            copyTargets(m_targetsAccepted, m_effective);
            again = prepareTargets(changed);
            layoutChanged = layoutChanged || changed;
        }
        while (!again && m_effective.msaa > 1) {
            m_effective.msaa = uint8_t(m_effective.msaa / 2);
            again = prepareTargets(changed);
            layoutChanged = layoutChanged || changed;
        }
        if (!again) {
            m_materialsNeedTarget = m_materialsNeedTarget || layoutChanged;
            return abandon(again.error());
        }
    }
    // Before anything below can abandon the frame: a layout change must reach the pipelines even if this frame doesn't
    // draw, because the next prepare won't report it again.
    m_materialsNeedTarget = m_materialsNeedTarget || layoutChanged;
    m_targetsAccepted = settings;
    m_hasAcceptedTargets = true;
    m_stats.settingsRefused = m_hasRefusedTargets;
    const bool hdrEffects = post.hdr && settings.debugView != DebugView::Overdraw;
    const bool ssaoOn = hdrEffects && post.ssao;
    // The environment replaced a texture or the shadow maps were recreated: the frame set must follow, or it would keep
    // pointing at released images. The forward passes' inputs (set 1: linear depth, SSAO, refraction grab) likewise. A
    // failed rebuild (out of memory) skips drawing until a retry succeeds.
    if ((m_environment.revision() != m_frameSetRevision || m_shadows.revision() != m_frameSetShadowRevision) && !rebuildFrameSet())
        return abandon(core::Error::OutOfMemory);
    if (!rebuildForwardSets()) return abandon(core::Error::OutOfMemory);
    if (!m_particleRenderer.setHazeInputs(m_pipeline.linearDepth(), m_post.grab())) return abandon(core::Error::OutOfMemory);
    // A failed rebuild stays pending: the next frame tries again instead of drawing without those pipelines.
    if (m_materialsNeedTarget) {
        if (auto r = m_materials.setTarget(m_pipeline.sceneLayout()); !r) return abandon(r.error());
        if (!m_blobs.setTarget(m_pipeline.sceneLayout())) return abandon(core::Error::CompileFailed);
        if (!m_debugDraw.setTarget(m_pipeline.sceneLayout())) return abandon(core::Error::CompileFailed);
        if (!m_sprites.setTarget(m_pipeline.sceneLayout())) return abandon(core::Error::CompileFailed);
        if (!m_particleRenderer.setTarget(m_pipeline.sceneLayout())) return abandon(core::Error::CompileFailed);
        m_materialsNeedTarget = false;
    }

    FrameUniforms fu;
    const ClipSpace clipSpace(m_device->caps().clipSpace);
    const float aspect = float(extent.width) / float(std::max(1u, extent.height));
    const glm::mat4 projection = clipSpace.perspective(camera.fovYRadians, aspect, camera.zNear, camera.zFar);
    fu.viewProj = projection * camera.view;
    // Rotation only: the sky pass turns far-plane clip positions straight into view directions (M3 plan D9).
    fu.skyFromClip = glm::inverse(projection * glm::mat4(glm::mat3(camera.view)));
    fu.time = glm::vec4(float(timeSeconds), deltaSeconds, 0.0f, 0.0f);
    // Alpha-tested pipelines turn alpha to coverage on whenever the scene target is multisampled (M2 plan D6).
    // The wireframe view shades the scene as lit: its lines come from a pass of their own (M5 plan D10).
    const DebugView shadedView = settings.debugView == DebugView::Wireframe ? DebugView::Lit : settings.debugView;
    fu.outputParams = glm::vec4(shaderEncodesSrgb(m_device->caps()) ? 1.0f : 0.0f, m_pipeline.msaa() > 1 ? 1.0f : 0.0f,
                                float(shadedView), 0.0f);
    fu.cameraPosition = glm::vec4(camera.position, 1.0f);
    fu.cameraUp = glm::vec4(glm::normalize(glm::vec3(camera.view[0][1], camera.view[1][1], camera.view[2][1])), 0.0f); // the view's +Y row
    fu.sunDirection = glm::vec4(glm::normalize(m_lighting.sunDirection), std::cos(glm::radians(m_lighting.sunDiscRadiusDeg)));
    fu.sunColor = glm::vec4(m_lighting.sunColor, m_lighting.sunDiscIntensity);
    const float skyLuminance = glm::dot(m_lighting.ambientSky, glm::vec3(0.2126f, 0.7152f, 0.0722f));
    fu.ambientSky = glm::vec4(m_lighting.ambientSky, skyLuminance);
    fu.ambientGround = glm::vec4(m_lighting.ambientGround, m_lighting.skyIntensity);
    const FogSettings& fog = m_lighting.fog;
    fu.fogColor = glm::vec4(fog.color, float(fog.mode));
    fu.fogParams = glm::vec4(fog.start, fog.end, fog.density, fog.maxOpacity);
    fu.fogHeight = glm::vec4(fog.heightBase, fog.heightDensity, fog.heightFalloff, fog.height ? 1.0f : 0.0f);
    fu.fogSky = glm::vec4(fog.skyHorizon, 0.0f, 0.0f, 0.0f);
    // Post (M8 plan D13): on the LDR path the forward shaders apply exposure and the curve; data views never do.
    const bool dataView = shadedView != DebugView::Lit;
    fu.screen = glm::vec4(float(extent.width), float(extent.height), 1.0f / float(std::max(1u, extent.width)),
                          1.0f / float(std::max(1u, extent.height)));
    fu.post = glm::vec4(m_lighting.exposure, !settings.post.hdr && !dataView ? 1.0f : 0.0f, float(settings.post.tonemapper),
                        ssaoOn && m_ssao.ready() ? 1.0f : 0.0f);
    // Refraction (M8 plan D10): water and glass read the grab and its linear depth (set 1, pass B) while it is on and both
    // targets exist; SSR (a P2 toggle, D12) marches the same two. Otherwise refractive materials draw as blended twins.
    const bool refractionOn = post.refraction && hdrEffects && m_post.grab() && m_pipeline.softDepthAvailable();
    // Heat haze (M8 plan D11): distort emitters bend a grab of the finished scene in pass 8.
    const bool hazeOn = post.heatHaze && hdrEffects && m_post.grab() && m_pipeline.softDepthAvailable();
    fu.postParams = glm::vec4(refractionOn ? 1.0f : 0.0f, refractionOn && post.ssr ? 1.0f : 0.0f, 0.0f, 0.0f);
    m_environment.fillFrame(fu);
    ambientSh(m_lighting.ambientSky, m_lighting.ambientGround, m_ambientProbe.sh);
    m_ambientProbe.sunVisibility = 1.0f;
    m_ambientProbe.reflectionSlot = 0;

    uint64_t t0 = platform::nowTicks();
    // Visibility (M5 plan D3): the camera in canonical clip space, so both backends cull identically.
    Visibility::Settings vis;
    vis.culling = settings.culling;
    vis.occlusion = settings.occlusion;
    vis.staticCasters = settings.shadows.staticCasters;
    vis.forceLod = settings.forceLod;
    vis.lodBias = settings.lodBias;
    vis.lodFade = settings.lodFade;
    vis.freeze = settings.freezeCulling;
    m_visibility.beginFrame(ClipSpace::canonicalPerspective(camera.fovYRadians, aspect, camera.zNear, camera.zFar) * camera.view,
                            camera.position, std::tan(camera.fovYRadians * 0.5f), vis, deltaSeconds);
    // Particles (M7 plan D4, D11): effects start and finish and their lights follow, before the frame's lights are
    // gathered; every emitter simulates on the job workers. Switched off, effects keep their clocks (one-shots still
    // finish and free themselves) but spawn nothing, their lights go dark and nothing draws.
    {
        ParticleSystem::Settings ps;
        ps.enabled = settings.particles;
        ps.speed = settings.particleSpeed;
        ps.budgetScale = settings.particleBudgetScale;
        ps.coverageCap = settings.particleCoverageCap;
        m_particles.update(*m_jobs, std::max(0.0f, deltaSeconds * settings.particleSpeed), camera.position, ps);
    }
    m_stats.particles = m_particles.stats();
    uint32_t lightOffset = 0;
    if (!uploadLights(camera, float(timeSeconds), lightOffset)) return abandon(core::Error::OutOfMemory); // also picks the spot shadows
    // Shadow fit (M4 plan D4, D8): cascades from the camera and the sun; the receivers' fields of FrameUniforms.
    ShadowMaps::FrameInput shadowInput;
    shadowInput.clipSpace = &clipSpace;
    shadowInput.cameraPosition = camera.position;
    shadowInput.cameraForward = -glm::vec3(camera.view[0][2], camera.view[1][2], camera.view[2][2]); // the view's −Z row
    shadowInput.fovY = camera.fovYRadians;
    shadowInput.aspect = aspect;
    shadowInput.zNear = camera.zNear;
    shadowInput.zFar = camera.zFar;
    shadowInput.sunDirection = glm::vec3(fu.sunDirection);
    shadowInput.time = float(timeSeconds);
    m_shadows.prepare(shadowInput, fu);
    // Every shadow view culls on its own (M5 plan D3, D8): cascades without a near plane (casters toward the sun are
    // pancaked), spot tiles with their whole frustum.
    const auto fits = m_shadows.cascades();
    for (uint32_t c = 0; c < fits.size(); ++c) m_visibility.setShadowView(c, fits[c].projection * fits[c].view, false);
    for (uint32_t k = 0; k < m_shadows.spotCount(); ++k)
        m_visibility.setShadowView(kMaxCascades + k, m_shadows.spotFit(k).projection * m_shadows.spotFit(k).view, true);
    m_visibility.run(*m_scene);
    m_stats.visibility = m_visibility.stats();
    const rhi::TransientRing::Allocation frameData = m_device->transient(rhi::TransientKind::Uniform).allocate(sizeof(fu));
    if (!frameData.cpu) return abandon(core::Error::OutOfMemory);
    std::memcpy(frameData.cpu, &fu, sizeof(fu));
    // Bone palettes of the skinned renderables some view sees (M6 plan D8), before the queue build reads their offsets, and
    // after the frame's own uniforms: if many skinned renderables fill the ring, they go undrawn, not the whole frame.
    const uint64_t paletteStart = platform::nowTicks();
    m_scene->writePalettes(*m_jobs, m_visibility.dynamicMasks(), m_device->transient(rhi::TransientKind::Uniform));
    m_stats.paletteMs = platform::ticksToMilliseconds(platform::nowTicks() - paletteStart);
    m_stats.palettes = m_scene->palettesWritten();
    m_stats.palettesDropped = m_scene->palettesDropped();
    if (m_stats.palettesDropped)
        FL_LOG_WARN("Uniform ring full: %u skinned renderables without a bone palette this frame (not drawn)", m_stats.palettesDropped);
    // Particle instances (M7 plan D5): the visible emitters, sorted and written into the vertex ring on the job workers;
    // their screen coverage drives the coverage cap from the next frame on (D10).
    if (settings.particles) {
        ParticleRenderer::FrameInput pin;
        pin.system = &m_particles;
        pin.visibility = &m_visibility;
        pin.probes = &m_environment.probes();
        pin.ambientSh = m_ambientProbe.sh;
        pin.eye = camera.position;
        pin.invFar = 1.0f / camera.zFar;
        pin.tanHalfFovY = std::tan(camera.fovYRadians * 0.5f);
        pin.aspect = aspect;
        pin.soft = settings.softParticles && m_pipeline.softDepthAvailable();
        pin.haze = hazeOn;
        m_particleRenderer.prepare(*m_jobs, pin);
        m_particles.reportCoverage(m_particleRenderer.stats().coverage);
    } else {
        m_particleRenderer.clearFrame();
    }
    m_stats.particleDraws = m_particleRenderer.stats();
    buildQueue(camera, settings, refractionOn); // also queues the shadow casters
    // Blob shadows draw in the lit views only: a data view (albedo, normals, ...) shows the raw value.
    const bool blobView = settings.debugView == DebugView::Lit || settings.debugView == DebugView::Lighting ||
                          settings.debugView == DebugView::Shadow;
    m_stats.blobs = settings.shadows.blobShadows && blobView ? m_blobs.prepare(*m_scene, settings.shadows.blobStrength, &m_visibility) : 0u;
    // Sprites (M6 plan D11): culled with the camera, animated and sorted back to front.
    m_stats.sprites = m_sprites.prepare(m_visibility, camera.position, float(timeSeconds));
    m_stats.spriteDraws = m_sprites.drawCount();
    uint64_t t1 = platform::nowTicks();
    m_stats.buildMs = platform::ticksToMilliseconds(t1 - t0);
    const uint32_t frameOffsets[2] = {frameData.gpu.offset, lightOffset};

    m_device->reserveCommandBytes(commandBytesBound(settings)); // grows only when the scene or the settings do
    rhi::CommandList* cmd = m_device->allocateCommandList();
    m_pipeline.recordSplitWarmup(*cmd, m_frameSet, {frameOffsets, 2}, camera.zNear, camera.zFar); // after soft targets were made
    m_post.recordWarmup(*cmd); // a new grab gets the shader-read layout before set 1 binds it (M8 plan D10)
    // Passes 1a/1b (§6): the shadow maps the scene pass samples.
    m_shadows.record(*cmd, *m_scene, m_resources, m_device->transient(rhi::TransientKind::Vertex).buffer(), m_timers);
    // With a sky the clear color is never seen: it is covered by the sky pass.
    // The overdraw view adds up from black (M5 plan D10).
    const bool overdraw = settings.debugView == DebugView::Overdraw;
    // On the LDR path the clear color is what the background shows, so it takes the forward shaders' curve (M8 plan D8).
    const glm::vec3 background = fu.post.y > 0.5f ? tone::apply(settings.post.tonemapper, m_lighting.clearColor * m_lighting.exposure)
                                                  : m_lighting.clearColor;
    const float clear[4] = {overdraw ? 0.0f : background.r, overdraw ? 0.0f : background.g, overdraw ? 0.0f : background.b, 1.0f};
    // The scene pass splits only while a soft emitter (M7 plan D8) or water or glass (M8 plan D10, with the grab between
    // the two passes) is visible; the overdraw view has no depth.
    const bool grab = refractionOn && !overdraw && queueHasPass(QueuePass::Refraction);
    const bool split = (m_particleRenderer.needsSoftDepth() && m_pipeline.softDepthAvailable() && !overdraw) || grab;
    m_stats.softParticlePass = split;
    // Passes 2 and 2b (High, M8 plan D9): the opaque world's depth, copied as linear depth, becomes half-res AO that the
    // lit shader applies to indirect light; the scene pass then loads that depth instead of clearing it.
    const bool prepass = fu.post.w > 0.5f && !overdraw;
    m_stats.prepassDraws = 0;
    if (prepass) {
        const uint32_t timer = m_timers.beginPass(*cmd, "Depth prepass");
        cmd->beginRenderPass(m_pipeline.prepassPass());
        recordPrepass(*cmd, frameOffsets);
        cmd->endRenderPass();
        m_timers.endPass(*cmd, timer);
        const uint32_t copyTimer = m_timers.beginPass(*cmd, "Depth copy");
        m_pipeline.recordDepthCopy(*cmd, m_frameSet, {frameOffsets, 2}, camera.zNear, camera.zFar);
        m_timers.endPass(*cmd, copyTimer);
        m_ssao.record(*cmd, m_frameSet, {frameOffsets, 2}, post.ssaoRadius, post.ssaoIntensity, camera.zFar, m_timers);
    }
    // After the scene: pass 8 when distort emitters are visible (M8 plan D11), the P2 effects when on (D12). Both read the
    // opaque world's linear depth: pass 5c ran between A and B, or a whole pass keeps its depth for a copy now. Water and
    // glass (pass B) aren't in it, by design: haze behind a pane stays visible through it, and depth of field and motion
    // blur treat what shows through as the background it is.
    const bool rawView = dataView || overdraw;
    const bool hdrPost = settings.post.hdr && !rawView;
    const bool haze = hazeOn && !overdraw && m_particleRenderer.hazeCount() > 0;
    const bool p2 = hdrPost && m_post.p2Ready();
    const bool postDepthCopy = (haze || p2) && !split && m_pipeline.softDepthAvailable();
    const uint32_t scenePass = m_timers.beginPass(*cmd, "Scene");
    cmd->beginRenderPass(m_pipeline.scenePass(clear, split ? FramePipeline::ScenePart::Opaque : FramePipeline::ScenePart::Whole, prepass,
                                              postDepthCopy, grab));
    recordScene(*cmd, frameOffsets, m_stats.blobs > 0, settings.debugView, fu.viewProj, split, grab, camera);
    cmd->endRenderPass();
    m_timers.endPass(*cmd, scenePass);
    if (postDepthCopy) {
        const uint32_t copyTimer = m_timers.beginPass(*cmd, "Depth copy");
        m_pipeline.recordDepthCopy(*cmd, m_frameSet, {frameOffsets, 2}, camera.zNear, camera.zFar);
        m_timers.endPass(*cmd, copyTimer);
    }
    if (haze) {
        // The finished scene (resolved, particles included) is grabbed at half resolution, and the distort emitters draw
        // it back bent, faded by the linear depth.
        const uint32_t hazeTimer = m_timers.beginPass(*cmd, "Heat haze");
        m_post.recordGrab(*cmd, m_frameSet, {frameOffsets, 2});
        cmd->beginRenderPass(m_pipeline.hazePass());
        m_particleRenderer.recordHaze(*cmd, m_frameSet, {frameOffsets, 2});
        cmd->endRenderPass();
        m_timers.endPass(*cmd, hazeTimer);
    }

    // Post (ARCHITECTURE.md §6 passes 9b–9e; M8 plan D4–D8, D12). Data views (and the overdraw count) skip the chain: the
    // present pass shows their raw values. The HDR path runs the P2 effects and bloom, then ends in the tonemap, or in FXAA
    // after a tonemap into ldrColor; the LDR path ends in FXAA or the present copy.
    if (p2) {
        PostProcess::P2Frame p2Frame;
        p2Frame.previousViewProj = m_hasPreviousViewProj ? m_previousViewProj : fu.viewProj;
        p2Frame.zFar = camera.zFar;
        // The sun's screen position (backend clip space, so it lands where the sky drew it) and how far off screen it is.
        const glm::vec4 sun = fu.viewProj * glm::vec4(camera.position + glm::vec3(fu.sunDirection) * (camera.zFar * 0.5f), 1.0f);
        if (sun.w > 0.0f) {
            p2Frame.sunUv = glm::vec2(sun) / sun.w * 0.5f + 0.5f;
            const float outside = std::max(std::max(-p2Frame.sunUv.x, p2Frame.sunUv.x - 1.0f), std::max(-p2Frame.sunUv.y, p2Frame.sunUv.y - 1.0f));
            p2Frame.sunFade = 1.0f - glm::smoothstep(0.0f, 0.3f, outside);
        }
        m_post.recordP2(*cmd, m_frameSet, {frameOffsets, 2}, settings.post, p2Frame, m_timers);
    }
    if (hdrPost && settings.post.bloom) m_post.recordBloom(*cmd, m_frameSet, {frameOffsets, 2}, settings.post, m_lighting.exposure, m_timers);
    const bool fxaa = settings.post.fxaa && !rawView && (!settings.post.hdr || m_post.ldrReady());
    if (hdrPost && fxaa) m_post.recordTonemapToLdr(*cmd, m_frameSet, {frameOffsets, 2}, settings.post, m_timers);

    rhi::RenderPassDesc backbuffer;
    backbuffer.debugName = "Backbuffer";
    backbuffer.toBackbuffer = true;
    backbuffer.colorCount = 1;
    backbuffer.color[0].load = rhi::LoadOp::DontCare; // the last pass covers every pixel
    cmd->beginRenderPass(backbuffer);
    if (fxaa) {
        const uint32_t pass = m_timers.beginPass(*cmd, "FXAA");
        m_post.recordFxaa(*cmd, m_frameSet, {frameOffsets, 2}, settings.post.hdr);
        m_timers.endPass(*cmd, pass);
    } else if (hdrPost) {
        const uint32_t pass = m_timers.beginPass(*cmd, "Tonemap");
        m_post.recordTonemap(*cmd, m_frameSet, {frameOffsets, 2}, settings.post);
        m_timers.endPass(*cmd, pass);
    } else {
        const uint32_t pass = m_timers.beginPass(*cmd, "Present");
        m_pipeline.recordPresent(*cmd, m_frameSet, frameOffsets);
        m_timers.endPass(*cmd, pass);
    }
    const uint32_t uiPass = m_timers.beginPass(*cmd, "Debug UI");
    m_imgui.render(*cmd, clipSpace, m_device->caps().srgbBackbuffer);
    m_timers.endPass(*cmd, uiPass);
    cmd->endRenderPass();
    t0 = platform::nowTicks();
    m_stats.recordMs = platform::ticksToMilliseconds(t0 - t1);

    m_device->submit({&cmd, 1});
    t1 = platform::nowTicks();
    m_stats.submitMs = platform::ticksToMilliseconds(t1 - t0);
    if (screenshot && screenshotExtent && !m_device->readbackBackbuffer(*screenshot, *screenshotExtent).ok())
        FL_LOG_ERROR("Backbuffer readback failed");
    t0 = platform::nowTicks();
    m_device->present();
    t1 = platform::nowTicks();
    m_stats.presentMs = platform::ticksToMilliseconds(t1 - t0);
    m_stats.renderMs = platform::ticksToMilliseconds(t1 - frameStart);
    m_timers.endFrame();
    ++m_framesRendered;
    if (m_budgetsPending) { // this frame made the targets of the settings that came with the budgets (setMemoryBudgets)
        applyBudgets(m_budgetsRequested);
        m_budgetsPending = false;
    }
    m_previousViewProj = fu.viewProj; // the next frame's motion blur reprojects against this one
    m_hasPreviousViewProj = true;
    return true;
}

// The frame set: the two ring-backed uniform blocks plus the shadow maps and the environment's textures. Loading-time
// only: it is rebuilt when the environment replaced a texture (a map's lightmaps or sky arrived) or the shadow settings
// recreated a shadow map, the old set going through deferred destruction.
bool Renderer::rebuildFrameSet() {
    const rhi::BufferHandle ring = m_device->transient(rhi::TransientKind::Uniform).buffer();
    const rhi::ResourceWrite writes[7] = {
        {0, rhi::BufferSlice{ring, 0, uint32_t(sizeof(FrameUniforms))}, {}, {}},
        {1, rhi::BufferSlice{ring, 0, kLightArrayBytes}, {}, {}},
        {2, {}, m_shadows.cascadeTexture(), m_shadows.compareSampler()},
        {3, {}, m_shadows.spotAtlas(), m_shadows.compareSampler()},
        {4, {}, m_environment.skyCube(), m_environment.cubeSampler()},
        {5, {}, m_environment.lightmaps(), m_environment.lightmapSampler()},
        {6, {}, m_environment.shadowmask(), m_environment.lightmapSampler()},
    };
    rhi::ResourceSetDesc sd;
    sd.debugName = "frame";
    sd.layout = m_frameLayout;
    sd.writes = writes;
    const rhi::ResourceSetHandle set = m_device->createResourceSet(sd);
    if (!set) {
        FL_LOG_ERROR("The frame resource set could not be created");
        return false;
    }
    if (m_frameSet) m_device->destroy(m_frameSet);
    m_frameSet = set;
    m_frameSetRevision = m_environment.revision();
    m_frameSetShadowRevision = m_shadows.revision();
    return true;
}

// Set 1 of the lit pipelines (ARCHITECTURE.md §4.6 "pass inputs"; M8 plan D9, D10): binding 1 the linear depth,
// binding 2 the SSAO result, binding 3 the refraction grab (binding 4 is reserved). One set for scene pass A (and a
// whole pass), whose grab isn't taken yet, and one for pass B. Rebuilt only when one of the textures changes (targets
// recreated, an effect switched on or off), never in steady frames.
bool Renderer::rebuildForwardSets() {
    const rhi::TextureHandle depth = m_pipeline.linearDepth() ? m_pipeline.linearDepth() : m_post.blackTexture();
    const rhi::TextureHandle ssao = m_ssao.ready() ? m_ssao.result() : m_post.whiteTexture(); // white: no occlusion
    const rhi::TextureHandle grab = m_post.grab() ? m_post.grab() : m_post.blackTexture();
    if (m_forwardSets[0] && depth == m_forwardInputs[0] && ssao == m_forwardInputs[1] && grab == m_forwardInputs[2]) return true;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        const rhi::ResourceWrite writes[4] = {{1, {}, depth, m_pointClamp},
                                              {2, {}, ssao, m_post.linearClamp()},
                                              {3, {}, pass == 1 ? grab : m_post.blackTexture(), m_post.linearClamp()},
                                              {4, {}, m_post.blackTexture(), m_post.linearClamp()}};
        rhi::ResourceSetDesc sd;
        sd.debugName = pass == 0 ? "forward inputs (pass A)" : "forward inputs (pass B)";
        sd.layout = m_forwardLayout;
        sd.writes = writes;
        const rhi::ResourceSetHandle set = m_device->createResourceSet(sd);
        if (!set) {
            FL_LOG_ERROR("The forward input set could not be created");
            return false;
        }
        m_device->destroy(m_forwardSets[pass]); // deferred: frames in flight keep the old set
        m_forwardSets[pass] = set;
    }
    m_forwardInputs[0] = depth;
    m_forwardInputs[1] = ssao;
    m_forwardInputs[2] = grab;
    return true;
}

// The frame's local lights go into one 16 KB LightArray block in the uniform ring (M2 plan D4).
bool Renderer::uploadLights(const CameraData& camera, float time, uint32_t& lightOffset) {
    FL_ZONE("Renderer::uploadLights");
    m_lightCandidates.clear();
    // A light whose range doesn't reach the camera frustum can't light a visible pixel (M5 plan D9).
    uint32_t inScene = 0;
    m_scene->forEachLight([&](LightId, const LightDesc& light) {
        ++inScene;
        if (m_visibility.cameraSees(glm::vec4(light.position, light.range))) m_lightCandidates.push_back(&light);
    });
    m_stats.lightsInScene = inScene;
    m_stats.lightsVisible = uint32_t(m_lightCandidates.size());
    m_stats.lightsDropped = 0;
    if (m_lightCandidates.size() > kMaxFrameLights) {
        // Light indices are 8 bits: keep the lights whose range comes nearest to the camera.
        auto gap = [&](const LightDesc* l) { return std::max(0.0f, glm::length(l->position - camera.position) - l->range); };
        std::nth_element(m_lightCandidates.begin(), m_lightCandidates.begin() + kMaxFrameLights, m_lightCandidates.end(),
                         [&](const LightDesc* a, const LightDesc* b) { return gap(a) < gap(b); });
        m_stats.lightsDropped = uint32_t(m_lightCandidates.size()) - kMaxFrameLights;
        m_lightCandidates.resize(kMaxFrameLights);
        if (!m_warnedLightOverflow) {
            FL_LOG_WARN("%u local lights in the scene; only the %u nearest to the camera are used", m_stats.lightsInScene, kMaxFrameLights);
            m_warnedLightOverflow = true;
        }
    }
    // Spot lights that get a shadow-atlas tile this frame (High; M4 plan D8).
    m_shadows.assignSpotShadows(m_lightCandidates, camera.position, time, m_spotSlots);
    // Always the whole block: the bound range is fixed at 16 KB, so it must lie inside the ring on both backends.
    const rhi::TransientRing::Allocation block = m_device->transient(rhi::TransientKind::Uniform).allocate(kLightArrayBytes);
    if (!block.cpu) return false;
    auto* gpu = static_cast<GpuLight*>(block.cpu);
    m_frameLights.clear();
    for (size_t i = 0; i < m_lightCandidates.size(); ++i) {
        GpuLight light = toGpuLight(*m_lightCandidates[i]); // built here, written once: ring memory is write-only
        light.spotParams.y = float(m_spotSlots[i]);
        gpu[i] = light;
        m_frameLights.push_back(toFrameLight(*m_lightCandidates[i]));
    }
    m_stats.lightsUploaded = uint32_t(m_frameLights.size());
    lightOffset = block.gpu.offset;
    return true;
}

void Renderer::buildQueue(const CameraData& camera, const FrameSettings& settings, bool refraction) {
    FL_ZONE("Renderer::buildQueue");
    m_queue.reset();
    m_shadows.resetCasters();
    const Scene& scene = *m_scene;
    // These grow only when the scene does (loading), never in a steady frame.
    if (m_dynamicLightLists.size() < scene.dynamics().size()) m_dynamicLightLists.resize(scene.dynamics().size());
    if (m_dynamicProbes.size() < scene.dynamics().size()) m_dynamicProbes.resize(scene.dynamics().size());
    if (m_staticLightLists.size() < scene.staticRows().size()) m_staticLightLists.resize(scene.staticRows().size());
    for (WorkerStats& w : m_workerStats) w = WorkerStats{};

    BuildContext ctx;
    ctx.self = this;
    ctx.scene = &scene;
    ctx.eye = camera.position;
    ctx.invFar = 1.0f / camera.zFar;
    ctx.instancing = settings.instancing;
    // Refraction off (or LDR, or without the grab or a sampleable depth): refractive materials draw as their blended twins
    // among the transparents (M8 plan D10). The same flag gates the grab and the shader's refraction branch.
    ctx.refraction = refraction;
    ctx.maxLights = std::min<uint32_t>(settings.maxLocalLights, kMaxLightsPerObject);
    ctx.staticItems = settings.instancing ? uint32_t(scene.staticGroups().size()) : uint32_t(scene.staticRows().size());
    ctx.lights = m_frameLights;
    ctx.rowMasks = m_visibility.rowMasks();
    ctx.dynMasks = m_visibility.dynamicMasks();
    ctx.rowLods = m_visibility.rowLods();
    ctx.dynLods = m_visibility.dynamicLods();
    ctx.rowShadowLevels = m_visibility.rowShadowLevels();
    ctx.dynShadowLevels = m_visibility.dynamicShadowLevels();
    ctx.shadowBits = uint8_t(m_visibility.activeViewBits() & ~1u);
    m_jobs->parallelFor(uint32_t(scene.staticRows().size()), 64, staticLightsRange, &ctx);
    m_jobs->parallelFor(ctx.staticItems + uint32_t(scene.dynamics().size()), 16, buildRange, &ctx);
    m_particleQueueItems = m_particleRenderer.queueDraws(m_queue); // emitters: transparent items keyed by their center depth (M7 plan D5)
    m_queue.sort();
    m_shadows.sortCasters();

    m_stats.queueItems = m_queue.size();
    m_stats.queueDropped = m_queue.dropped();
    if (m_stats.queueDropped) FL_LOG_WARN("Render queue full: %u draws dropped this frame", m_stats.queueDropped);
    m_stats.lightAssignments = 0;
    m_stats.litObjects = m_stats.maxObjectLights = 0;
    m_stats.probeSamples = 0;
    for (const WorkerStats& w : m_workerStats) {
        m_stats.lightAssignments += w.assignments;
        m_stats.litObjects += w.litObjects;
        m_stats.maxObjectLights = std::max(m_stats.maxObjectLights, w.maxLights);
        m_stats.probeSamples += w.probeSamples;
    }
}

// The command bytes of the largest frame this scene can record with these settings (M5): what the camera happens to see
// never enters, so flying into a dense block can't grow a command list in a steady frame; it grows when the scene or
// the settings do. Draw counts follow buildRange: a blended group draws each visible row, another group at most two
// buckets per level (steady and fading); a row or dynamic submesh drawn on its own at most twice (a fade); a casting
// group at most one bucket per level in each shadow view, a row or dynamic submesh one draw per view.
size_t Renderer::commandBytesBound(const FrameSettings& settings) const {
    const Scene& scene = *m_scene;
    const uint64_t views = std::min<uint64_t>(settings.shadows.cascades, kMaxCascades) + (settings.shadows.spotShadows ? kMaxSpotShadows : 0);
    uint64_t camera = 0, casters = 0;
    if (settings.instancing) {
        for (const Scene::StaticGroup& g : scene.staticGroups()) {
            camera += std::min<uint64_t>(2ull * g.rowCount, std::max<uint64_t>(g.rowCount, 2ull * g.lodCount));
            casters += views * std::min<uint64_t>(g.rowCount, g.lodCount);
        }
    } else {
        camera = 2ull * scene.staticRows().size();
        casters = views * scene.staticRows().size();
    }
    if (!settings.shadows.staticCasters) casters = 0;
    for (const Scene::Dynamic& d : scene.dynamics()) {
        camera += 2ull * d.submeshCount;
        casters += views * d.submeshCount;
    }
    // The wireframe and overdraw views draw the camera's items once more, and so does the depth prepass (M8 plan D9); 64
    // draws for the sky, blobs and the full-screen passes, and 512 KB for render passes, timer queries and the debug UI.
    const bool debugPass = settings.debugView == DebugView::Wireframe || settings.debugView == DebugView::Overdraw;
    const bool prepass = settings.post.ssao && settings.post.hdr;
    const uint64_t draws = camera * (1 + (debugPass ? 1 : 0) + (prepass ? 1 : 0)) + casters + 64;
    return size_t(draws * rhi::CommandList::maxBytesPerDraw() + 512 * 1024 + SpriteRenderer::kMaxCommandBytes +
                  ParticleRenderer::kMaxCommandBytes);
}

// Phase 1: light lists of the static rows the camera sees (the others draw no lit pixels).
void Renderer::staticLightsRange(void* context, uint32_t begin, uint32_t end, uint32_t worker) {
    const BuildContext& c = *static_cast<const BuildContext*>(context);
    Renderer& self = *c.self;
    WorkerStats& stats = self.m_workerStats[worker];
    const auto spheres = c.scene->staticSpheres();
    for (uint32_t row = begin; row < end; ++row) {
        if (!(c.rowMasks[row] & 1u)) continue;
        const glm::uvec4 list = selectLights(c.lights, spheres[row], true, c.maxLights);
        self.m_staticLightLists[row] = list;
        countLights(stats.assignments, stats.litObjects, stats.maxLights, list);
    }
}

namespace {

// Up to N queue items of one object or group, reserved together (one atomic add) once they are known.
template <uint32_t N>
struct ItemBatch {
    DrawItem items[N];
    uint64_t keys[N];
    uint32_t count = 0;
    void add(RenderQueue& queue, const DrawItem& item, uint64_t key) {
        if (count == N) flush(queue);
        items[count] = item;
        keys[count++] = key;
    }
    void flush(RenderQueue& queue) {
        if (count == 0) return;
        const uint32_t slot = queue.reserve(count);
        if (slot != RenderQueue::kFull)
            for (uint32_t k = 0; k < count; ++k) queue.set(slot + k, keys[k], items[k]);
        count = 0;
    }
};

// The shadow views (0-6) in a visibility mask: bit v + 1 = view v.
template <typename Fn>
void forEachShadowView(uint32_t bits, Fn&& fn) {
    for (uint32_t v = 0; v < kMaxCullViews - 1; ++v)
        if (bits & (2u << v)) fn(v);
}

} // namespace

// Phase 2 (M5 plan D7, D8): queue items, instance data and shadow casters. A static group's visible rows are bucketed by
// (level, fading) for the camera, up to 8 instanced draws each with its own instance data, and by (view, level) for
// the shadow views, whose draws get rows-only instance data of their own. Dynamic renderables and rows drawn one by
// one get a draw per level (two while fading) and a caster draw per shadow view that sees them.
void Renderer::buildRange(void* context, uint32_t begin, uint32_t end, uint32_t worker) {
    const BuildContext& c = *static_cast<const BuildContext*>(context);
    Renderer& self = *c.self;
    const Scene& scene = *c.scene;
    const MaterialSystem& materials = self.m_materials;
    const Environment& environment = self.m_environment;
    RenderQueue& queue = self.m_queue;
    RenderQueue& casters = self.m_shadows.casterQueue();
    rhi::TransientRing& vertexRing = self.m_device->transient(rhi::TransientKind::Vertex);
    WorkerStats& stats = self.m_workerStats[worker];
    const auto groups = scene.staticGroups();
    const auto dynamics = scene.dynamics();
    ItemBatch<8> camera;
    ItemBatch<32> shadow;

    // Opaque keys sort front to back by the nearest point; blended ones back to front by the center (M2 plan D6).
    auto nearestDepth = [&](const glm::vec4& s) { return std::max(0.0f, glm::length(glm::vec3(s) - c.eye) - s.w) * c.invFar; };
    auto centerDepth = [&](const glm::vec4& s) { return glm::length(glm::vec3(s) - c.eye) * c.invFar; };
    // Fills the pipeline, material and reflection slot of a camera item and returns its sort key; fading draws use the
    // material's cross-fade pipeline.
    auto keyFor = [&](const MaterialSystem::DrawState& ds, MaterialHandle material, const glm::vec4& sphere, float nearest,
                      uint8_t layer, uint32_t reflectionSlot, bool fading, DrawItem& item) {
        const uint16_t pipeline = fading ? ds.fadePipelineSlot : ds.pipelineSlot;
        const QueuePass pass = fading ? ds.fadePass : ds.pass;
        item.material = material.bits;
        item.pipeline = pipeline;
        item.flags |= (reflectionSlot & 0xFu) << DrawItem::kReflectionShift;
        const uint8_t sub = uint8_t(item.submesh);
        return pass == QueuePass::Transparent
                   ? sortkey::blended(QueuePass::Transparent, layer, centerDepth(sphere), pipeline, ds.sortId, sub)
                   : sortkey::opaque(pass, layer, pass == QueuePass::AlphaTest ? Translucency::AlphaTest : Translucency::Opaque, pipeline,
                                     ds.sortId, nearest, sub);
    };
    // A draw can cross-fade only through a pipeline that dithers: an alpha-tested material's own, or an opaque one's
    // alpha-test twin. Blended materials, and any material without a twin (the error material), switch levels at once.
    auto canFade = [](const MaterialSystem::DrawState& ds) {
        return ds.pass != QueuePass::Transparent && (ds.pass == QueuePass::AlphaTest || ds.fadePipelineSlot != ds.pipelineSlot);
    };
    // One draw of an object at its camera level, or two while it fades (incoming and outgoing, complementary dither).
    auto emitLevels = [&](const MaterialSystem::DrawState& ds, MaterialHandle material, const glm::vec4& sphere, uint8_t layer,
                          uint32_t reflectionSlot, const Visibility::DrawLod& lod, DrawItem item) {
        const bool fading = lod.fading && canFade(ds);
        const float progress = float(lod.progress) / 255.0f;
        const float nearest = nearestDepth(sphere);
        DrawItem incoming = item;
        incoming.lod = uint16_t(packLodWord(lod.level, fading, true, progress));
        camera.add(queue, incoming, keyFor(ds, material, sphere, nearest, layer, reflectionSlot, fading, incoming));
        if (fading) {
            DrawItem outgoing = item;
            outgoing.lod = uint16_t(packLodWord(lod.previous, true, false, progress));
            camera.add(queue, outgoing, keyFor(ds, material, sphere, nearest, layer, reflectionSlot, true, outgoing));
        }
    };
    auto emitCaster = [&](uint32_t view, const MaterialSystem::CasterState& cs, uint32_t meshRef, uint32_t submesh, uint32_t level,
                          bool instanced, uint32_t perDrawOrInstance, uint32_t instanceCount, uint32_t flags, bool skinned = false) {
        DrawItem item;
        item.mesh = meshRef;
        item.submesh = uint16_t(submesh);
        item.lod = uint16_t(level);
        item.material = cs.set.bits;
        item.pipeline = ShadowMaps::casterPipeline(instanced, cs.alphaTest, skinned);
        item.perDrawOrInstance = perDrawOrInstance;
        item.instanceCount = instanceCount;
        item.objectIndex = view;
        item.flags = flags;
        shadow.add(casters, item, ShadowMaps::casterKey(view, item.pipeline, cs.sortId, meshRef, submesh, level));
    };
    auto rowItem = [](const Scene::StaticGroup& g, uint32_t row) {
        DrawItem item;
        item.mesh = g.meshRef;
        item.submesh = uint16_t(g.submesh);
        item.perDrawOrInstance = row;
        item.objectIndex = row;
        item.flags = DrawItem::kStaticRow;
        return item;
    };
    auto rowVariant = [](const Scene::StaticGroup& g) { return g.lightmapped ? ObjectVariant::Lightmap : ObjectVariant::Dynamic; };
    // Refraction off (M8 plan D10): a refractive material's blended twin, sorted with the transparents. Always true, so it
    // can sit in a condition.
    auto effective = [&](MaterialSystem::DrawState& ds) {
        if (!c.refraction && ds.pass == QueuePass::Refraction) {
            ds.pipelineSlot = ds.fadePipelineSlot = ds.blendTwinSlot;
            ds.pass = ds.fadePass = ds.blendTwinPass;
        }
        return true;
    };

    for (uint32_t i = begin; i < end; ++i) {
        if (i < c.staticItems && c.instancing) {
            const Scene::StaticGroup& g = groups[i];
            MaterialSystem::DrawState rowState, groupState;
            if (!materials.drawState(g.material, rowVariant(g), g.mirrored, rowState)) continue;
            effective(rowState);
            if (rowState.pass == QueuePass::Transparent || rowState.pass == QueuePass::Refraction) {
                // One draw per visible instance, so blended instances sort back to front; blended materials cast nothing.
                for (uint32_t r = g.firstRow; r < g.firstRow + g.rowCount; ++r)
                    if (c.rowMasks[r] & 1u)
                        emitLevels(rowState, g.material, scene.staticSpheres()[r], g.layer, g.reflectionSlot, c.rowLods[r], rowItem(g, r));
                camera.flush(queue);
                continue;
            }
            const ObjectVariant variant = g.lightmapped ? ObjectVariant::LightmapInstanced : ObjectVariant::Instanced;
            if (!materials.drawState(g.material, variant, g.mirrored, groupState)) continue;
            effective(groupState);
            MaterialSystem::CasterState cs;
            const bool casts = c.shadowBits && g.castsShadows && materials.casterState(g.material, cs) && cs.casts;
            const bool groupFades = canFade(groupState);

            // Count: camera buckets [level] steady, [4 + level] fading; shadow buckets [view][level].
            uint32_t camCount[8] = {}, shadowCount[kMaxCullViews - 1][kMaxLodLevels] = {};
            float nearest = 1.0f;
            glm::vec4 nearestSphere(0.0f);
            bool anyVisible = false;
            for (uint32_t r = g.firstRow; r < g.firstRow + g.rowCount; ++r) {
                const uint8_t mask = c.rowMasks[r];
                if (mask & 1u) {
                    const Visibility::DrawLod& lod = c.rowLods[r];
                    if (lod.fading && groupFades) {
                        ++camCount[4 + lod.level];
                        ++camCount[4 + lod.previous];
                    } else {
                        ++camCount[lod.level];
                    }
                    const float d = nearestDepth(scene.staticSpheres()[r]);
                    if (!anyVisible || d < nearest) {
                        nearest = d;
                        nearestSphere = scene.staticSpheres()[r];
                    }
                    anyVisible = true;
                }
                if (casts) forEachShadowView(mask & c.shadowBits, [&](uint32_t v) { ++shadowCount[v][c.rowShadowLevels[r]]; });
            }

            uint32_t camTotal = 0, camOffset[8];
            for (uint32_t b = 0; b < 8; ++b) camOffset[b] = camTotal, camTotal += camCount[b];
            if (camTotal) {
                const rhi::TransientRing::Allocation a = vertexRing.allocate(camTotal * uint32_t(sizeof(InstanceData)));
                if (a.cpu) {
                    auto* instances = static_cast<InstanceData*>(a.cpu);
                    uint32_t cursor[8];
                    std::copy(camOffset, camOffset + 8, cursor);
                    for (uint32_t r = g.firstRow; r < g.firstRow + g.rowCount; ++r) {
                        if (!(c.rowMasks[r] & 1u)) continue;
                        // Ring memory is write-combined on Vulkan: build the record here and write it once, never read it back.
                        InstanceData record;
                        record.rows = scene.staticRows()[r];
                        record.lightmapST = g.lightmapped ? scene.staticLightmapST()[r] : glm::vec4(0.0f);
                        record.lights = self.m_staticLightLists[r];
                        record.lights.z |= scene.staticDrawFlags()[r];
                        const Visibility::DrawLod& lod = c.rowLods[r];
                        const float progress = float(lod.progress) / 255.0f;
                        if (!(lod.fading && groupFades)) {
                            record.lights.w = packLodWord(lod.level, false, false, 1.0f);
                            instances[cursor[lod.level]++] = record;
                        } else {
                            record.lights.w = packLodWord(lod.level, true, true, progress);
                            instances[cursor[4 + lod.level]++] = record;
                            record.lights.w = packLodWord(lod.previous, true, false, progress);
                            instances[cursor[4 + lod.previous]++] = record;
                        }
                    }
                    for (uint32_t b = 0; b < 8; ++b) {
                        if (!camCount[b]) continue;
                        DrawItem item;
                        item.mesh = g.meshRef;
                        item.submesh = uint16_t(g.submesh);
                        item.lod = uint16_t(packLodWord(b & 3u, b >= 4, false, 0.0f));
                        item.perDrawOrInstance = a.gpu.offset + camOffset[b] * uint32_t(sizeof(InstanceData));
                        item.instanceCount = camCount[b];
                        item.objectIndex = i;
                        item.flags = DrawItem::kInstanced;
                        camera.add(queue, item, keyFor(groupState, g.material, nearestSphere, nearest, g.layer, g.reflectionSlot, b >= 4, item));
                    }
                }
            }

            uint32_t shadowTotal = 0, shadowOffset[kMaxCullViews - 1][kMaxLodLevels];
            for (uint32_t v = 0; v < kMaxCullViews - 1; ++v)
                for (uint32_t l = 0; l < kMaxLodLevels; ++l) shadowOffset[v][l] = shadowTotal, shadowTotal += shadowCount[v][l];
            if (shadowTotal) {
                const rhi::TransientRing::Allocation a = vertexRing.allocate(shadowTotal * uint32_t(sizeof(InstanceRows)));
                if (a.cpu) {
                    auto* rows = static_cast<InstanceRows*>(a.cpu);
                    uint32_t cursor[kMaxCullViews - 1][kMaxLodLevels];
                    std::copy(&shadowOffset[0][0], &shadowOffset[0][0] + (kMaxCullViews - 1) * kMaxLodLevels, &cursor[0][0]);
                    for (uint32_t r = g.firstRow; r < g.firstRow + g.rowCount; ++r)
                        forEachShadowView(c.rowMasks[r] & c.shadowBits,
                                          [&](uint32_t v) { rows[cursor[v][c.rowShadowLevels[r]]++] = scene.staticRows()[r]; });
                    for (uint32_t v = 0; v < kMaxCullViews - 1; ++v)
                        for (uint32_t l = 0; l < kMaxLodLevels; ++l)
                            if (shadowCount[v][l])
                                emitCaster(v, cs, g.meshRef, g.submesh, l, true, a.gpu.offset + shadowOffset[v][l] * uint32_t(sizeof(InstanceRows)),
                                           shadowCount[v][l], DrawItem::kInstanced);
                }
            }
        } else if (i < c.staticItems) {
            // Instancing off (measurement mode): each static row on its own.
            const uint32_t row = i;
            const Scene::StaticGroup& g = groups[scene.staticRowGroups()[row]];
            const uint8_t mask = c.rowMasks[row];
            MaterialSystem::DrawState ds;
            if ((mask & 1u) && materials.drawState(g.material, rowVariant(g), g.mirrored, ds) && effective(ds))
                emitLevels(ds, g.material, scene.staticSpheres()[row], g.layer, g.reflectionSlot, c.rowLods[row], rowItem(g, row));
            MaterialSystem::CasterState cs;
            if ((mask & c.shadowBits) && g.castsShadows && materials.casterState(g.material, cs) && cs.casts)
                forEachShadowView(mask & c.shadowBits, [&](uint32_t v) {
                    emitCaster(v, cs, g.meshRef, g.submesh, c.rowShadowLevels[row], false, row, 1, DrawItem::kStaticRow);
                });
        } else {
            const uint32_t d = i - c.staticItems;
            const Scene::Dynamic& r = dynamics[d];
            const uint8_t mask = c.dynMasks[d];
            // A skinned renderable without a palette this frame (the uniform ring was full) draws nothing anywhere.
            if (r.skinned && r.paletteOffset == Scene::kNoPalette) continue;
            const ObjectVariant variant = r.skinned ? ObjectVariant::Skinned : ObjectVariant::Dynamic;
            const uint32_t drawFlags = DrawItem::kDynamic | (r.skinned ? uint32_t(DrawItem::kSkinned) : 0u);
            if (mask & 1u) {
                const glm::uvec4 list = selectLights(c.lights, r.sphere, false, c.maxLights);
                self.m_dynamicLightLists[d] = list;
                countLights(stats.assignments, stats.litObjects, stats.maxLights, list);
                // Probe lighting at the center of the object, and the reflection probe whose box holds it (M3 plan D6, D8).
                ProbeSample& probe = self.m_dynamicProbes[d];
                if (environment.probes().empty()) {
                    probe = self.m_ambientProbe;
                } else {
                    environment.probes().sample(glm::vec3(r.sphere), probe.sh, probe.sunVisibility);
                    ++stats.probeSamples;
                }
                probe.reflectionSlot = environment.selectReflection(glm::vec3(r.sphere));
                for (uint32_t s = 0; s < r.submeshCount; ++s) {
                    const MaterialHandle material = scene.dynamicMaterials()[r.firstMaterial + s];
                    MaterialSystem::DrawState ds;
                    if (!materials.drawState(material, variant, r.mirrored, ds)) continue;
                    effective(ds);
                    DrawItem item;
                    item.mesh = r.meshRef;
                    item.submesh = uint16_t(s);
                    item.perDrawOrInstance = d;
                    item.objectIndex = d;
                    item.flags = drawFlags;
                    emitLevels(ds, material, r.sphere, r.layer, probe.reflectionSlot, c.dynLods[d], item);
                }
            }
            if ((mask & c.shadowBits) && r.castsShadows)
                for (uint32_t s = 0; s < r.submeshCount; ++s) {
                    MaterialSystem::CasterState cs;
                    if (!materials.casterState(scene.dynamicMaterials()[r.firstMaterial + s], cs) || !cs.casts) continue;
                    forEachShadowView(mask & c.shadowBits, [&](uint32_t v) {
                        emitCaster(v, cs, r.meshRef, s, c.dynShadowLevels[d], false, d, 1, drawFlags, r.skinned);
                    });
                }
        }
        camera.flush(queue);
        shadow.flush(casters);
    }
}

// Whether the sorted camera queue holds a draw of `pass` (the pass is the key's top four bits): a binary search.
bool Renderer::queueHasPass(QueuePass pass) const {
    uint32_t lo = 0, hi = m_queue.size();
    while (lo < hi) { // the first item whose pass is not below `pass`
        const uint32_t mid = lo + (hi - lo) / 2;
        if ((m_queue.sortedKey(mid) >> 60) < uint64_t(pass)) lo = mid + 1;
        else hi = mid;
    }
    return lo < m_queue.size() && (m_queue.sortedKey(lo) >> 60) == uint64_t(pass);
}

// Pass 2 (M8 plan D9): the queue's opaque segment again, depth only, plus the LOD fade twins of opaque materials from
// the alpha-test segment with their dither (so a fading object keeps its AO). Alpha-tested materials stay out (pass 4
// writes their depth). Materials don't matter here, so only the transforms (and the LOD word) are bound.
void Renderer::recordPrepass(rhi::CommandList& cmd, const uint32_t frameOffsets[2]) {
    FL_ZONE("Renderer::recordPrepass");
    const Scene& scene = *m_scene;
    const rhi::BufferHandle vertexRing = m_device->transient(rhi::TransientKind::Vertex).buffer();
    rhi::PipelineHandle boundPipeline;
    uint32_t boundMesh = ~0u;
    bool skinBound = false;
    PerDraw perDraw;
    for (uint32_t i = 0; i < m_queue.size() && (m_queue.sortedKey(i) >> 60) <= uint64_t(QueuePass::AlphaTest); ++i) {
        const DrawItem& item = m_queue.sortedItem(i);
        if (item.flags == 0 || (item.flags & DrawItem::kParticles)) continue;
        const rhi::PipelineHandle pipeline = m_materials.prepassPipeline(uint16_t(item.pipeline));
        if (!pipeline) continue;
        const Scene::MeshRef& ref = scene.meshRefs()[item.mesh];
        const ModelAsset* model = m_resources.model(ref.model);
        if (!model || model->state != LoadState::Ready) continue;
        const GpuMesh* mesh = &model->meshes[ref.mesh];
        if (pipeline != boundPipeline) {
            cmd.bindPipeline(pipeline);
            if (!boundPipeline) cmd.bindResourceSet(0, m_frameSet, {frameOffsets, 2});
            boundPipeline = pipeline;
        }
        const bool skinned = (item.flags & DrawItem::kSkinned) != 0;
        if (skinned) {
            const uint32_t palette = scene.dynamics()[item.perDrawOrInstance].paletteOffset;
            const rhi::ResourceSetHandle set = m_environment.skinnedSet((item.flags >> DrawItem::kReflectionShift) & 0xFu);
            if (palette == Scene::kNoPalette || !set) continue;
            cmd.bindResourceSet(3, set, {&palette, 1});
        }
        if (item.mesh != boundMesh || (skinned && !skinBound)) {
            const rhi::BufferSlice streams[3] = {mesh->positions, mesh->surface, mesh->skin};
            cmd.bindVertexStreams(0, {streams, size_t(skinned ? 3 : 2)});
            cmd.bindIndexBuffer(mesh->indices, mesh->indexType);
            skinBound = skinned;
            boundMesh = item.mesh;
        }
        const GpuSubmesh& sub = model->submeshes[mesh->firstSubmesh + item.submesh];
        const uint32_t level = std::min<uint32_t>(lodWordLevel(item.lod), mesh->lodCount - 1);
        if (item.flags & DrawItem::kInstanced) {
            const rhi::BufferSlice data{vertexRing, item.perDrawOrInstance, item.instanceCount * uint32_t(sizeof(InstanceData))};
            cmd.bindVertexStreams(2, {&data, 1});
            skinBound = false;
            cmd.drawIndexed(sub.lodCount[level], item.instanceCount, sub.lodFirst[level]);
        } else {
            perDraw.model = item.flags & DrawItem::kStaticRow ? scene.staticRows()[item.perDrawOrInstance]
                                                              : scene.dynamics()[item.perDrawOrInstance].rows;
            perDraw.lights.w = item.lod; // the FADE variant's dither (instanced draws carry it per instance)
            cmd.setPerDraw(&perDraw, uint8_t(sizeof(perDraw)));
            cmd.drawIndexed(sub.lodCount[level], 1, sub.lodFirst[level]);
        }
        ++m_stats.prepassDraws;
    }
}

void Renderer::recordScene(rhi::CommandList& cmd, const uint32_t frameOffsets[2], bool drawBlobs, DebugView view,
                           const glm::mat4& viewProj, bool split, bool grab, const CameraData& camera) {
    FL_ZONE("Renderer::recordScene");
    m_stats.sceneDraws = m_stats.instances = m_stats.transparentDraws = m_stats.skinnedDraws = m_stats.particleSceneDraws = 0;
    m_stats.refractionDraws = 0;
    m_stats.triangles = 0;
    m_stats.skyDrawn = false;
    const Scene& scene = *m_scene;
    const rhi::BufferHandle vertexRing = m_device->transient(rhi::TransientKind::Vertex).buffer();
    if (view == DebugView::Overdraw) { // every draw adds a constant; the present pass maps the count to a heat ramp
        const uint32_t timer = m_timers.beginPass(cmd, "Overdraw");
        m_debugDraw.record(cmd, DebugDraw::Mode::Overdraw, m_queue, scene, m_resources, m_materials, vertexRing, viewProj);
        // Particles add one step per covered pixel of every quad (M7 plan D13): the fill they cost.
        m_particleRenderer.recordOverdraw(cmd, m_frameSet, {frameOffsets, 2});
        m_timers.endPass(cmd, timer);
        m_stats.sceneDraws = m_queue.size() - m_particleQueueItems; // mesh draws, as the lit path counts them
        m_stats.particleSceneDraws = m_particleQueueItems;
        return;
    }
    uint32_t boundPipeline = ~0u, boundMaterial = 0, boundMesh = ~0u, boundReflection = ~0u;
    rhi::ResourceSetHandle forwardSet = m_forwardSets[0]; // set 1 of the lit pipelines (M8 plan D9, D10)
    const GpuMesh* boundGpuMesh = nullptr; // these three always describe the same mesh
    bool skinBound = false; // stream 2 holds boundMesh's skin stream (an instanced draw puts instance data there)
    const ModelAsset* boundModel = nullptr;
    PerDraw perDraw;

    // Frame order (ARCHITECTURE.md §6): opaque front to back, alpha test, the sky, then blended back to front. The sorted
    // queue already holds the draws in that order (pass is the top key field); each range gets its own CPU/GPU timer.
    static constexpr struct {
        QueuePass pass;
        const char* name;
    } kSegments[] = {{QueuePass::Opaque, "Opaque"}, {QueuePass::AlphaTest, "Alpha test"}, {QueuePass::Refraction, "Refraction"},
                     {QueuePass::Transparent, "Transparent"}};
    uint32_t i = 0;
    for (const auto& segment : kSegments) {
        if (segment.pass == QueuePass::Refraction && m_environment.hasSky()) {
            // Pass 5: the sky fills what opaques left at the far plane, before transparents blend over it.
            const uint32_t skyTimer = m_timers.beginPass(cmd, "Sky");
            m_pipeline.recordSky(cmd, m_frameSet, {frameOffsets, 2});
            m_timers.endPass(cmd, skyTimer);
            m_stats.skyDrawn = true;
            boundPipeline = ~0u; // the sky has its own layout: rebind the pipeline and every set
            boundMaterial = 0;
            boundReflection = ~0u;
        }
        if (segment.pass == QueuePass::Refraction && drawBlobs) {
            // Pass 5b (DEV-3; M4 plan D9): blob shadows darken what opaques and the sky left, under the transparents.
            const uint32_t blobTimer = m_timers.beginPass(cmd, "Blob shadows");
            m_blobs.record(cmd, m_frameSet, {frameOffsets, 2});
            m_timers.endPass(cmd, blobTimer);
            boundPipeline = ~0u;
            boundMaterial = 0;
            boundReflection = ~0u;
            boundMesh = ~0u; // the blob draw rebinds vertex stream 0
        }
        if (segment.pass == QueuePass::Refraction && split) {
            // Pass 5c (M7 plan D8): pass A ends with the opaque world stored, its depth becomes linear view depth for soft
            // particles and refraction, the resolved world is grabbed at half resolution when water or glass is visible
            // (pass 5d, M8 plan D10), and pass B continues on the same color and depth: refractive surfaces write depth,
            // everything after them only tests it.
            cmd.endRenderPass();
            const uint32_t copyTimer = m_timers.beginPass(cmd, "Depth copy");
            m_pipeline.recordDepthCopy(cmd, m_frameSet, {frameOffsets, 2}, camera.zNear, camera.zFar);
            m_timers.endPass(cmd, copyTimer);
            if (grab) {
                const uint32_t grabTimer = m_timers.beginPass(cmd, "Grab");
                m_post.recordGrab(cmd, m_frameSet, {frameOffsets, 2});
                m_timers.endPass(cmd, grabTimer);
            }
            const float unused[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            cmd.beginRenderPass(m_pipeline.scenePass(unused, FramePipeline::ScenePart::Blended));
            forwardSet = m_forwardSets[1]; // pass B: the refraction grab is taken
            boundPipeline = ~0u; // a new render pass: every binding starts over
            boundMaterial = 0;
            boundReflection = ~0u;
            boundMesh = ~0u;
            skinBound = false;
        }
        const uint32_t timer = m_timers.beginPass(cmd, segment.name);
        for (; i < m_queue.size() && (m_queue.sortedKey(i) >> 60) <= uint64_t(segment.pass); ++i) {
            const DrawItem& item = m_queue.sortedItem(i);
            if (item.flags == 0) continue;
            if (item.flags & DrawItem::kParticles) { // one emitter (M7 plan D5), interleaved with the transparent meshes
                m_particleRenderer.record(cmd, item.perDrawOrInstance, m_frameSet, {frameOffsets, 2}, m_pipeline.softSet(), split);
                ++m_stats.particleSceneDraws;
                boundPipeline = ~0u; // its own pipeline, sets and vertex stream 0
                boundMaterial = 0;
                boundReflection = ~0u;
                boundMesh = ~0u;
                skinBound = false;
                continue;
            }
            const bool instanced = (item.flags & DrawItem::kInstanced) != 0;
            const ModelAsset* model = boundModel;
            const GpuMesh* mesh = boundGpuMesh;
            if (item.mesh != boundMesh) {
                const Scene::MeshRef& ref = scene.meshRefs()[item.mesh];
                model = m_resources.model(ref.model);
                if (!model || model->state != LoadState::Ready) continue;
                mesh = &model->meshes[ref.mesh];
            }
            if (item.pipeline != boundPipeline) {
                cmd.bindPipeline(m_materials.pipeline(uint16_t(item.pipeline)));
                if (boundPipeline == ~0u) { // sets 0 and 1 are shared by every lit layout
                    cmd.bindResourceSet(0, m_frameSet, {frameOffsets, 2});
                    cmd.bindResourceSet(1, forwardSet);
                }
                boundPipeline = item.pipeline;
            }
            if (item.material != boundMaterial) {
                MaterialSystem::DrawState ds;
                if (!m_materials.drawState(MaterialHandle{item.material}, ObjectVariant::Dynamic, false, ds)) continue; // only the set is used
                cmd.bindResourceSet(2, ds.set);
                boundMaterial = item.material;
            }
            const uint32_t reflection = (item.flags >> DrawItem::kReflectionShift) & 0xFu;
            const bool skinned = (item.flags & DrawItem::kSkinned) != 0;
            if (skinned) {
                // Set 3 of skinned pipelines (M6 plan D9): the renderable's bone palette (a dynamic offset) and its
                // reflection cube. Bound for every skinned draw; the next rigid draw rebinds its own set 3.
                const uint32_t palette = scene.dynamics()[item.perDrawOrInstance].paletteOffset;
                const rhi::ResourceSetHandle set = m_environment.skinnedSet(reflection);
                if (palette == Scene::kNoPalette || !set) continue;
                cmd.bindResourceSet(3, set, {&palette, 1});
                boundReflection = ~0u;
            } else if (reflection != boundReflection) {
                const rhi::ResourceSetHandle set = m_environment.reflectionSet(reflection);
                if (!set) continue; // only after an out-of-memory failure in Environment::rebuildSets
                cmd.bindResourceSet(3, set);
                boundReflection = reflection;
            }
            if (item.mesh != boundMesh || (skinned && !skinBound)) {
                const rhi::BufferSlice streams[3] = {mesh->positions, mesh->surface, mesh->skin};
                cmd.bindVertexStreams(0, {streams, size_t(skinned ? 3 : 2)});
                cmd.bindIndexBuffer(mesh->indices, mesh->indexType);
                skinBound = skinned;
                boundMesh = item.mesh;
                boundModel = model;
                boundGpuMesh = mesh;
            }
            const GpuSubmesh& sub = model->submeshes[mesh->firstSubmesh + item.submesh];
            // The draw's LOD (M5 plan D6): its level's index range over the shared streams.
            const uint32_t level = std::min<uint32_t>(lodWordLevel(item.lod), mesh->lodCount - 1);
            const uint32_t indexCount = sub.lodCount[level], firstIndex = sub.lodFirst[level];
            uint32_t instances = 1;
            if (instanced) {
                instances = item.instanceCount;
                const rhi::BufferSlice data{vertexRing, item.perDrawOrInstance, instances * uint32_t(sizeof(InstanceData))};
                cmd.bindVertexStreams(2, {&data, 1});
                skinBound = false;
                cmd.drawIndexed(indexCount, instances, firstIndex);
            } else {
                const uint32_t index = item.perDrawOrInstance;
                const ProbeSample* probe = nullptr;
                if (item.flags & DrawItem::kStaticRow) {
                    perDraw.model = scene.staticRows()[index];
                    perDraw.lights = m_staticLightLists[index];
                    perDraw.lights.z |= scene.staticDrawFlags()[index];
                    // Statics without a lightmap take the hemispherical ambient (as SH), the same light as when they are
                    // instanced, so --instancing never changes the picture.
                    if (scene.staticGroups()[scene.staticRowGroups()[index]].lightmapped) perDraw.lightmapST = scene.staticLightmapST()[index];
                    else probe = &m_ambientProbe;
                } else {
                    perDraw.model = scene.dynamics()[index].rows;
                    probe = &m_dynamicProbes[index];
                    perDraw.lights = m_dynamicLightLists[index];
                    perDraw.lights.z |= packDrawFlags(0, 0, probe->reflectionSlot);
                }
                if (probe) {
                    perDraw.lightmapST = glm::vec4(probe->sunVisibility, 0.0f, 0.0f, 0.0f);
                    for (int c = 0; c < 3; ++c) perDraw.sh[c] = probe->sh[c];
                }
                perDraw.lights.w = item.lod; // level and cross-fade state (M5 plan D7)
                cmd.setPerDraw(&perDraw, uint8_t(sizeof(perDraw)));
                cmd.drawIndexed(indexCount, 1, firstIndex);
            }
            ++m_stats.sceneDraws;
            if (segment.pass == QueuePass::Transparent) ++m_stats.transparentDraws;
            if (segment.pass == QueuePass::Refraction) ++m_stats.refractionDraws;
            if (skinned) ++m_stats.skinnedDraws;
            m_stats.instances += instances;
            m_stats.triangles += uint64_t(indexCount / 3) * instances;
        }
        m_timers.endPass(cmd, timer);
    }
    if (m_sprites.drawCount()) { // after the transparent meshes: sprites sort among themselves (M6 plan D11)
        const uint32_t timer = m_timers.beginPass(cmd, "Sprites");
        m_sprites.record(cmd, m_frameSet, {frameOffsets, 2});
        m_timers.endPass(cmd, timer);
    }
    if (view == DebugView::Wireframe) { // every draw's triangles over the lit frame (M5 plan D10)
        const uint32_t timer = m_timers.beginPass(cmd, "Wireframe");
        m_debugDraw.record(cmd, DebugDraw::Mode::Wireframe, m_queue, scene, m_resources, m_materials, vertexRing, viewProj);
        m_timers.endPass(cmd, timer);
    }
}

} // namespace renderer
