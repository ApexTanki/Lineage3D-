#include "renderer/visibility/Visibility.h"

#include "core/Profile.h"
#include "platform/JobSystem.h"
#include "platform/Timer.h"
#include "renderer/core/Scene.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace renderer {

struct Visibility::Context {
    Visibility* self = nullptr;
    const Scene* scene = nullptr;
    SphereArrays rows, dynamics;
    const Frustum* frusta[kMaxCullViews] = {};
    uint8_t frustumBits[kMaxCullViews] = {}; // the mask bit each frustum sets
    uint32_t frustumCount = 0;
    uint8_t rowBits = 0xFF; // views static rows may join (no shadow bits without static casters)
    uint8_t dynBits = 0xFF;
    uint32_t rowBlocks = 0; // index space of cullRange: row blocks, then dynamic blocks (4 objects each)
};

namespace {

template <typename T>
void growTo(std::vector<T>& v, size_t n) {
    if (v.size() < n) v.resize(n); // only when the scene grows (loading), never in a steady frame
}

} // namespace

void Visibility::init(platform::JobSystem& jobs) {
    m_jobs = &jobs;
    m_workerStats.resize(jobs.participantCount());
}

void Visibility::beginFrame(const glm::mat4& cameraViewProjection, const glm::vec3& cameraPosition, float tanHalfFovY,
                            const Settings& settings, float deltaSeconds) {
    m_settings = settings;
    ++m_frame;
    m_clock += std::clamp(deltaSeconds, 0.0f, 1.0f);
    if (!settings.freeze || !m_frozen) { // a frozen camera keeps the frame where freezing began
        m_cameraViewProjection = cameraViewProjection;
        m_cameraPosition = cameraPosition;
        m_tanHalfFovY = tanHalfFovY;
        m_camera = Frustum::fromViewProjection(cameraViewProjection);
    }
    m_frozen = settings.freeze;
    m_activeBits = 1;
}

void Visibility::setShadowView(uint32_t index, const glm::mat4& viewProjection, bool nearPlane) {
    if (index >= kMaxCullViews - 1) return;
    m_shadow[index] = Frustum::fromViewProjection(viewProjection, nearPlane, true);
    m_activeBits = uint8_t(m_activeBits | (1u << (index + 1)));
}

bool Visibility::cameraSees(const glm::vec4& sphere) const {
    return !m_settings.culling || m_camera.sphereVisible(glm::vec3(sphere), sphere.w);
}

void Visibility::rasterizeBands(void* context, uint32_t begin, uint32_t end, uint32_t) {
    Visibility& self = *static_cast<Visibility*>(context);
    for (uint32_t band = begin; band < end; ++band) self.m_occlusion.rasterizeBand(band);
}

void Visibility::run(const Scene& scene) {
    FL_ZONE("Visibility::run");
    const SphereArrays rows = scene.staticBounds();
    const auto dynamics = scene.dynamics();
    m_rowCount = rows.count;
    m_dynCount = uint32_t(dynamics.size());
    const size_t rowPadded = (size_t(m_rowCount) + 3) / 4 * 4, dynPadded = (size_t(m_dynCount) + 3) / 4 * 4;
    growTo(m_rowMask, rowPadded), growTo(m_rowDraw, rowPadded), growTo(m_rowShadowLevel, rowPadded), growTo(m_rowState, rowPadded);
    growTo(m_dynMask, dynPadded), growTo(m_dynDraw, dynPadded), growTo(m_dynShadowLevel, dynPadded), growTo(m_dynState, dynPadded);
    growTo(m_dynX, dynPadded), growTo(m_dynY, dynPadded), growTo(m_dynZ, dynPadded), growTo(m_dynR, dynPadded);
    growTo(m_dynBoxMin, dynPadded), growTo(m_dynBoxMax, dynPadded);
    // Re-indexed objects start their LOD over (they snap to their level on the next visible frame).
    if (scene.staticRevision() != m_staticRevision) {
        std::fill(m_rowState.begin(), m_rowState.end(), LodState{});
        m_staticRevision = scene.staticRevision();
    }
    if (scene.dynamicRevision() != m_dynamicRevision) {
        std::fill(m_dynState.begin(), m_dynState.end(), LodState{});
        m_dynamicRevision = scene.dynamicRevision();
    }
    for (size_t i = 0; i < dynPadded; ++i) {
        const bool real = i < m_dynCount;
        const glm::vec4 s = real ? dynamics[i].sphere : glm::vec4(0.0f, 0.0f, 0.0f, -FLT_MAX);
        m_dynX[i] = s.x, m_dynY[i] = s.y, m_dynZ[i] = s.z, m_dynR[i] = s.w;
        m_dynBoxMin[i] = real ? dynamics[i].boxMin : glm::vec3(0.0f);
        m_dynBoxMax[i] = real ? dynamics[i].boxMax : glm::vec3(0.0f);
    }

    // Occlusion (camera only): the frame's on-screen occluder boxes, rasterized in bands on the workers.
    uint64_t t0 = platform::nowTicks();
    m_occlusionActive = m_settings.culling && m_settings.occlusion && !scene.occluders().empty();
    if (m_occlusionActive) {
        m_occlusion.begin(m_cameraViewProjection, m_cameraPosition);
        for (const Scene::Occluder& o : scene.occluders())
            if (m_camera.sphereVisible(glm::vec3(o.sphere), o.sphere.w)) m_occlusion.addOccluder(o.corners);
        m_occlusion.prepare();
        m_jobs->parallelFor(OcclusionBuffer::kBands, 1, rasterizeBands, this);
        m_occlusion.finishStats();
        m_stats.occlusion = m_occlusion.stats();
    } else {
        m_stats.occlusion = OcclusionBuffer::Stats{};
    }
    uint64_t t1 = platform::nowTicks();
    m_stats.occlusionMs = platform::ticksToMilliseconds(t1 - t0);

    // Culling + LOD of every object.
    Context c;
    c.self = this;
    c.scene = &scene;
    c.rows = rows;
    c.dynamics.x = m_dynX.data(), c.dynamics.y = m_dynY.data(), c.dynamics.z = m_dynZ.data(), c.dynamics.r = m_dynR.data();
    c.dynamics.boxMin = m_dynBoxMin.data(), c.dynamics.boxMax = m_dynBoxMax.data();
    c.dynamics.count = m_dynCount;
    c.frusta[c.frustumCount] = &m_camera;
    c.frustumBits[c.frustumCount++] = 1;
    for (uint32_t v = 0; v < kMaxCullViews - 1; ++v)
        if (m_activeBits & (1u << (v + 1))) {
            c.frusta[c.frustumCount] = &m_shadow[v];
            c.frustumBits[c.frustumCount++] = uint8_t(1u << (v + 1));
        }
    c.dynBits = m_activeBits;
    c.rowBits = m_settings.staticCasters ? m_activeBits : uint8_t(1);
    c.rowBlocks = (m_rowCount + 3) / 4;
    for (WorkerStats& w : m_workerStats) w = WorkerStats{};
    m_jobs->parallelFor(c.rowBlocks + (m_dynCount + 3) / 4, 16, cullRange, &c);

    Stats& s = m_stats;
    s.objects = m_rowCount + m_dynCount;
    s.frustumCulled = s.occlusionTested = s.occluded = s.visible = s.fading = 0;
    std::fill(std::begin(s.viewObjects), std::end(s.viewObjects), 0u);
    std::fill(std::begin(s.lodObjects), std::end(s.lodObjects), 0u);
    for (const WorkerStats& w : m_workerStats) {
        s.frustumCulled += w.frustumCulled;
        s.occlusionTested += w.occlusionTested;
        s.occluded += w.occluded;
        s.visible += w.visible;
        s.fading += w.fading;
        for (uint32_t v = 0; v < kMaxCullViews; ++v) s.viewObjects[v] += w.viewObjects[v];
        for (uint32_t l = 0; l < kMaxLodLevels; ++l) s.lodObjects[l] += w.lodObjects[l];
    }
    s.cullMs = platform::ticksToMilliseconds(platform::nowTicks() - t1);
}

void Visibility::cullRange(void* context, uint32_t begin, uint32_t end, uint32_t worker) {
    const Context& c = *static_cast<const Context*>(context);
    Visibility& self = *c.self;
    const Settings& s = self.m_settings;
    WorkerStats& stats = self.m_workerStats[worker];
    const auto groups = c.scene->staticGroups();
    const auto rowGroups = c.scene->staticRowGroups();
    const auto dynamics = c.scene->dynamics();
    for (uint32_t block = begin; block < end; ++block) {
        const bool isRow = block < c.rowBlocks;
        const SphereArrays& bounds = isRow ? c.rows : c.dynamics;
        const uint32_t first = (isRow ? block : block - c.rowBlocks) * 4;
        uint8_t compact[4] = {0xFF, 0xFF, 0xFF, 0xFF};
        if (s.culling) cullSpheres4(c.frusta, c.frustumCount, bounds, first, compact);
        for (uint32_t k = 0; k < 4; ++k) {
            const uint32_t i = first + k;
            if (i >= bounds.count) break;
            uint8_t mask = 0xFF;
            if (s.culling) {
                mask = 0;
                for (uint32_t f = 0; f < c.frustumCount; ++f)
                    if (compact[k] & (1u << f)) mask = uint8_t(mask | c.frustumBits[f]);
            }
            mask = uint8_t(mask & (isRow ? c.rowBits : c.dynBits));
            if (mask & 1) {
                if (self.m_occlusionActive) {
                    ++stats.occlusionTested;
                    if (self.m_occlusion.boxOccluded(bounds.boxMin[i], bounds.boxMax[i])) {
                        mask = uint8_t(mask & ~1u);
                        ++stats.occluded;
                    }
                }
            } else if (s.culling) {
                ++stats.frustumCulled;
            }

            // LOD from the bounding sphere's screen size (M5 plan D6); shadow draws switch instantly.
            LodThresholds t;
            const float* sizes = nullptr;
            if (isRow) {
                const Scene::StaticGroup& g = groups[rowGroups[i]];
                t.count = g.lodCount;
                sizes = g.lodScreenSize;
            } else {
                t.count = dynamics[i].lodCount;
                sizes = dynamics[i].lodScreenSize;
            }
            std::copy(sizes, sizes + kMaxLodLevels, t.screenSize);
            const glm::vec3 center(bounds.x[i], bounds.y[i], bounds.z[i]);
            const float size = lodScreenSize(bounds.r[i], glm::length(center - self.m_cameraPosition), self.m_tanHalfFovY);
            // The level is tracked for objects that any view sees, so a caster keeps the level its camera draw uses (and its
            // hysteresis) when it leaves the screen, instead of popping to another level near a threshold.
            if (mask) {
                LodState& state = isRow ? self.m_rowState[i] : self.m_dynState[i];
                const LodDraw d = updateLod(state, size, t, s.lodBias, s.forceLod, s.lodFade, self.m_frame, self.m_clock);
                if (mask & 1) {
                    DrawLod& draw = isRow ? self.m_rowDraw[i] : self.m_dynDraw[i];
                    draw.level = uint8_t(d.level);
                    draw.previous = uint8_t(d.previous);
                    draw.fading = d.fading ? 1 : 0;
                    draw.progress = uint8_t(std::lround(std::clamp(d.progress, 0.0f, 1.0f) * 255.0f));
                    ++stats.visible;
                    ++stats.lodObjects[std::min<uint32_t>(d.level, kMaxLodLevels - 1)];
                    if (d.fading) ++stats.fading;
                }
                if (mask & ~1u) (isRow ? self.m_rowShadowLevel[i] : self.m_dynShadowLevel[i]) = uint8_t(d.level); // instant
            }
            (isRow ? self.m_rowMask[i] : self.m_dynMask[i]) = mask;
            for (uint32_t v = 0; v < kMaxCullViews; ++v) stats.viewObjects[v] += (mask >> v) & 1u;
        }
    }
}

} // namespace renderer
