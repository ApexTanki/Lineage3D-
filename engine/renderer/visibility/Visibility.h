#pragma once

#include "renderer/visibility/Frustum.h"
#include "renderer/visibility/Lod.h"
#include "renderer/visibility/OcclusionBuffer.h"

#include <span>
#include <vector>

namespace platform {
class JobSystem;
}

namespace renderer {

class Scene;

// Per-frame visibility (ARCHITECTURE.md §4.15; M5 plan D1–D4, D6, D11): every static row and dynamic renderable gets a
// view mask (bit 0 camera, bits 1–3 cascades, bits 4–7 spot shadows), its camera LOD (with the cross-fade) and the
// level its shadow draws use. Occluders are rasterized and objects culled on the job workers; arrays grow with the
// scene only. Order per frame: beginFrame() (camera, clock), setShadowView() for each shadow view, run().
class Visibility {
public:
    struct Settings {
        bool culling = true;       // false: everything is in every view (the M4 behavior; measurements, reference images)
        bool occlusion = true;     // software occlusion for the camera (needs culling)
        bool staticCasters = true; // static rows get shadow bits (High); off: camera bit only
        int forceLod = -1;         // −1 automatic; 0–3 a fixed level (clamped per mesh)
        float lodBias = 1.0f;      // > 1 picks coarser levels sooner
        bool lodFade = true;       // dithered cross-fade between levels
        bool freeze = false;       // culling, occlusion and LOD stay at the camera of the frame it was turned on
    };
    struct DrawLod { // the camera's level of a visible object
        uint8_t level = 0;
        uint8_t previous = 0; // outgoing level while fading
        uint8_t fading = 0;
        uint8_t progress = 255; // fade progress 0–255
    };
    struct Stats {
        uint32_t objects = 0;         // static rows + dynamic renderables
        uint32_t frustumCulled = 0;   // outside the camera frustum
        uint32_t occlusionTested = 0; // in the frustum, tested against the occlusion buffer
        uint32_t occluded = 0;
        uint32_t visible = 0;         // to the camera
        uint32_t viewObjects[kMaxCullViews] = {}; // objects with each mask bit
        uint32_t lodObjects[kMaxLodLevels] = {};  // camera-visible objects per level
        uint32_t fading = 0;
        OcclusionBuffer::Stats occlusion;
        double occlusionMs = 0.0; // occluder setup + rasterization
        double cullMs = 0.0;      // culling + LOD of every object
    };

    void init(platform::JobSystem& jobs);

    // The camera (canonical view-projection) unless frozen; advances the fade clock by the frame time. Clears the
    // shadow views.
    void beginFrame(const glm::mat4& cameraViewProjection, const glm::vec3& cameraPosition, float tanHalfFovY,
                    const Settings& settings, float deltaSeconds);
    // Shadow view `index` (0–2 cascades, 3–6 spot tiles) from its canonical view-projection; cascades have no near plane.
    void setShadowView(uint32_t index, const glm::mat4& viewProjection, bool nearPlane);
    // A sphere (center, radius) the camera may see: lights and blob shadows. Always true with culling off.
    bool cameraSees(const glm::vec4& sphere) const;
    // Occlusion, then culling and LOD of the scene's objects (main thread; fans out to the job workers).
    void run(const Scene& scene);

    std::span<const uint8_t> rowMasks() const { return {m_rowMask.data(), m_rowCount}; }
    std::span<const uint8_t> dynamicMasks() const { return {m_dynMask.data(), m_dynCount}; }
    std::span<const DrawLod> rowLods() const { return {m_rowDraw.data(), m_rowCount}; }       // where bit 0 is set
    std::span<const DrawLod> dynamicLods() const { return {m_dynDraw.data(), m_dynCount}; }
    std::span<const uint8_t> rowShadowLevels() const { return {m_rowShadowLevel.data(), m_rowCount}; } // where a shadow bit is set
    std::span<const uint8_t> dynamicShadowLevels() const { return {m_dynShadowLevel.data(), m_dynCount}; }
    uint8_t activeViewBits() const { return m_activeBits; }
    const Stats& stats() const { return m_stats; }
    const OcclusionBuffer& occlusion() const { return m_occlusion; }
    bool frozen() const { return m_frozen; }
    const glm::vec3& cullingCameraPosition() const { return m_cameraPosition; }

private:
    struct alignas(64) WorkerStats {
        uint32_t frustumCulled = 0, occlusionTested = 0, occluded = 0, visible = 0, fading = 0;
        uint32_t viewObjects[kMaxCullViews] = {};
        uint32_t lodObjects[kMaxLodLevels] = {};
    };
    struct Context;
    static void rasterizeBands(void* context, uint32_t begin, uint32_t end, uint32_t worker);
    static void cullRange(void* context, uint32_t begin, uint32_t end, uint32_t worker);

    platform::JobSystem* m_jobs = nullptr;
    Settings m_settings;
    Frustum m_camera;
    glm::mat4 m_cameraViewProjection{1.0f};
    glm::vec3 m_cameraPosition{0.0f};
    float m_tanHalfFovY = 0.577f;
    bool m_frozen = false;
    Frustum m_shadow[kMaxCullViews - 1];
    uint8_t m_activeBits = 1; // bit 0 (camera) plus each active shadow view
    uint32_t m_frame = 0;
    float m_clock = 0.0f;
    uint32_t m_staticRevision = ~0u, m_dynamicRevision = ~0u;
    OcclusionBuffer m_occlusion;
    bool m_occlusionActive = false;

    uint32_t m_rowCount = 0, m_dynCount = 0;
    std::vector<uint8_t> m_rowMask, m_dynMask;
    std::vector<DrawLod> m_rowDraw, m_dynDraw;
    std::vector<uint8_t> m_rowShadowLevel, m_dynShadowLevel;
    std::vector<LodState> m_rowState, m_dynState;
    std::vector<float> m_dynX, m_dynY, m_dynZ, m_dynR; // dynamic bounds as SoA, refreshed every frame
    std::vector<glm::vec3> m_dynBoxMin, m_dynBoxMax;
    std::vector<WorkerStats> m_workerStats;
    Stats m_stats;
};

} // namespace renderer
