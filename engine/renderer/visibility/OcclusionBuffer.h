#pragma once

#include "core/Math.h"

#include <cstdint>
#include <vector>

namespace renderer {

// Software occlusion culling (ARCHITECTURE.md §4.15; M5 plan D4). Occluders are boxes (a solid static mesh's bounding
// box, given as its 8 world corners); they are rasterized on the CPU into a small canonical-depth buffer, and objects
// whose screen rectangle lies wholly behind it are culled. Everything is conservative:
//   - a pixel counts as covered only when one box covers all of it (inner coverage of the box's convex screen hull,
//     which is clipped at the near plane), so cracks between occluders, edges and sub-pixel gaps never cull;
//   - it stores the farthest depth the box reaches inside the pixel: the max over the box's front-face planes (the
//     entry depth of a convex body) plus half a pixel of each plane's slope;
//   - a tile of 8×8 pixels keeps its farthest pixel (far where any pixel is uncovered), and an object is occluded only
//     when every tile under its screen rectangle is nearer than the object's nearest point.
// Canonical clip space (z in [0, 1]) on the CPU, so both backends cull identically. Usage per frame: begin(), add
// occluders, prepare(), rasterizeBand() for every band (jobs, any order), then boxOccluded() from any thread.
class OcclusionBuffer {
public:
    static constexpr uint32_t kWidth = 256;
    static constexpr uint32_t kHeight = 128;
    static constexpr uint32_t kTile = 8;
    static constexpr uint32_t kTilesX = kWidth / kTile;
    static constexpr uint32_t kTilesY = kHeight / kTile;
    static constexpr uint32_t kBands = kTilesY; // one tile row of pixels per band
    static constexpr uint32_t kMaxOccluders = 512;
    static constexpr float kMinArea = 16.0f; // occluders whose hull covers fewer pixels aren't worth drawing

    struct Stats {
        uint32_t submitted = 0;   // occluders offered this frame
        uint32_t offscreen = 0;   // outside the view, behind the camera or around the eye
        uint32_t tooSmall = 0;    // hull under kMinArea
        uint32_t overBudget = 0;  // beyond the kMaxOccluders largest
        uint32_t drawn = 0;
        uint32_t coveredTiles = 0; // tiles with every pixel covered (after rasterizeBand of every band)
    };

    OcclusionBuffer();

    // `viewProjection`: canonical perspective (ClipSpace::canonicalPerspective × view); `eye`: the camera position. Its depth
    // mapping (z = A − B / view distance) comes from the matrix itself, so it always matches the projection.
    void begin(const glm::mat4& viewProjection, const glm::vec3& eye);
    // An object counts as hidden only when it lies at least this far behind the occluders (meters + fraction of its
    // distance), so float rounding in the plane depths can never cull a decal or sign that sits on an occluder face.
    static constexpr float kMarginMeters = 0.02f;
    static constexpr float kMarginRelative = 0.002f;
    // Corner k = (k & 1 ? max.x : min.x, k & 2 ? max.y : min.y, k & 4 ? max.z : min.z) of the box, transformed to world.
    void addOccluder(const glm::vec3 corners[8]);
    void prepare(); // keeps the kMaxOccluders largest; then every band can be rasterized
    void rasterizeBand(uint32_t band);
    // After every band: true when the world box is hidden behind the occluders. A box reaching behind the near plane is
    // never occluded.
    bool boxOccluded(const glm::vec3& boxMin, const glm::vec3& boxMax) const;

    const Stats& stats() const { return m_stats; }
    void finishStats(); // counts covered tiles (after every band)
    float tileDepth(uint32_t tx, uint32_t ty) const { return m_tileMax[ty * kTilesX + tx]; }
    float pixelDepth(uint32_t x, uint32_t y) const { return m_depth[y * kWidth + x]; } // debug display and tests

private:
    struct Prepared {
        float edgeA[16], edgeB[16], edgeK[16]; // inner-coverage edge functions: a·X + b·Y + k ≥ 0 at pixel centers
        uint32_t edgeCount = 0;
        float planeA[6], planeB[6], planeK[6];  // per front face: depth = a·X + b·Y + k (pixel center, + half-pixel slope)
        uint32_t planeCount = 0;
        float depthCap = 1.0f;                  // the farthest point of the (near-clipped) box
        int x0 = 0, x1 = -1, y0 = 0, y1 = -1;   // pixel bounds of the hull
        float area = 0.0f;
    };

    bool setupOccluder(const glm::vec3 corners[8], Prepared& out);

    glm::mat4 m_viewProjection{1.0f};
    glm::dmat4 m_toClipPlanes{1.0}; // inverse-transpose of the view-projection, in double: world planes → clip planes
    glm::vec3 m_eye{0.0f};
    double m_zNear = 0.1;
    double m_depthA = 1.0, m_depthB = 0.1; // canonical depth = A − B / view distance (A = 0: not a perspective, no culling)
    std::vector<Prepared> m_occluders; // reserved once; prepare() keeps the largest kMaxOccluders
    std::vector<float> m_depth;        // kWidth × kHeight
    std::vector<float> m_tileMax;      // kTilesX × kTilesY
    Stats m_stats;
};

} // namespace renderer
