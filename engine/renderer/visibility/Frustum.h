#pragma once

#include "core/Math.h"

#include <cstdint>

namespace renderer {

// A convex view volume of up to 6 planes, n·p + d ≥ 0 inside, n unit length (ARCHITECTURE.md §4.15; M5 plan D1, D3).
struct Frustum {
    glm::vec4 planes[6]{};
    uint32_t count = 0;

    // From a canonical view-projection (ClipSpace: right-handed, x and y in [−1, 1], z in [0, 1]): left, right, bottom,
    // top, then near and far when asked. Cascades drop the near plane: casters toward the sun are pancaked, not culled.
    static Frustum fromViewProjection(const glm::mat4& viewProjection, bool nearPlane = true, bool farPlane = true);

    // Scalar reference tests: false = entirely outside one plane (conservative: true near the corners).
    bool sphereVisible(const glm::vec3& center, float radius) const;
    bool boxVisible(const glm::vec3& boxMin, const glm::vec3& boxMax) const; // the box's farthest corner along each normal
};

// Bounding spheres as structure-of-arrays, padded to a multiple of 4 with entries that are never visible (radius
// −FLT_MAX), so 4-wide tests read whole blocks. Optional world boxes refine spheres that straddle a plane.
struct SphereArrays {
    const float* x = nullptr;
    const float* y = nullptr;
    const float* z = nullptr;
    const float* r = nullptr;
    const glm::vec3* boxMin = nullptr; // per entry, or null: spheres only
    const glm::vec3* boxMax = nullptr;
    uint32_t count = 0;                // real entries (the arrays hold count rounded up to 4)
};

inline constexpr uint32_t kMaxCullViews = 8; // one bit each in a visibility byte

// Tests the 4 spheres starting at `first` (a multiple of 4) against up to 8 frusta: out[k] gets bit v set when sphere
// first + k may be visible in frusta[v]. SSE2: one pass of 4 spheres per plane; spheres that straddle a plane of a view
// they are not outside of get the box test when boxes are given. Entries at or past `count` get 0.
void cullSpheres4(const Frustum* const* frusta, uint32_t viewCount, const SphereArrays& spheres, uint32_t first, uint8_t out[4]);

// The same answer one sphere at a time (unit-test reference and fallback).
uint8_t cullSphereScalar(const Frustum* const* frusta, uint32_t viewCount, const SphereArrays& spheres, uint32_t index);

} // namespace renderer
