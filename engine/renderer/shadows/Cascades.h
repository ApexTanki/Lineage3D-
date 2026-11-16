#pragma once

#include "core/Math.h"

#include <cstdint>
#include <span>

namespace renderer {

// The math of stable cascaded shadow maps and spot shadows (ARCHITECTURE.md §4.14; M4 plan D4, D8). Canonical matrices
// only (right-handed, looking down −Z, depth 0 at the near plane): ClipSpace::fix() turns them into backend clip space.
// Pure functions, unit-tested.

// Practical split scheme: the far end of cascade i blends the logarithmic and the linear split of [zNear, distance].
void cascadeSplits(float zNear, float shadowDistance, uint32_t count, float lambda, float* outFar);

// The minimal sphere around the slice [n, f] of a symmetric perspective frustum. Its center lies on the view axis at
// view depth `centerDepth`. It depends only on fov, aspect, n and f, never on the camera's orientation.
void frustumSliceSphere(float fovY, float aspect, float n, float f, float& centerDepth, float& radius);

// Light-space axes that depend only on the sun direction, so texel snapping happens on a grid fixed in the world.
// The light looks along −sunDirection.
void lightBasis(const glm::vec3& sunDirection, glm::vec3& right, glm::vec3& up);

struct CascadeSetup {
    glm::vec3 cameraPosition{0.0f};
    glm::vec3 cameraForward{0.0f, 0.0f, -1.0f}; // unit
    float fovY = 1.0f;
    float aspect = 1.0f;
    float zNear = 0.1f;
    float shadowDistance = 60.0f;
    glm::vec3 sunDirection{0.0f, 1.0f, 0.0f}; // unit, toward the sun
    uint32_t cascades = 2;
    uint32_t mapSize = 1024;
    float lambda = 0.7f;
    float blendBand = 0.1f;  // the last 10 % of each cascade is shared with the next one (dithered at the receiver)
    float extension = 25.0f; // the depth range reaches this far beyond the sphere toward the sun; casters farther
                             // away are flattened onto the near plane (pancaking)
    float edgeMargin = 6.0f; // texels kept free around the sphere: the receiver's normal offset (≤ 2.2 texels) plus the
                             // widest filter (2.5) and its bilinear footprint stay inside the map
};

struct CascadeFit {
    float splitNear = 0.0f; // view-depth range the cascade must cover (including the band shared with the previous one)
    float splitFar = 0.0f;  // receivers up to this view depth use it
    float band = 0.0f;      // the last `band` meters before splitFar dither into the next cascade (or fade out)
    glm::vec3 center{0.0f}; // texel-snapped sphere center, world space
    float radius = 0.0f;
    float depthRange = 0.0f; // light-view depth from the near plane to the far side of the sphere
    float texelSize = 0.0f;  // world size of one shadow-map texel: 2r / mapSize
    glm::mat4 view{1.0f};       // world → light view
    glm::mat4 projection{1.0f}; // canonical ortho over [−r, r]² × [0, depthRange]
};

// Fits up to out.size() cascades; returns the count (setup.cascades clamped to out.size()).
uint32_t fitCascades(const CascadeSetup& setup, std::span<CascadeFit> out);

struct SpotShadowFit {
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f}; // canonical perspective, fov = 2 × outer cone, aspect 1
    float nearPlane = 0.05f;
    float farPlane = 1.0f;
    float texelAngle = 0.0f; // 2 tan(fov / 2) / tileSize: world texel size per meter of distance from the light
};

SpotShadowFit fitSpotShadow(const glm::vec3& position, const glm::vec3& direction, float outerConeDeg, float range,
                            uint32_t tileSize);

// Spot shadow ranking (M4 plan D8): brightness × the light's projected size (range over distance to the camera).
float spotShadowScore(float luminanceIntensity, float range, const glm::vec3& lightPosition, const glm::vec3& cameraPosition);

} // namespace renderer
