#include "renderer/shadows/Cascades.h"

#include "renderer/core/ClipSpace.h"
#include "renderer/shadows/ShadowSettings.h"

#include <algorithm>
#include <cmath>

namespace renderer {
namespace {

// World → view for a right-handed camera at `eye` with the given orthonormal axes, looking along `forward`.
glm::mat4 viewFromBasis(const glm::vec3& eye, const glm::vec3& right, const glm::vec3& up, const glm::vec3& forward) {
    glm::mat4 v(1.0f);
    v[0][0] = right.x;
    v[1][0] = right.y;
    v[2][0] = right.z;
    v[0][1] = up.x;
    v[1][1] = up.y;
    v[2][1] = up.z;
    v[0][2] = -forward.x;
    v[1][2] = -forward.y;
    v[2][2] = -forward.z;
    v[3][0] = -glm::dot(right, eye);
    v[3][1] = -glm::dot(up, eye);
    v[3][2] = glm::dot(forward, eye);
    return v;
}

} // namespace

void cascadeSplits(float zNear, float shadowDistance, uint32_t count, float lambda, float* outFar) {
    zNear = std::max(zNear, 1e-3f);
    shadowDistance = std::max(shadowDistance, zNear * 1.01f);
    for (uint32_t i = 1; i <= count; ++i) {
        const float t = float(i) / float(count);
        const float logSplit = zNear * std::pow(shadowDistance / zNear, t);
        const float linearSplit = zNear + (shadowDistance - zNear) * t;
        outFar[i - 1] = lambda * logSplit + (1.0f - lambda) * linearSplit;
    }
    if (count) outFar[count - 1] = shadowDistance; // exactly, whatever the rounding
}

void frustumSliceSphere(float fovY, float aspect, float n, float f, float& centerDepth, float& radius) {
    // Corners lie at lateral distance depth × k, k = tan of the half-diagonal angle. The center that is equally far
    // from the near and the far corners sits at (n + f)(1 + k²) / 2; past the far plane, the far corners alone decide.
    const float t = std::tan(fovY * 0.5f);
    const float k2 = t * t * (1.0f + aspect * aspect);
    const float c = 0.5f * (n + f) * (1.0f + k2);
    if (c >= f) {
        centerDepth = f;
        radius = f * std::sqrt(k2);
    } else {
        centerDepth = c;
        radius = std::sqrt((f - c) * (f - c) + f * f * k2);
    }
}

void lightBasis(const glm::vec3& sunDirection, glm::vec3& right, glm::vec3& up) {
    const glm::vec3 forward = -sunDirection;
    const glm::vec3 hint = std::abs(sunDirection.y) > 0.99f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    right = glm::normalize(glm::cross(forward, hint));
    up = glm::cross(right, forward);
}

uint32_t fitCascades(const CascadeSetup& s, std::span<CascadeFit> out) {
    const uint32_t count = std::min<uint32_t>(std::min<uint32_t>(s.cascades, uint32_t(out.size())), 8u);
    if (count == 0 || s.mapSize == 0) return 0;
    float splitFar[8];
    cascadeSplits(s.zNear, s.shadowDistance, count, s.lambda, splitFar);
    glm::vec3 right, up;
    lightBasis(s.sunDirection, right, up);
    const glm::vec3 forward = -s.sunDirection;
    for (uint32_t i = 0; i < count; ++i) {
        CascadeFit& c = out[i];
        const float start = i == 0 ? s.zNear : splitFar[i - 1];
        c.splitFar = splitFar[i];
        c.band = s.blendBand * (c.splitFar - start);
        // Receivers in the previous cascade's band may pick this one, so it covers that band too.
        c.splitNear = i == 0 ? s.zNear : start - out[i - 1].band;
        float centerDepth = 0.0f;
        frustumSliceSphere(s.fovY, s.aspect, c.splitNear, c.splitFar, centerDepth, c.radius);
        // Grow the square so `edgeMargin` texels of it lie outside the sphere: r' = r + margin × 2r' / size.
        c.radius /= std::max(1.0f - 2.0f * s.edgeMargin / float(s.mapSize), 0.5f);
        c.texelSize = 2.0f * c.radius / float(s.mapSize);
        // Snap the center on the light-space texel grid (fixed in the world: the basis depends only on the sun), so a
        // moving camera shifts the map by whole texels and static shadows don't crawl.
        const glm::vec3 center = s.cameraPosition + s.cameraForward * centerDepth;
        auto snap = [&](float v) { return std::floor(v / c.texelSize + 0.5f) * c.texelSize; };
        c.center = right * snap(glm::dot(center, right)) + up * snap(glm::dot(center, up)) + forward * snap(glm::dot(center, forward));
        const float back = c.radius + s.extension;
        c.depthRange = back + c.radius;
        c.view = viewFromBasis(c.center - forward * back, right, up, forward);
        c.projection = ClipSpace::canonicalOrthographic(-c.radius, c.radius, -c.radius, c.radius, 0.0f, c.depthRange);
    }
    return count;
}

SpotShadowFit fitSpotShadow(const glm::vec3& position, const glm::vec3& direction, float outerConeDeg, float range,
                            uint32_t tileSize) {
    SpotShadowFit f;
    const glm::vec3 forward = glm::normalize(direction);
    const glm::vec3 hint = std::abs(forward.y) > 0.99f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 right = glm::normalize(glm::cross(forward, hint));
    f.view = viewFromBasis(position, right, glm::cross(right, forward), forward);
    // A square frustum whose half-angle is the outer cone contains the whole cone.
    const float fov = 2.0f * glm::radians(std::clamp(outerConeDeg, 1.0f, kMaxSpotShadowConeDeg));
    f.nearPlane = std::max(0.05f, range * 0.01f);
    f.farPlane = std::max(range, f.nearPlane * 2.0f);
    f.projection = ClipSpace::canonicalPerspective(fov, 1.0f, f.nearPlane, f.farPlane);
    f.texelAngle = 2.0f * std::tan(fov * 0.5f) / float(std::max(tileSize, 1u));
    return f;
}

float spotShadowScore(float luminanceIntensity, float range, const glm::vec3& lightPosition, const glm::vec3& cameraPosition) {
    return luminanceIntensity * range / std::max(glm::length(lightPosition - cameraPosition), 1.0f);
}

} // namespace renderer
