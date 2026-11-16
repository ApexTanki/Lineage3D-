#include "renderer/core/Lights.h"

#include <cmath>

namespace renderer {
namespace {

struct Cone {
    float cosOuter, cosInner;
};

// Outer half-angle in [1°, 89°] and inner strictly inside it: smoothstep(cosOuter, cosInner, ·) needs cosOuter < cosInner.
Cone spotCone(const LightDesc& light) {
    const float outer = std::clamp(light.outerConeDeg, 1.0f, 89.0f);
    const float inner = std::clamp(light.innerConeDeg, 0.0f, outer);
    Cone c{std::cos(glm::radians(outer)), std::cos(glm::radians(inner))};
    c.cosInner = std::max(c.cosInner, c.cosOuter + 1e-4f);
    return c;
}

glm::vec3 spotDirection(const LightDesc& light) {
    const float len = glm::length(light.direction);
    return len > 1e-6f ? light.direction / len : glm::vec3(0.0f, -1.0f, 0.0f);
}

} // namespace

GpuLight toGpuLight(const LightDesc& light) {
    GpuLight g;
    g.positionInvRange = glm::vec4(light.position, 1.0f / std::max(light.range, 1e-3f));
    g.color = glm::vec4(light.color * light.intensity, 0.0f);
    if (light.type == LightType::Spot) {
        const Cone c = spotCone(light);
        g.spotDirection = glm::vec4(spotDirection(light), c.cosOuter);
        g.spotParams = glm::vec4(c.cosInner, -1.0f, 0.0f, 0.0f); // y = spot shadow slot (set per frame), −1 = none
    } else {
        // smoothstep(-2, -1, x) is 1 for every x in [-1, 1]: points need no branch in the shader.
        g.spotDirection = glm::vec4(0.0f, 0.0f, 0.0f, -2.0f);
        g.spotParams = glm::vec4(-1.0f, -1.0f, 0.0f, 0.0f);
    }
    return g;
}

FrameLight toFrameLight(const LightDesc& light) {
    FrameLight f;
    f.position = light.position;
    f.range = std::max(light.range, 1e-3f);
    const glm::vec3 c = light.color * light.intensity;
    f.weight = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
    f.spot = light.type == LightType::Spot;
    f.affectsStatic = light.affectsStatic;
    if (f.spot) {
        const Cone cone = spotCone(light);
        f.direction = spotDirection(light);
        f.cosOuter = cone.cosOuter;
        f.sinOuter = std::sqrt(std::max(0.0f, 1.0f - cone.cosOuter * cone.cosOuter));
    }
    return f;
}

glm::uvec4 packLightList(const uint8_t* indices, uint32_t count) {
    count = std::min(count, kMaxLightsPerObject);
    glm::uvec4 packed(0u);
    for (uint32_t i = 0; i < count; ++i) (i < 4 ? packed.x : packed.y) |= uint32_t(indices[i]) << ((i & 3u) * 8u);
    packed.z = count;
    return packed;
}

glm::uvec4 selectLights(std::span<const FrameLight> lights, const glm::vec4& sphere, bool isStatic, uint32_t maxLights) {
    maxLights = std::min(maxLights, kMaxLightsPerObject);
    uint8_t chosen[kMaxLightsPerObject];
    float scores[kMaxLightsPerObject];
    uint32_t count = 0;
    const glm::vec3 center(sphere);
    const float radius = sphere.w;
    const auto n = uint32_t(std::min<size_t>(lights.size(), kMaxFrameLights));
    for (uint32_t i = 0; i < n && maxLights > 0; ++i) {
        const FrameLight& l = lights[i];
        if (isStatic && !l.affectsStatic) continue;
        const glm::vec3 v = center - l.position;
        const float distSq = glm::dot(v, v);
        const float reach = l.range + radius;
        if (distSq >= reach * reach) continue;
        if (l.spot) {
            // Sphere against the spot's bounding cone ("Cull that cone", Wronski 2017).
            const float along = glm::dot(v, l.direction);
            const float across = std::sqrt(std::max(0.0f, distSq - along * along));
            if (l.cosOuter * across - along * l.sinOuter > radius || along > reach || along < -radius) continue;
        }
        const float score = l.weight * lightAttenuation(std::max(0.0f, std::sqrt(distSq) - radius), l.range);
        if (score <= 0.0f) continue;
        // Insert into the descending top-maxLights list.
        uint32_t pos = count < maxLights ? count : maxLights;
        while (pos > 0 && scores[pos - 1] < score) --pos;
        if (pos >= maxLights) continue;
        const uint32_t last = count < maxLights ? count : maxLights - 1;
        for (uint32_t k = last; k > pos; --k) {
            scores[k] = scores[k - 1];
            chosen[k] = chosen[k - 1];
        }
        scores[pos] = score;
        chosen[pos] = uint8_t(i);
        if (count < maxLights) ++count;
    }
    return packLightList(chosen, count);
}

} // namespace renderer
