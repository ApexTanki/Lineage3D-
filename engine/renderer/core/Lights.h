#pragma once

#include "renderer/core/ShaderTypes.h"

#include <algorithm>
#include <cstdint>
#include <span>

namespace renderer {

// Local lights (ARCHITECTURE.md §4.4–4.5; M2 plan D4, D5). The sun is not a local light: it lives in the
// environment (renderer::LightingData), because there is exactly one and it drives the shadow cascades later.
enum class LightType : uint8_t { Point, Spot };

struct LightDesc {
    LightType type = LightType::Point;
    glm::vec3 position{0.0f};
    glm::vec3 direction{0.0f, -1.0f, 0.0f}; // spot: the direction it points along; normalized by the renderer
    glm::vec3 color{1.0f};                  // linear
    float intensity = 1.0f;                 // π folded in: a white Lambertian surface facing the light reflects color × intensity
    float range = 10.0f;                    // meters; the light reaches exactly zero here
    float innerConeDeg = 20.0f;             // spot: full intensity inside this half-angle
    float outerConeDeg = 30.0f;             // spot: zero outside this half-angle (clamped to 1°–89°)
    bool affectsStatic = true;              // also lights static geometry; otherwise only dynamic objects
    bool castsShadows = false;              // spot: may get a shadow-atlas tile (High; M4 plan D8). Points: no (P2)
};

// A frame light prepared for per-object selection: the CPU side of one GpuLight.
struct FrameLight {
    glm::vec3 position{0.0f};
    float range = 1.0f;
    glm::vec3 direction{0.0f, -1.0f, 0.0f}; // spots
    float cosOuter = -2.0f;                 // spots: cos(outer half-angle)
    float sinOuter = 0.0f;
    float weight = 0.0f;                    // luminance of color × intensity: the ranking scale
    bool spot = false;
    bool affectsStatic = true;
};

// The range window of fl_lighting.glsl: 1 at the light, 0 at the range, zero slope at both ends.
inline float lightAttenuation(float distance, float range) {
    const float r = distance / std::max(range, 1e-6f);
    const float x = std::clamp(1.0f - r * r, 0.0f, 1.0f);
    return x * x;
}

GpuLight toGpuLight(const LightDesc& light);
FrameLight toFrameLight(const LightDesc& light);

// Per-object light lists (ARCHITECTURE.md §4.7): x, y = 8 × u8 indices into the frame's LightArray, z bits 0–3 =
// count (bits 4+ carry the lightmap page and flags from M3), w = LOD fade (M5).
glm::uvec4 packLightList(const uint8_t* indices, uint32_t count);
inline uint32_t lightListCount(const glm::uvec4& packed) { return std::min(packed.z & 0xFu, kMaxLightsPerObject); }
inline uint32_t lightListIndex(const glm::uvec4& packed, uint32_t i) {
    return ((i < 4 ? packed.x : packed.y) >> ((i & 3u) * 8u)) & 0xFFu;
}

// Picks up to `maxLights` (≤ 8) of the frame's lights that reach the bounding sphere (xyz center, w radius), ranked
// by weight × attenuation at the sphere's nearest point. Spots also need the sphere to touch their outer cone. Static
// objects only take lights with affectsStatic. `lights` holds at most kMaxFrameLights entries.
glm::uvec4 selectLights(std::span<const FrameLight> lights, const glm::vec4& sphere, bool isStatic, uint32_t maxLights);

} // namespace renderer
