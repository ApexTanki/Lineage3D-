#pragma once

#include "HdrCodec.h"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace tools {

// The procedural sky of a baked map (M3 plan D8): a zenith → horizon gradient, a soft glow around the sun, a cloud
// layer and a ground color below the horizon, in the engine's light unit (a uniform sky of radiance L lights a white
// Lambertian surface to L). The hard sun disc is not part of it: the sky shader draws the disc, and lighting gets the
// sun from its own direct term, so neither bakes nor reflections count the sun twice.
struct SkySettings {
    glm::vec3 zenith{0.16f, 0.30f, 0.62f};
    glm::vec3 horizon{0.60f, 0.70f, 0.80f};
    glm::vec3 ground{0.22f, 0.20f, 0.18f};
    float intensity = 1.0f;
    glm::vec3 sunDirection{0.0f, 1.0f, 0.0f}; // toward the sun
    glm::vec3 glow{1.0f, 0.82f, 0.55f};       // halo color × strength around the sun
    float glowExponent = 24.0f;
    float clouds = 0.45f;                     // coverage, 0–1
    float cloudScale = 1.6f;
    glm::vec3 cloudColor{1.0f, 0.98f, 0.94f};
    uint32_t seed = 7;
};

glm::vec3 skyRadiance(const SkySettings& sky, const glm::vec3& direction);

// Cube faces in the API order +X, −X, +Y, −Y, +Z, −Z with the Vulkan/GL face coordinate table; s, t in [0, 1] with t
// growing down the face (texel row 0 at t = 0), as both APIs sample it.
glm::vec3 cubeFaceDirection(uint32_t face, float s, float t);
void cubeFaceCoordinates(const glm::vec3& direction, uint32_t& face, float& s, float& t);

using CubeFaces = std::array<HdrImage, 6>;
using CubeMips = std::vector<CubeFaces>; // [mip][face]

CubeFaces renderSkyCube(const SkySettings& sky, uint32_t faceSize);

// Mip m of an environment cube is filtered for gloss g = 1 − m/(mips − 1) (what lit.frag reads at lod (1 − g)·maxMip):
// a Phong lobe of exponent n_Blinn/4, where n_Blinn = 2^(1 + 10g) is the specular exponent of that gloss. Mip 0 is the
// unfiltered cube. Filtered importance sampling: each sample reads a pre-downsampled level that matches its solid angle.
float environmentPhongExponent(uint32_t mip, uint32_t mips);
CubeMips prefilterCube(const CubeFaces& base, uint32_t mips);

// E/π of a surface facing `normal` under the sky (cosine-weighted integral), for the hemispherical fallback ambient.
glm::vec3 skyIrradiance(const SkySettings& sky, const glm::vec3& normal);

// Runs fn(i) for i in [0, count) on all hardware threads (offline tools only).
void parallelFor(uint32_t count, const std::function<void(uint32_t)>& fn);

} // namespace tools
