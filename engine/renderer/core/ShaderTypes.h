#pragma once

#include "core/Math.h"

#include <cstdint>

namespace renderer {

// C++ mirrors of the std140 blocks in shaders/include (vec4-only members, so both layouts agree; P5).

inline constexpr uint32_t kReflectionSlots = 8; // FL_REFLECTION_SLOTS: slot 0 = the sky, 1–7 = local probes (M3 plan D8)

// fl_bindings.glsl, set 0 binding 0: per-frame constants.
struct FrameUniforms {
    glm::mat4 viewProj{1.0f};       // backend-ready, built by ClipSpace
    glm::mat4 skyFromClip{1.0f};    // sky pass: backend clip position on the far plane → world direction
    glm::vec4 time{0.0f};           // x = seconds since start, y = frame delta
    glm::vec4 outputParams{0.0f};   // x = 1: the shader encodes sRGB; y = 1: alpha to coverage; z = DebugView; w = 1: RGBM
    glm::vec4 cameraPosition{0.0f};
    glm::vec4 sunDirection{0.0f, 1.0f, 0.0f, 0.0f}; // xyz = unit vector toward the sun; w = cos(sun disc radius)
    glm::vec4 sunColor{0.0f};       // rgb = color × intensity; w = sun disc radiance multiplier
    glm::vec4 ambientSky{0.0f};     // hemispherical ambient (ARCHITECTURE.md §4.4); w = luminance of the sky's E/π
    glm::vec4 ambientGround{0.0f};  // w = sky cube intensity
    glm::vec4 fogColor{0.0f};       // rgb; w = mode (FogSettings::Mode)
    glm::vec4 fogParams{0.0f};      // start, end, density, max opacity
    glm::vec4 fogHeight{0.0f};      // base height, density, falloff, enabled
    glm::vec4 fogSky{0.0f};         // x = horizon band
    glm::vec4 environment{0.0f};    // x = max mip of the sky cube, y = 1: a sky is loaded
    glm::vec4 reflectionProbes[3 * kReflectionSlots]{}; // per slot: boxMin (w = box projection), boxMax (w = max mip), position
    // M4 plan D4, D5, D8 (fl_shadows.glsl)
    glm::mat4 cascadeMatrix[3]{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)}; // world → (u, v, depth)
    glm::vec4 cascadeSplits{0.0f}; // xyz = view depth where cascade i ends; w = cascade count
    glm::vec4 cascadeBands{0.0f};  // xyz = dither / fade band width (m); w = ShadowFilterMode
    glm::vec4 cascadeTexel{0.0f};  // xyz = world texel size; w = map size
    glm::vec4 cascadeFilter{0.0f}; // xyz = Poisson radius in UV; w = 1: static casters in the cascades
    glm::vec4 cameraForward{0.0f, 0.0f, -1.0f, 0.0f};
    glm::mat4 spotMatrix[4]{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)}; // world → atlas × w
    glm::vec4 spotTile[4]{};       // per slot: UV min.xy, max.zw
    glm::vec4 spotTexel{0.0f};     // per slot: world texel size per meter from the light
    glm::vec4 cameraUp{0.0f, 1.0f, 0.0f, 0.0f}; // xyz = the view's up axis in world space (billboards, M7)
    // Post-processing (M8 plan D13).
    glm::vec4 screen{0.0f};     // xy = scene target size in pixels, zw = 1 / size
    glm::vec4 post{1.0f, 0.0f, 0.0f, 0.0f}; // x = exposure; y = 1: curve in the forward shaders (LDR); z = tonemapper; w = 1: SSAO
    glm::vec4 postParams{0.0f}; // x = 1: refraction grab + its linear depth are this frame's; y = 1: SSR
};
static_assert(sizeof(FrameUniforms) == 1376, "FrameUniforms must match fl_bindings.glsl");

// cascadeBands.w: how receivers filter the cascades (M4 plan D2, D5).
enum class ShadowFilterMode : uint32_t { Off = 0, HardwareTap = 1, Poisson16 = 2, FourNearest = 3 };

// shadow.vert, set 1 binding 0: one shadow pass (M4 plan D7).
struct ShadowPassUniforms {
    glm::mat4 viewProj{1.0f}; // backend clip space
    glm::vec4 params{0.0f};   // x = 1 pancake, y = NDC depth offset, z = time, w = NDC z of the near plane
};
static_assert(sizeof(ShadowPassUniforms) == 80, "ShadowPassUniforms must match shadow.vert");

// debug.vert / debug.frag, set 1 binding 0 (M5 plan D10): the wireframe and overdraw passes.
struct DebugPassUniforms {
    glm::mat4 viewProj{1.0f}; // backend clip space: the scene camera's
    glm::vec4 color{1.0f};    // linear
};
static_assert(sizeof(DebugPassUniforms) == 80, "DebugPassUniforms must match debug.vert");

// blob.vert, per instance (locations 8, 9): one blob shadow (M4 plan D9).
struct BlobInstance {
    glm::vec4 centerRadius{0.0f}; // xyz = center on the ground, w = radius along its X axis
    glm::vec4 params{0.0f};       // x = darkening at the center, y = radius along its Z axis, zw = X axis (unit, xz)
};
static_assert(sizeof(BlobInstance) == 32, "BlobInstance must match blob.vert");

// One sprite (M6 plan D11), instance attributes 8–11 of sprite.vert.
struct SpriteInstance {
    glm::vec4 centerRotation{0.0f}; // xyz = center (world), w = rotation (radians, camera-facing sprites)
    glm::vec4 sizeFacing{0.0f};     // xy = size (m), z = 1: upright about Y, w = blend for fog (0 alpha, 1 additive, 2 premultiplied)
    glm::vec4 uvRect{0.0f};         // the flipbook cell: UV offset xy, scale zw
    glm::vec4 color{1.0f};          // linear RGB × opacity
};
static_assert(sizeof(SpriteInstance) == 64, "SpriteInstance must match sprite.vert");

// fl_bindings.glsl: per-draw constants, 128 bytes (ARCHITECTURE.md §4.7).
struct PerDraw {
    core::Affine3x4 model;      // object → world, with position dequantization folded in
    glm::vec4 lightmapST{0.0f}; // lightmapped: UV1 scale.xy, offset.zw; dynamic: x = sun visibility from the probes
    glm::uvec4 lights{0u};      // packed light list | lightmap page | reflection slot (packDrawFlags)
    glm::vec4 sh[3]{};          // dynamic: L1 SH irradiance (E/π = sh.w + dot(sh.xyz, n)) per channel R, G, B
};
static_assert(sizeof(PerDraw) == 128, "PerDraw must be 128 bytes");

// lights.z beyond the light count (bits 0–3): lightmap page in bits 4–11, reflection slot in bits 12–15 (M3 plan D7).
inline uint32_t packDrawFlags(uint32_t lightCountBits, uint32_t lightmapPage, uint32_t reflectionSlot) {
    return (lightCountBits & 0xFu) | (lightmapPage & 0xFFu) << 4 | (reflectionSlot & 0xFu) << 12;
}

// fl_material.glsl, set 2 binding 0 (ARCHITECTURE.md §5.4). Colors are linear.
struct MaterialUniforms {
    glm::vec4 baseColor{1.0f};     // rgb tint, a opacity
    glm::vec4 specularGloss{0.0f}; // rgb specular color, a gloss
    glm::vec4 emission{0.0f};      // rgb color, a intensity
    glm::vec4 uvAnim{0.0f};        // xy = scroll (units/s), z = rotation (rad/s); w = alpha cutoff
    glm::vec4 flipbook{1.0f, 1.0f, 0.0f, 0.0f}; // columns, rows, frames per second
    glm::vec4 detail{0.0f};        // xy = detail scroll, z = detail tiling, w = detail strength
    glm::vec4 misc{0.0f};          // x = reflection strength, y = 1 additive, z = normal strength, w = 1 premultiplied
    glm::vec4 refraction{1.0f, 1.0f, 1.0f, 0.0f}; // rgb tint, a strength in m (0: not refractive; M8 plan D10)
};
static_assert(sizeof(MaterialUniforms) == 128, "MaterialUniforms must match fl_material.glsl");

// fl_lighting.glsl, set 0 binding 1: one local light.
struct GpuLight {
    glm::vec4 positionInvRange{0.0f}; // xyz world position, w = 1 / range
    glm::vec4 color{0.0f};            // rgb = color × intensity (linear)
    glm::vec4 spotDirection{0.0f};    // xyz unit direction the spot points along, w = cos(outer); points: -2
    glm::vec4 spotParams{0.0f};       // x = cos(inner) > cos(outer); points: -1
};
static_assert(sizeof(GpuLight) == 64, "GpuLight must match fl_lighting.glsl");

inline constexpr uint32_t kMaxFrameLights = 256;     // FL_MAX_LIGHTS; light indices are 8 bits
inline constexpr uint32_t kMaxLightsPerObject = 8;   // FL_MAX_LIGHTS_PER_OBJECT
inline constexpr uint32_t kLightArrayBytes = kMaxFrameLights * uint32_t(sizeof(GpuLight)); // 16 KB, the GL 3.3 minimum

// Static rows as the scene stores them (object → world, dequantization folded in).
using InstanceRows = core::Affine3x4;
static_assert(sizeof(InstanceRows) == 48, "instance rows are 48 bytes");

// Per-instance vertex data of instanced static draws (locations 8–12, ARCHITECTURE.md §4.7), 80 bytes.
struct InstanceData {
    InstanceRows rows;          // locations 8–10
    glm::vec4 lightmapST{0.0f}; // location 11: UV1 → lightmap page (lightmapped groups)
    glm::uvec4 lights{0u};      // location 12: packed light list | page | reflection slot
};
static_assert(sizeof(InstanceData) == 80, "instance data is 80 bytes");

// Bone palettes (M6 plan D8, D9): up to 128 joints × three vec4 rows, read by skinned pipelines at set 3, binding 0 with a
// dynamic offset into the frame's uniform ring. Always allocated whole, so the bound range never leaves the ring.
inline constexpr uint32_t kMaxPaletteJoints = 128;
inline constexpr uint32_t kPaletteBytes = kMaxPaletteJoints * 48;

// Material debug views (lit.frag), selected per frame.
enum class DebugView : uint8_t {
    Lit, Albedo, Lighting, Normal, Specular, Gloss, Emission, LightCount,
    Lightmap,      // lightmapped statics: the lightmap alone (E/π without the sun)
    Probes,        // dynamic objects: the probe SH irradiance
    SunVisibility, // shadowmask, or the probes' sun visibility
    Reflection,    // the environment reflection term
    Cascades,      // the lit frame tinted by the shadow cascade that each pixel samples (M4 plan D12)
    Shadow,        // the combined sun visibility: baked × cascades (M4 plan D6)
    Uv,            // UV0 (fract) (M5 plan D10)
    Tangent,       // the world tangent of normal-mapped materials (others gray)
    LodLevel,      // the lit frame tinted by LOD level: 0 green, 1 yellow, 2 orange, 3 red
    Wireframe,     // the lit frame with every draw's triangles over it (a separate pass; lit.frag shades it as Lit)
    Overdraw,      // how many draws cover each pixel, as a heat ramp (a separate pass and the present pass)
    Bones,         // skinned draws: each vertex's joints as hashed colors blended by their weights; others gray (M6 plan D12)
    Ssao,          // the ambient occlusion the lit shader applies (white without SSAO; M8 plan D9)
    Count
};

} // namespace renderer
