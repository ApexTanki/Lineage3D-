#pragma once

#include "core/Math.h"

#include <cstddef>
#include <span>
#include <string>

namespace assets {

// Refractive (M8 plan D10): water and glass, which refract the scene behind them (or blend when refraction is off).
enum class BlendMode : uint8_t { Opaque, AlphaTest, AlphaBlend, Additive, Premultiplied, Refractive };

// Material features that select shader permutations (M2 plan D1). Each one needs its texture, and each texture
// needs its feature. ALPHA_TEST isn't listed: it follows from "blend": "alphaTest".
enum MaterialFeature : uint32_t {
    kFeatureNormalMap = 1u << 0,    // "NORMAL_MAP"     ↔ textures.normal
    kFeatureSpecGlossMap = 1u << 1, // "SPEC_GLOSS_MAP" ↔ textures.specGloss
    kFeatureEmissionMap = 1u << 2,  // "EMISSION_MAP"   ↔ textures.emission
    kFeatureDetailLayer = 1u << 3,  // "DETAIL_LAYER"   ↔ textures.detail (M3: secondary scrolling layer)
};
inline constexpr uint32_t kMaterialFeatureCount = 4;

// .fmat (JSON, ARCHITECTURE.md §7.4; M2 plan D2, M3 plan D11):
//   { "template": "lit",
//     "features": ["NORMAL_MAP", "SPEC_GLOSS_MAP", "EMISSION_MAP", "DETAIL_LAYER"],
//     "textures": { "albedo": "textures/a.ktx2", "normal": "textures/n.ktx2", "specGloss": "textures/s.ktx2",
//                   "emission": "textures/e.ktx2", "detail": "textures/d.ktx2" },
//     "params": { "baseColor": [1, 1, 1, 1], "specular": [1, 1, 1], "gloss": 1, "emission": [1, 1, 1],
//                 "emissionIntensity": 2, "normalStrength": 1, "alphaCutoff": 0.5, "reflection": 1,
//                 "uvScroll": [0.1, 0], "uvRotate": 45, "flipbook": [4, 4, 8],
//                 "detailScale": 2, "detailScroll": [0, -0.05], "detailStrength": 0.6 },
//     "blend": "opaque", "doubleSided": false, "castShadows": true }
// Parsing is strict: unknown keys, unknown features, a feature without its texture or a texture without its
// feature fail the load. Colors are linear. A parameter whose map is present multiplies the map and defaults to 1;
// without the map it is the value itself (defaults: specular 0.04, gloss 0.25, emission 0).
// castShadows (M4 plan D10): opaque and alpha-tested materials cast sun and light shadows unless it is false: into the
// shadow maps and into the baked shadowmask and light visibility (they still occlude and bounce indirect light). Blended
// materials never cast, and saying true on one is an error.
// Texture animation applies to every UV0 lookup: rotate about (0.5, 0.5), then scroll, then pick the flipbook cell
// (the mesh's UVs cover one cell). The detail layer is a second sRGB RGBA texture on UV0 × detailScale, scrolling on
// its own, blended over the albedo by its alpha × detailStrength.
struct MaterialFile {
    std::string templateName = "lit";
    uint32_t features = 0; // MaterialFeature bits
    std::string albedo;    // data-relative .ktx2 paths; empty = none (albedo: white)
    std::string normal;
    std::string specGloss;
    std::string emission;
    std::string detail;
    glm::vec4 baseColor{1.0f};    // rgb tint, a opacity
    glm::vec3 specular{0.04f};    // specular color
    float gloss = 0.25f;          // 0–1 → Blinn-Phong exponent 2–2048
    glm::vec3 emissionColor{0.0f};
    float emissionIntensity = 1.0f;
    float normalStrength = 1.0f;  // scales the tangent-space XY of the normal map
    float alphaCutoff = 0.5f;
    float reflection = 1.0f;      // multiplies the environment reflection (0 = none)
    float refractionStrength = 0.3f; // refractive: m that normal-map bends shift the refracted image (M8 plan D10)
    glm::vec3 refractionTint{1.0f};  // refractive: the share of each channel the surface lets through
    glm::vec2 uvScroll{0.0f};     // UV units per second
    float uvRotate = 0.0f;        // degrees per second, about UV (0.5, 0.5)
    glm::vec3 flipbook{1.0f, 1.0f, 0.0f}; // columns, rows, frames per second
    float detailScale = 1.0f;     // detail UV = UV0 × scale + scroll × time
    glm::vec2 detailScroll{0.0f};
    float detailStrength = 1.0f;
    BlendMode blend = BlendMode::Opaque;
    bool doubleSided = false;
    bool castShadows = true; // always false for blended materials
};

bool parseMaterial(std::span<const std::byte> json, MaterialFile& out, std::string* error);
std::string writeMaterial(const MaterialFile& material); // every field explicit, so a re-read never depends on defaults

const char* blendModeName(BlendMode mode);
const char* materialFeatureName(uint32_t featureIndex); // index of the MaterialFeature bit

} // namespace assets
