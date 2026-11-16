#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Quality tiers (data/settings/tiers.json; ARCHITECTURE.md §7.9, M9 plan D1). Plain data: renderer/core/QualityTiers maps
// a tier onto the renderer's FrameSettings, and the resource manager takes its memory budgets. One entry:
//
//   { "name": "medium", "hdr": true, "msaa": 2, "fxaa": false,
//     "shadows": { "cascades": 2, "size": 1024, "filter": "hardware", "staticCasters": false, "spotShadows": false,
//                  "blobShadows": true, "blobStrength": 0.35, "distance": 60 },
//     "ssao": false, "bloom": true, "softParticles": true, "refraction": true, "heatHaze": true,
//     "p2": { "depthOfField": false, "motionBlur": false, "godRays": false, "ssr": false },
//     "maxLocalLights": 4, "anisotropy": 4,
//     "budgets": { "textures": 512, "meshes": 192, "targets": 224, "buffers": 96 } }
//
// The file holds exactly the four tiers low, medium, high and custom ({ "tiers": [ ... ] }). Parsing is strict, like the
// other formats: every key must be present, unknown or repeated keys and out-of-range values fail, and so do effects that
// only exist on the HDR path (SSAO, bloom, refraction, heat haze, the P2 effects) in a tier without HDR. fl_assetc runs
// the same parser when it cooks, so a bad file fails the build. Budgets are MiB.

namespace assets {

enum class TierShadowFilter : uint8_t { Hardware, Poisson };

struct TierDesc {
    std::string name;
    bool hdr = true;
    uint32_t msaa = 1; // 1, 2 or 4
    bool fxaa = false;
    // Shadows (renderer::ShadowSettings).
    uint32_t cascades = 3;          // 0–3; 0 = blob shadows only
    uint32_t shadowMapSize = 2048;  // a power of two from 256 to 4096
    TierShadowFilter shadowFilter = TierShadowFilter::Poisson;
    bool staticCasters = true;
    bool spotShadows = true;
    bool blobShadows = true;
    float blobStrength = 0.35f;     // 0–1
    float shadowDistance = 120.0f;  // m, 10–1000
    // Post-processing and particles (renderer::PostSettings, FrameSettings).
    bool ssao = false;
    bool bloom = true;
    bool softParticles = true;
    bool refraction = true;
    bool heatHaze = true;
    bool depthOfField = false;
    bool motionBlur = false;
    bool godRays = false;
    bool ssr = false;
    uint32_t maxLocalLights = 4;    // 0–8
    uint32_t anisotropy = 4;        // 1–16
    // GPU memory budgets in MiB (M9 plan D6), 16–65536 each.
    uint32_t textureBudgetMiB = 512;
    uint32_t meshBudgetMiB = 256;
    uint32_t targetBudgetMiB = 256;
    uint32_t bufferBudgetMiB = 64;
};

struct TierFile {
    std::vector<TierDesc> tiers; // low, medium, high, custom (in file order)
};

inline constexpr const char* kTierNames[] = {"low", "medium", "high", "custom"};

bool parseTiers(std::span<const std::byte> json, TierFile& out, std::string* error = nullptr);
const TierDesc* findTier(const TierFile& file, std::string_view name);

} // namespace assets
