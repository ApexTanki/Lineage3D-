#pragma once

#include <cstdint>

namespace renderer {

// How shadow maps are filtered (M4 plan D5).
enum class ShadowFilter : uint8_t {
    Hardware2x2, // one comparison tap; with linear depth compare the hardware blends a 2×2 footprint (Medium)
    Poisson16    // 16 taps on a Poisson disk rotated per pixel, each a hardware 2×2 tap (High)
};

// The shadow part of the quality tiers (ARCHITECTURE.md §7.9; M4 plan D1). The other tier settings (MSAA, light
// counts, post) join them in one tier switch in M9.
struct ShadowSettings {
    uint8_t cascades = 3;        // 0 = no shadow map: blob shadows only (Low)
    uint16_t mapSize = 2048;     // per cascade; clamped to the device
    ShadowFilter filter = ShadowFilter::Poisson16;
    bool staticCasters = true;   // static geometry also casts into the cascades (High)
    bool spotShadows = true;     // up to 4 spot lights with castsShadows get an atlas tile (High)
    bool blobShadows = true;     // models flagged blobShadow get a grounding quad
    float blobStrength = 0.35f;  // darkening at the blob's center: 0.6 when it is the only dynamic shadow (Low)
    float distance = 120.0f;     // meters covered by the cascades; shadows fade out over the last 10 %

    static ShadowSettings low() { return {0, 1024, ShadowFilter::Hardware2x2, false, false, true, 0.6f, 60.0f}; }
    static ShadowSettings medium() { return {2, 1024, ShadowFilter::Hardware2x2, false, false, true, 0.35f, 60.0f}; }
    static ShadowSettings high() { return {3, 2048, ShadowFilter::Poisson16, true, true, true, 0.35f, 120.0f}; }
};

inline constexpr uint32_t kMaxCascades = 3;
inline constexpr uint32_t kMaxSpotShadows = 4;     // atlas tiles
inline constexpr uint32_t kSpotAtlasSize = 2048;   // 2 × 2 tiles of 1024²
inline constexpr uint32_t kSpotTileSize = kSpotAtlasSize / 2;
inline constexpr float kMaxSpotShadowConeDeg = 75.0f; // wider spots get no shadow (fov = 2 × outer cone)

} // namespace renderer
