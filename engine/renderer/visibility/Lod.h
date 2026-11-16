#pragma once

#include <cstdint>

namespace renderer {

// LOD selection (ARCHITECTURE.md §4.15; M5 plan D5–D7). A mesh has 1–4 levels; level l ≥ 1 is good enough while the
// object's screen size is at most screenSize[l] (the cooker derives it from the level's simplification error).
inline constexpr uint32_t kMaxLodLevels = 4;
inline constexpr float kLodHysteresis = 0.1f;  // a finer level needs 10 % more than its threshold, a coarser 10 % less
inline constexpr float kLodFadeSeconds = 0.25f;

struct LodThresholds {
    float screenSize[kMaxLodLevels] = {}; // [0] unused; decreasing for l ≥ 1
    uint32_t count = 1;
};

// Fraction of the viewport height the bounding sphere's diameter covers: r / (distance · tan(fovY / 2)). The camera's
// orientation doesn't enter, so turning around never changes a level. Inside the sphere: very large (level 0).
float lodScreenSize(float radius, float distance, float tanHalfFovY);

// The coarsest level whose threshold is ≥ size / bias (bias > 1 picks coarser levels sooner). No hysteresis.
uint32_t selectLod(float size, const LodThresholds& t, float bias);
// From `current`: coarser only below the threshold × 0.9, finer only above the next threshold × 1.1.
uint32_t selectLodWithHysteresis(float size, uint32_t current, const LodThresholds& t, float bias);

// Per-object level and cross-fade state, owned by the job that culls the object.
struct LodState {
    uint8_t level = 0;      // the level drawn (the incoming one while fading)
    uint8_t previous = 0;   // the outgoing level while fading
    bool fading = false;
    float fadeStart = 0.0f; // renderer clock (seconds) when the fade began
    uint32_t lastFrame = 0; // last frame any view (camera or shadow) saw the object (0 = never)
};

struct LodDraw {
    uint32_t level = 0;
    uint32_t previous = 0; // valid while fading
    bool fading = false;
    float progress = 1.0f; // 0 → 1 over kLodFadeSeconds
};

// One frame in which some view sees the object. `forceLevel` ≥ 0 pins a level (no fade); an object that no view saw in
// the previous frame (or ever) snaps to its level without fading. `fade` = false switches instantly. `frame` starts at 1.
LodDraw updateLod(LodState& state, float size, const LodThresholds& t, float bias, int forceLevel, bool fade, uint32_t frame,
                  float clock);

// lights.w of a draw (M5 plan D7): fade progress in bits 0–7, incoming in bit 8, fading in bit 9, level in bits 12–13.
inline constexpr uint32_t kLodFadeIncoming = 1u << 8;
inline constexpr uint32_t kLodFading = 1u << 9;
uint32_t packLodWord(uint32_t level, bool fading, bool incoming, float progress);
inline uint32_t lodWordLevel(uint32_t word) { return (word >> 12) & 3u; }

} // namespace renderer
