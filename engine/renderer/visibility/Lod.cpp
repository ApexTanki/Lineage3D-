#include "renderer/visibility/Lod.h"

#include <algorithm>
#include <cmath>

namespace renderer {

float lodScreenSize(float radius, float distance, float tanHalfFovY) {
    if (!(distance > radius)) return 1e9f; // the camera is inside the bounding sphere
    return radius / std::max(distance * tanHalfFovY, 1e-6f);
}

uint32_t selectLod(float size, const LodThresholds& t, float bias) {
    const float s = size / std::max(bias, 1e-3f);
    uint32_t level = 0;
    for (uint32_t l = 1; l < t.count && l < kMaxLodLevels; ++l)
        if (s <= t.screenSize[l]) level = l;
    return level;
}

uint32_t selectLodWithHysteresis(float size, uint32_t current, const LodThresholds& t, float bias) {
    const uint32_t count = std::clamp<uint32_t>(t.count, 1u, kMaxLodLevels);
    current = std::min(current, count - 1);
    const float s = size / std::max(bias, 1e-3f);
    // The coarsest level that size clearly fits (10 % under its threshold), and the coarsest one it still fits with 10 %
    // of slack: go coarser to the first if it is coarser, finer to the second if the current level is too coarse even
    // with the slack, else stay.
    uint32_t strict = 0, lenient = 0;
    for (uint32_t l = 1; l < count; ++l) {
        if (s <= t.screenSize[l] * (1.0f - kLodHysteresis)) strict = l;
        if (s <= t.screenSize[l] * (1.0f + kLodHysteresis)) lenient = l;
    }
    if (strict > current) return strict;
    if (lenient < current) return lenient;
    return current;
}

LodDraw updateLod(LodState& state, float size, const LodThresholds& t, float bias, int forceLevel, bool fade, uint32_t frame,
                  float clock) {
    const uint32_t count = std::clamp<uint32_t>(t.count, 1u, kMaxLodLevels);
    const bool continuing = state.lastFrame != 0 && state.lastFrame + 1 == frame;
    state.lastFrame = frame;
    if (forceLevel >= 0) {
        state.level = uint8_t(std::min<uint32_t>(uint32_t(forceLevel), count - 1));
        state.fading = false;
    } else if (!continuing) {
        state.level = uint8_t(selectLod(size, t, bias));
        state.fading = false;
    } else {
        const uint32_t next = selectLodWithHysteresis(size, state.level, t, bias);
        if (next != state.level) {
            if (fade) {
                // A change during a fade restarts it from the level that was coming in.
                state.previous = state.level;
                state.fading = true;
                state.fadeStart = clock;
            } else {
                state.fading = false;
            }
            state.level = uint8_t(next);
        }
    }
    LodDraw draw;
    draw.level = state.level;
    if (state.fading) {
        const float progress = (clock - state.fadeStart) / kLodFadeSeconds;
        if (progress >= 1.0f || progress < 0.0f || !fade) {
            state.fading = false;
        } else {
            draw.fading = true;
            draw.previous = state.previous;
            draw.progress = progress;
        }
    }
    return draw;
}

uint32_t packLodWord(uint32_t level, bool fading, bool incoming, float progress) {
    const uint32_t p = uint32_t(std::lround(std::clamp(progress, 0.0f, 1.0f) * 255.0f));
    return p | (incoming ? kLodFadeIncoming : 0u) | (fading ? kLodFading : 0u) | (level & 3u) << 12;
}

} // namespace renderer
