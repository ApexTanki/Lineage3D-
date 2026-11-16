#pragma once

#include <algorithm>
#include <cstdint>

namespace renderer {

// Refraction (water, glass; M8 plan D10) draws before the transparents: they blend over it, and it writes depth.
enum class QueuePass : uint8_t { Shadow, DepthPrepass, Opaque, AlphaTest, Decal, Refraction, Transparent };
enum class Translucency : uint8_t { Opaque, AlphaTest, Blended };

// 64-bit keys, sorted ascending (ARCHITECTURE.md §5.4):
//   opaque / alpha-test: pass:4 | layer:4 | translucency:2 | pipeline:12 | material:16 | depth:18 (near→far) | sub:8
//   blended:             pass:4 | layer:4 | translucency:2 | depth:24 (far→near) | pipeline:12 | material:16 | sub:2
// Opaques sort state-first and front-to-back only within a state group; blended draws must sort depth-first.
namespace sortkey {

// In double precision: in float, (2^24 − 1) + 0.5 rounds up to 2^24, which overflowed the 24-bit blended field into the
// pass bits. The clamp keeps the result inside the field whatever the rounding.
inline uint64_t quantizeDepth(float depth01, uint32_t bits) {
    const double d = std::clamp(double(depth01), 0.0, 1.0);
    const uint64_t maxValue = (1ull << bits) - 1;
    return std::min(maxValue, uint64_t(d * double(maxValue) + 0.5));
}

inline uint64_t opaque(QueuePass pass, uint8_t layer, Translucency t, uint16_t pipeline, uint16_t material, float depth01,
                       uint8_t sub = 0) {
    return (uint64_t(pass) & 0xF) << 60 | (uint64_t(layer) & 0xF) << 56 | (uint64_t(t) & 0x3) << 54 |
           (uint64_t(pipeline) & 0xFFF) << 42 | uint64_t(material) << 26 | quantizeDepth(depth01, 18) << 8 | uint64_t(sub);
}

inline uint64_t blended(QueuePass pass, uint8_t layer, float depth01, uint16_t pipeline, uint16_t material, uint8_t sub = 0) {
    const uint64_t farFirst = ((1ull << 24) - 1) - quantizeDepth(depth01, 24);
    return (uint64_t(pass) & 0xF) << 60 | (uint64_t(layer) & 0xF) << 56 | uint64_t(Translucency::Blended) << 54 |
           farFirst << 30 | (uint64_t(pipeline) & 0xFFF) << 18 | uint64_t(material) << 2 | (uint64_t(sub) & 0x3);
}

} // namespace sortkey
} // namespace renderer
