#pragma once

#include "core/Math.h"

#include <cstdint>
#include <vector>

namespace tools {

// Linear RGB, rows top to bottom.
struct HdrImage {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<glm::vec3> pixels;

    HdrImage() = default;
    HdrImage(uint32_t w, uint32_t h) : width(w), height(h), pixels(size_t(w) * h, glm::vec3(0.0f)) {}
    glm::vec3& at(uint32_t x, uint32_t y) { return pixels[size_t(y) * width + x]; }
    const glm::vec3& at(uint32_t x, uint32_t y) const { return pixels[size_t(y) * width + x]; }
};

uint16_t floatToHalf(float value); // round to nearest even; overflow saturates to infinity
float halfToFloat(uint16_t half);

// BC6H, unsigned float variant (M3 plan D5). libktx 4.4 has no BC6H encoder or transcoder, so lightmaps and
// environment cubes are encoded here. The encoder uses the four one-region modes (11–14): it fits each block's
// endpoints along its principal axis in the half-float bit domain the format interpolates in, tries every mode, refines
// the endpoints by least squares once, and keeps the mode whose exact decode is closest. Two-region modes, which help
// sharp edges, aren't used: lightmaps are smooth (sun shadows live in the shadowmask).
void encodeBc6hBlock(const glm::vec3 texels[16], uint8_t out[16]);
// Decodes blocks in modes 11–14 (what the encoder writes); returns false for other modes.
bool decodeBc6hBlock(const uint8_t block[16], glm::vec3 out[16]);
// Whole image, blocks row-major; edge blocks of sizes that aren't multiples of 4 replicate the last row/column.
std::vector<uint8_t> encodeBc6hImage(const HdrImage& image);
HdrImage decodeBc6hImage(const std::vector<uint8_t>& blocks, uint32_t width, uint32_t height);

// RGBM with M = 8 in linear 8-bit RGBA: a = ⌈max(r, g, b) / 8 · 255⌉ / 255, rgb = color / (8a). Colors above 8 clip.
inline constexpr float kRgbmRange = 8.0f;
void encodeRgbm(const glm::vec3& color, uint8_t out[4]);
glm::vec3 decodeRgbm(const uint8_t rgba[4]);

} // namespace tools
