#pragma once

#include "core/Math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace assets {

// Position quantization (ARCHITECTURE.md §4.1, §7.2): int16 grid coordinates with one uniform power-of-two
// step per mesh. A uniform step keeps normals and tangents unskewed under the cofactor transform, and a
// power-of-two step makes grid * step exact in float.
struct Dequantization {
    float step = 1.0f;
    glm::vec3 offset{0.0f}; // on the grid: an exact multiple of step
};

inline Dequantization chooseDequantization(const glm::vec3& boundsMin, const glm::vec3& boundsMax) {
    const glm::vec3 center = (boundsMin + boundsMax) * 0.5f;
    const glm::vec3 half = (boundsMax - boundsMin) * 0.5f;
    const float extent = std::max(std::max(half.x, half.y), std::max(half.z, 1e-6f));
    Dequantization d;
    d.step = std::exp2(std::ceil(std::log2(extent / 32767.0f)));
    // Snapping the offset onto the grid can move it by half a step; widen the step until everything fits.
    for (int attempt = 0; attempt < 4; ++attempt) {
        d.offset = glm::round(center / d.step) * d.step;
        const glm::vec3 reach = glm::max(glm::abs(boundsMax - d.offset), glm::abs(boundsMin - d.offset)) / d.step;
        if (std::max(reach.x, std::max(reach.y, reach.z)) <= 32767.0f) break;
        d.step *= 2.0f;
    }
    return d;
}

inline int16_t quantizeCoordinate(float value, float step, float offset) {
    const float q = std::round((value - offset) / step);
    return int16_t(std::clamp(q, -32767.0f, 32767.0f));
}

inline glm::vec3 dequantize(const int16_t q[3], const Dequantization& d) {
    return glm::vec3(float(q[0]), float(q[1]), float(q[2])) * d.step + d.offset;
}

// model · D, where D maps grid coordinates to object space: the matrix the GPU receives (§4.1).
inline core::Affine3x4 foldDequantization(const core::Affine3x4& model, const Dequantization& d) {
    core::Affine3x4 out;
    for (int r = 0; r < 3; ++r) {
        const glm::vec4& row = model.rows[r];
        const float translation = row.x * d.offset.x + row.y * d.offset.y + row.z * d.offset.z + row.w;
        out.rows[r] = glm::vec4(row.x * d.step, row.y * d.step, row.z * d.step, translation);
    }
    return out;
}

} // namespace assets
