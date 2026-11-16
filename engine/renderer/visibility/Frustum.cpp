#include "renderer/visibility/Frustum.h"

#include <emmintrin.h>

namespace renderer {

Frustum Frustum::fromViewProjection(const glm::mat4& m, bool nearPlane, bool farPlane) {
    // Clip-space rows: inside is −w ≤ x ≤ w, −w ≤ y ≤ w, 0 ≤ z ≤ w (glm is column-major: row i = m[0..3][i]).
    auto row = [&](int i) { return glm::vec4(m[0][i], m[1][i], m[2][i], m[3][i]); };
    const glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    Frustum f;
    auto add = [&](const glm::vec4& p) {
        const float length = glm::length(glm::vec3(p));
        f.planes[f.count++] = length > 0.0f ? p / length : p;
    };
    add(r3 + r0); // left
    add(r3 - r0); // right
    add(r3 + r1); // bottom
    add(r3 - r1); // top
    if (nearPlane) add(r2);
    if (farPlane) add(r3 - r2);
    return f;
}

bool Frustum::sphereVisible(const glm::vec3& c, float radius) const {
    for (uint32_t i = 0; i < count; ++i) {
        const glm::vec4& p = planes[i];
        if (p.x * c.x + p.y * c.y + p.z * c.z + p.w < -radius) return false;
    }
    return true;
}

bool Frustum::boxVisible(const glm::vec3& lo, const glm::vec3& hi) const {
    for (uint32_t i = 0; i < count; ++i) {
        const glm::vec4& p = planes[i];
        const glm::vec3 far(p.x >= 0.0f ? hi.x : lo.x, p.y >= 0.0f ? hi.y : lo.y, p.z >= 0.0f ? hi.z : lo.z);
        if (p.x * far.x + p.y * far.y + p.z * far.z + p.w < 0.0f) return false;
    }
    return true;
}

uint8_t cullSphereScalar(const Frustum* const* frusta, uint32_t viewCount, const SphereArrays& s, uint32_t index) {
    if (index >= s.count) return 0;
    const float x = s.x[index], y = s.y[index], z = s.z[index], r = s.r[index];
    uint8_t bits = 0;
    for (uint32_t v = 0; v < viewCount && v < kMaxCullViews; ++v) {
        const Frustum& f = *frusta[v];
        bool outside = false, straddles = false;
        for (uint32_t i = 0; i < f.count; ++i) {
            const glm::vec4& p = f.planes[i];
            // Same operations in the same order as the SSE2 path, so both give bit-identical answers.
            const float dist = p.x * x + p.y * y + p.z * z + p.w;
            outside = outside || dist < 0.0f - r;
            straddles = straddles || dist < r;
        }
        if (outside) continue;
        if (straddles && s.boxMin && !f.boxVisible(s.boxMin[index], s.boxMax[index])) continue;
        bits = uint8_t(bits | (1u << v));
    }
    return bits;
}

void cullSpheres4(const Frustum* const* frusta, uint32_t viewCount, const SphereArrays& s, uint32_t first, uint8_t out[4]) {
    out[0] = out[1] = out[2] = out[3] = 0;
    if (first >= s.count) return;
    const __m128 x = _mm_loadu_ps(s.x + first);
    const __m128 y = _mm_loadu_ps(s.y + first);
    const __m128 z = _mm_loadu_ps(s.z + first);
    const __m128 r = _mm_loadu_ps(s.r + first);
    const __m128 negR = _mm_sub_ps(_mm_setzero_ps(), r);
    const uint32_t live = s.count - first < 4 ? s.count - first : 4;
    for (uint32_t v = 0; v < viewCount && v < kMaxCullViews; ++v) {
        const Frustum& f = *frusta[v];
        __m128 outside = _mm_setzero_ps(), straddles = _mm_setzero_ps();
        for (uint32_t i = 0; i < f.count; ++i) {
            const glm::vec4& p = f.planes[i];
            const __m128 dist = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(p.x), x), _mm_mul_ps(_mm_set1_ps(p.y), y)),
                                                      _mm_mul_ps(_mm_set1_ps(p.z), z)),
                                           _mm_set1_ps(p.w));
            outside = _mm_or_ps(outside, _mm_cmplt_ps(dist, negR));
            straddles = _mm_or_ps(straddles, _mm_cmplt_ps(dist, r));
        }
        const int outsideBits = _mm_movemask_ps(outside);
        const int straddleBits = _mm_movemask_ps(straddles);
        for (uint32_t k = 0; k < live; ++k) {
            if (outsideBits & (1 << k)) continue;
            if ((straddleBits & (1 << k)) && s.boxMin && !f.boxVisible(s.boxMin[first + k], s.boxMax[first + k])) continue;
            out[k] = uint8_t(out[k] | (1u << v));
        }
    }
}

} // namespace renderer
