#include "renderer/lighting/ProbeGrid.h"

#include <algorithm>
#include <cmath>

namespace renderer {

uint32_t ProbeGrid::validCount() const {
    uint32_t n = 0;
    for (const auto& p : m_data.probes) n += p.valid > 0.5f ? 1u : 0u;
    return n;
}

void ProbeGrid::sample(const glm::vec3& position, glm::vec4 sh[3], float& sunVisibility) const {
    const glm::vec3 p = (position - m_data.origin) / m_data.spacing;
    uint32_t i0[3], i1[3];
    float f[3];
    for (int a = 0; a < 3; ++a) {
        const float maxIndex = float(m_data.dims[a] - 1);
        // A non-finite position (a degenerate transform) would turn into an undefined index: take the grid's corner.
        const float c = std::isfinite(p[a]) ? std::clamp(p[a], 0.0f, maxIndex) : 0.0f;
        i0[a] = std::min(uint32_t(c), m_data.dims[a] - 1);
        i1[a] = std::min(i0[a] + 1, m_data.dims[a] - 1);
        f[a] = c - float(i0[a]);
    }
    for (int pass = 0; pass < 2; ++pass) { // pass 1 ignores validity: better than nothing deep inside geometry
        glm::vec4 sum[3] = {glm::vec4(0.0f), glm::vec4(0.0f), glm::vec4(0.0f)};
        float sun = 0.0f, total = 0.0f;
        for (uint32_t corner = 0; corner < 8; ++corner) {
            const uint32_t x = corner & 1 ? i1[0] : i0[0], y = corner & 2 ? i1[1] : i0[1], z = corner & 4 ? i1[2] : i0[2];
            const auto& probe = m_data.probes[(size_t(z) * m_data.dims[1] + y) * m_data.dims[0] + x];
            if (pass == 0 && probe.valid < 0.5f) continue;
            const float w = (corner & 1 ? f[0] : 1.0f - f[0]) * (corner & 2 ? f[1] : 1.0f - f[1]) * (corner & 4 ? f[2] : 1.0f - f[2]);
            for (int c = 0; c < 3; ++c) sum[c] += probe.sh[c] * w;
            sun += probe.sunOcclusion * w;
            total += w;
        }
        if (total > 1e-5f) {
            for (int c = 0; c < 3; ++c) sh[c] = sum[c] / total;
            sunVisibility = sun / total;
            return;
        }
    }
    // Only reached when every weight is zero, which needs all eight corners at zero weight: keep a neutral result.
    for (int c = 0; c < 3; ++c) sh[c] = glm::vec4(0.0f);
    sunVisibility = 1.0f;
}

void ambientSh(const glm::vec3& sky, const glm::vec3& ground, glm::vec4 out[3]) {
    for (int c = 0; c < 3; ++c) out[c] = glm::vec4(0.0f, (sky[c] - ground[c]) * 0.5f, 0.0f, (sky[c] + ground[c]) * 0.5f);
}

} // namespace renderer
