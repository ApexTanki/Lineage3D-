#pragma once

#include "assets/MapFormat.h"
#include "core/Math.h"

namespace renderer {

// The baked L1 SH light probes of a map (ARCHITECTURE.md R4, M3 plan D6). Each frame every dynamic object samples the
// grid trilinearly at its bounding-sphere center; probes inside geometry are skipped and the weights renormalized.
// Read-only after set(), so job threads may sample concurrently.
class ProbeGrid {
public:
    void set(assets::ProbeGridData data) { m_data = std::move(data); }
    void clear() { m_data = assets::ProbeGridData{}; }
    bool empty() const { return m_data.probes.empty(); }
    uint32_t count() const { return uint32_t(m_data.probes.size()); }
    uint32_t validCount() const;

    // Positions outside the grid take its nearest boundary. If no probe of the cell is valid, all eight are used.
    void sample(const glm::vec3& position, glm::vec4 sh[3], float& sunVisibility) const;

private:
    assets::ProbeGridData m_data;
};

// The hemispherical ambient in the same SH form: mix(ground, sky, n.y/2 + 1/2) = (sky + ground)/2 + (sky − ground)/2 · n.y.
void ambientSh(const glm::vec3& sky, const glm::vec3& ground, glm::vec4 out[3]);

} // namespace renderer
