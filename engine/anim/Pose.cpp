#include "anim/Pose.h"

#include <algorithm>
#include <cmath>

namespace anim {

core::Affine3x4 toAffine(const Transform& t) {
    const glm::mat3 r = glm::mat3_cast(t.rotation); // column-major: r[column][row]
    core::Affine3x4 m;
    for (int row = 0; row < 3; ++row)
        m.rows[row] = glm::vec4(r[0][row] * t.scale.x, r[1][row] * t.scale.y, r[2][row] * t.scale.z, t.translation[row]);
    return m;
}

core::Affine3x4 multiply(const core::Affine3x4& a, const core::Affine3x4& b) {
    core::Affine3x4 out;
    for (int r = 0; r < 3; ++r) {
        const glm::vec4& ar = a.rows[r];
        out.rows[r] = ar.x * b.rows[0] + ar.y * b.rows[1] + ar.z * b.rows[2] + glm::vec4(0.0f, 0.0f, 0.0f, ar.w);
    }
    return out;
}

float maxScale(const core::Affine3x4& m) {
    glm::vec3 c[3];
    float best = 0.0f, sum = 0.0f;
    for (int i = 0; i < 3; ++i) {
        c[i] = glm::vec3(m.rows[0][i], m.rows[1][i], m.rows[2][i]);
        const float len2 = glm::dot(c[i], c[i]);
        best = std::max(best, len2);
        sum += len2;
    }
    // Orthogonal columns (rotation × scale, the usual case): the stretch is the longest column, exactly. A hierarchy with
    // non-uniform scale under rotation shears, and then the longest column can understate it; the Frobenius norm never
    // does.
    const float tolerance = 1e-4f * best;
    const bool orthogonal = std::fabs(glm::dot(c[0], c[1])) <= tolerance && std::fabs(glm::dot(c[0], c[2])) <= tolerance &&
                            std::fabs(glm::dot(c[1], c[2])) <= tolerance;
    return std::sqrt(orthogonal ? best : sum);
}

glm::quat nlerp(const glm::quat& a, const glm::quat& b, float t) {
    const float sign = glm::dot(a, b) < 0.0f ? -1.0f : 1.0f;
    const glm::quat q = a * (1.0f - t) + b * (sign * t);
    const float len2 = glm::dot(q, q);
    return len2 > 1e-20f ? q * (1.0f / std::sqrt(len2)) : a;
}

Transform blend(const Transform& a, const Transform& b, float t) {
    Transform out;
    out.translation = a.translation + (b.translation - a.translation) * t;
    out.rotation = nlerp(a.rotation, b.rotation, t);
    out.scale = a.scale + (b.scale - a.scale) * t;
    return out;
}

void evaluateHierarchy(std::span<const int32_t> parent, std::span<const Transform> local, std::span<core::Affine3x4> outModel) {
    const size_t n = std::min({parent.size(), local.size(), outModel.size()});
    for (size_t i = 0; i < n; ++i) {
        const core::Affine3x4 l = toAffine(local[i]);
        const int32_t p = parent[i];
        outModel[i] = p >= 0 && size_t(p) < i ? multiply(outModel[size_t(p)], l) : l;
    }
}

void maskBranch(std::span<const int32_t> parent, uint32_t root, float weight, std::span<float> mask) {
    const size_t n = std::min(parent.size(), mask.size());
    if (root >= n) return;
    // Parents come first, so only nodes after the root can be below it; walk each one up until it passes the root.
    // Masks are built once at setup, so the O(nodes × depth) walk doesn't matter.
    for (size_t i = root; i < n; ++i) {
        int32_t p = int32_t(i);
        while (p > int32_t(root)) p = parent[size_t(p)];
        if (p == int32_t(root)) mask[i] = weight;
    }
}

} // namespace anim
