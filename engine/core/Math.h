#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace core {

// Affine object→world transform stored as three rows: exactly the layout the GPU receives
// (ARCHITECTURE.md §4.1). Applying it is three dot products, and it avoids std140 matrix-layout
// ambiguity.
struct Affine3x4 {
    glm::vec4 rows[3] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};

    static Affine3x4 fromMat4(const glm::mat4& m);
    glm::mat4 toMat4() const;

    glm::vec3 transformPoint(const glm::vec3& p) const;
    glm::vec3 transformVector(const glm::vec3& v) const;
    // Transforms a normal by the cofactor of the linear part, times sign(det). Same direction as the inverse
    // transpose (correct under non-uniform scale and mirroring) without a division; not normalized.
    // The vertex shaders use the same formula.
    glm::vec3 transformNormal(const glm::vec3& n) const;
};

static_assert(sizeof(Affine3x4) == 48, "Affine3x4 must match the GPU layout");

} // namespace core
