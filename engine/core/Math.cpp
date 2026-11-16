#include "core/Math.h"

namespace core {

Affine3x4 Affine3x4::fromMat4(const glm::mat4& m) {
    Affine3x4 a;
    for (int r = 0; r < 3; ++r) a.rows[r] = glm::vec4(m[0][r], m[1][r], m[2][r], m[3][r]);
    return a;
}

glm::mat4 Affine3x4::toMat4() const {
    glm::mat4 m(1.0f);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) m[c][r] = rows[r][c];
    return m;
}

glm::vec3 Affine3x4::transformPoint(const glm::vec3& p) const {
    const glm::vec4 h(p, 1.0f);
    return {glm::dot(rows[0], h), glm::dot(rows[1], h), glm::dot(rows[2], h)};
}

glm::vec3 Affine3x4::transformVector(const glm::vec3& v) const {
    const glm::vec4 h(v, 0.0f);
    return {glm::dot(rows[0], h), glm::dot(rows[1], h), glm::dot(rows[2], h)};
}

glm::vec3 Affine3x4::transformNormal(const glm::vec3& n) const {
    // Columns of the linear part.
    const glm::vec3 a(rows[0].x, rows[1].x, rows[2].x);
    const glm::vec3 b(rows[0].y, rows[1].y, rows[2].y);
    const glm::vec3 c(rows[0].z, rows[1].z, rows[2].z);
    // cofactor(M) = det(M) · M^-T, whose columns are b×c, c×a, a×b. Multiplying by sign(det) keeps mirrored
    // transforms (negative scale) pointing outward, like the inverse transpose, without dividing by det.
    const glm::vec3 bc = glm::cross(b, c);
    const float detSign = glm::dot(a, bc) < 0.0f ? -1.0f : 1.0f;
    return detSign * (n.x * bc + n.y * glm::cross(c, a) + n.z * glm::cross(a, b));
}

} // namespace core
