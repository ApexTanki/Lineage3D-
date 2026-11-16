#include "renderer/core/ClipSpace.h"

#include <cmath>

namespace renderer {

ClipSpace::ClipSpace(const rhi::ClipSpaceConvention& convention) : m_conv(convention) {
    // glm is column-major: m[column][row].
    if (convention.originTopLeft) m_fix[1][1] = -1.0f; // y' = -y
    if (!convention.depthZeroToOne) {                  // z' = 2z - w
        m_fix[2][2] = 2.0f;
        m_fix[3][2] = -1.0f;
    }
}

glm::mat4 ClipSpace::canonicalPerspective(float fovY, float aspect, float zNear, float zFar) {
    // Canonical: right-handed view space (camera looks down -Z), near → 0, far → 1.
    const float f = 1.0f / std::tan(fovY * 0.5f);
    glm::mat4 p(0.0f);
    p[0][0] = f / aspect;
    p[1][1] = f;
    p[2][2] = zFar / (zNear - zFar);
    p[2][3] = -1.0f;
    p[3][2] = zNear * zFar / (zNear - zFar);
    return p;
}

glm::mat4 ClipSpace::canonicalOrthographic(float left, float right, float bottom, float top, float zNear, float zFar) {
    glm::mat4 p(1.0f);
    p[0][0] = 2.0f / (right - left);
    p[1][1] = 2.0f / (top - bottom);
    p[2][2] = -1.0f / (zFar - zNear);
    p[3][0] = -(right + left) / (right - left);
    p[3][1] = -(top + bottom) / (top - bottom);
    p[3][2] = -zNear / (zFar - zNear);
    return p;
}

glm::mat4 ClipSpace::perspective(float fovY, float aspect, float zNear, float zFar) const {
    return m_fix * canonicalPerspective(fovY, aspect, zNear, zFar);
}

glm::mat4 ClipSpace::orthographic(float left, float right, float bottom, float top, float zNear, float zFar) const {
    return m_fix * canonicalOrthographic(left, right, bottom, top, zNear, zFar);
}

glm::mat4 ClipSpace::clipToUvDepth() const {
    // On both backends NDC y = -1 lands on texel row 0, so u and v are the same affine map; only depth differs.
    glm::mat4 m(1.0f);
    m[0][0] = 0.5f;
    m[1][1] = 0.5f;
    m[3][0] = 0.5f;
    m[3][1] = 0.5f;
    if (!m_conv.depthZeroToOne) {
        m[2][2] = 0.5f;
        m[3][2] = 0.5f;
    }
    return m;
}

glm::vec2 ClipSpace::depthToNdc() const {
    return m_conv.depthZeroToOne ? glm::vec2(1.0f, 0.0f) : glm::vec2(2.0f, -1.0f);
}

rhi::Rect ClipSpace::fromTopLeft(const rhi::Rect& r, uint32_t attachmentHeight) const {
    if (m_conv.originTopLeft) return r;
    return {r.x, int32_t(attachmentHeight) - r.y - int32_t(r.height), r.width, r.height};
}

} // namespace renderer
