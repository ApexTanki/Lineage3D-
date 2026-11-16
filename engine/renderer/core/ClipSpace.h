#pragma once

#include "core/Math.h"
#include "rhi/Types.h"

namespace renderer {

// The ONLY code that knows backend clip-space and framebuffer conventions (ARCHITECTURE.md §4.2).
// Everything else builds canonical matrices (right-handed, Y-up, z in [0,1], standard Z) and asks this class.
//   Vulkan: the Y flip is folded into the projection; viewports stay positive.
//   GL:     z is remapped from [0,1] to [-1,1].
// Result: CCW front faces and identical depth-buffer contents on both backends.
class ClipSpace {
public:
    explicit ClipSpace(const rhi::ClipSpaceConvention& convention);

    glm::mat4 perspective(float fovY, float aspect, float zNear, float zFar) const;
    glm::mat4 orthographic(float left, float right, float bottom, float top, float zNear, float zFar) const;
    glm::mat4 fix(const glm::mat4& canonicalProjection) const { return m_fix * canonicalProjection; }
    // The canonical matrices (right-handed view space looking down −Z, near → depth 0, far → 1) before fix().
    static glm::mat4 canonicalPerspective(float fovY, float aspect, float zNear, float zFar);
    static glm::mat4 canonicalOrthographic(float left, float right, float bottom, float top, float zNear, float zFar);
    // Backend NDC z of depth 0 (the near plane): 0 on Vulkan, −1 on GL. Shadow casters are clamped to it (pancaking).
    float nearClipZ() const { return depthToNdc().y; }

    // Backend clip space → (u, v, depth01) for shadow and other projective lookups.
    glm::mat4 clipToUvDepth() const;
    // (zScale, zBias): turns a depth-buffer value into backend NDC z for position reconstruction.
    glm::vec2 depthToNdc() const;
    // Multiply dFdy() by this to get a "screen-up" derivative (rule C5).
    float dFdySign() const { return m_conv.originTopLeft ? -1.0f : 1.0f; }
    // Converts a top-left-origin pixel rectangle (UI, ImGui) into attachment texel rows (rule C6).
    rhi::Rect fromTopLeft(const rhi::Rect& topLeft, uint32_t attachmentHeight) const;
    bool readbackRowsBottomUp() const { return !m_conv.originTopLeft; }

private:
    rhi::ClipSpaceConvention m_conv;
    glm::mat4 m_fix{1.0f};
};

} // namespace renderer
