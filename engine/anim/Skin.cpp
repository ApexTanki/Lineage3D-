#include "anim/Skin.h"

#include <cfloat>

namespace anim {

core::Affine3x4 inverseBindOf(const assets::ModelJoint& joint) {
    core::Affine3x4 m;
    for (int r = 0; r < 3; ++r)
        m.rows[r] = glm::vec4(joint.inverseBind[4 * r + 0], joint.inverseBind[4 * r + 1], joint.inverseBind[4 * r + 2], joint.inverseBind[4 * r + 3]);
    return m;
}

void writePalette(const core::Affine3x4& instance, std::span<const core::Affine3x4> nodeModel,
                  std::span<const assets::ModelJoint> joints, const assets::Dequantization& dequant, core::Affine3x4* out) {
    for (size_t j = 0; j < joints.size(); ++j) {
        const assets::ModelJoint& joint = joints[j];
        const core::Affine3x4 node = joint.node < nodeModel.size() ? nodeModel[joint.node] : core::Affine3x4{};
        out[j] = assets::foldDequantization(multiply(multiply(instance, node), inverseBindOf(joint)), dequant);
    }
}

bool skinBounds(const core::Affine3x4& instance, std::span<const core::Affine3x4> nodeModel, std::span<const assets::ModelJoint> joints,
                glm::vec3& outMin, glm::vec3& outMax) {
    glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
    bool any = false;
    for (const assets::ModelJoint& joint : joints) {
        if (joint.radius < 0.0f || joint.node >= nodeModel.size()) continue;
        const core::Affine3x4 world = multiply(instance, nodeModel[joint.node]);
        const glm::vec3 center(world.rows[0].w, world.rows[1].w, world.rows[2].w);
        const float r = joint.radius * maxScale(world);
        lo = glm::min(lo, center - r);
        hi = glm::max(hi, center + r);
        any = true;
    }
    if (any) {
        outMin = lo;
        outMax = hi;
    }
    return any;
}

} // namespace anim
