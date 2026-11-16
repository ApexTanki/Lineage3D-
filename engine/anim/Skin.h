#pragma once

#include "anim/Pose.h"
#include "assets/ModelFormat.h"
#include "assets/Quantize.h"

#include <span>

// Skinning on the CPU side (M6 plan D1, D9, D10): the bone palette a skinned draw reads, and the conservative bounds that
// culling uses instead of skinning vertices.

namespace anim {

// palette[j] = instance · nodeModel[joints[j].node] · inverseBind[j] · dequant: grid coordinates straight to world space,
// so the vertex shader only blends up to four of these. Writes joints.size() entries.
void writePalette(const core::Affine3x4& instance, std::span<const core::Affine3x4> nodeModel,
                  std::span<const assets::ModelJoint> joints, const assets::Dequantization& dequant, core::Affine3x4* out);

// World AABB that contains every vertex of the skinned mesh in this pose: the union over its joints of the joint's world
// position ± (joint-space radius × the joint matrix's largest scale). A linear-blend-skinned vertex is a convex
// combination of points inside those spheres, so it lies inside their box. Joints with a negative radius influence no
// vertex and are skipped. False when no joint contributes (the box is left as it was).
bool skinBounds(const core::Affine3x4& instance, std::span<const core::Affine3x4> nodeModel, std::span<const assets::ModelJoint> joints,
                glm::vec3& outMin, glm::vec3& outMax);

// The affine matrix of a ModelJoint's inverse bind rows.
core::Affine3x4 inverseBindOf(const assets::ModelJoint& joint);

} // namespace anim
