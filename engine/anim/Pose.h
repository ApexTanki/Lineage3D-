#pragma once

#include "core/Math.h"

#include <cstdint>
#include <span>

// Pose math for the animation library (ARCHITECTURE.md §5.5, M6 plan D1, D6). The node hierarchy of a model is its
// skeleton: nodes are ordered parents first, so one forward pass turns local transforms into model space. Everything
// here is a pure function: no allocation and no shared state, so poses evaluate on any job thread.

namespace anim {

struct Transform {
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f}; // glm order: w, x, y, z
    glm::vec3 scale{1.0f};
};

// T · R · S as an affine row matrix.
core::Affine3x4 toAffine(const Transform& t);
// a · b (apply b first).
core::Affine3x4 multiply(const core::Affine3x4& a, const core::Affine3x4& b);
// An upper bound on how far the linear part can stretch a unit distance (culling bounds, M6 plan D10): exact for rotation
// × scale, conservative (Frobenius) when a hierarchy shears.
float maxScale(const core::Affine3x4& m);

// Normalized lerp along the shorter arc (the spec's nlerp; keys are dense enough that it tracks slerp closely).
glm::quat nlerp(const glm::quat& a, const glm::quat& b, float t);
// Per channel: lerp translation and scale, nlerp rotation.
Transform blend(const Transform& a, const Transform& b, float t);

// local → model space over a parents-first hierarchy (parent[i] < i, or −1 for roots).
void evaluateHierarchy(std::span<const int32_t> parent, std::span<const Transform> local, std::span<core::Affine3x4> outModel);

// mask[node] = weight for `root` and every node below it, leaving the others as they are: a per-node layer mask such as
// a turret layer (turret, gun and what hangs off them).
void maskBranch(std::span<const int32_t> parent, uint32_t root, float weight, std::span<float> mask);

} // namespace anim
