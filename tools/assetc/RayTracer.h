#pragma once

#include "core/Math.h"

#include <cstdint>
#include <vector>

namespace tools {

struct Ray {
    glm::vec3 origin{0.0f};
    glm::vec3 direction{0.0f, 0.0f, 1.0f}; // unit length
    float tMax = 1e30f;
};

struct RayHit {
    float t = 0.0f;
    uint32_t triangle = 0; // index in the order triangles were added
    float u = 0.0f, v = 0.0f; // barycentrics of corners 1 and 2
    bool backFace = false;    // the ray hit the side opposite the counter-clockwise front
};

// Rejects hits the ray should pass through (alpha-tested texels): return false to ignore the hit.
using HitFilter = bool (*)(const void* context, uint32_t triangle, float u, float v);

// A static triangle BVH for the offline baker (M3 plan D1): binned-SAH build, at most 4 triangles per leaf, single
// rays traversed near child first. Both faces are hit; RayHit::backFace says which.
class RayScene {
public:
    void addTriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
    void build();

    bool intersect(const Ray& ray, RayHit& hit, HitFilter filter = nullptr, const void* context = nullptr) const;
    bool occluded(const Ray& ray, HitFilter filter = nullptr, const void* context = nullptr) const;

    uint32_t triangleCount() const { return uint32_t(m_triangles.size()); }
    glm::vec3 boundsMin() const { return m_nodes.empty() ? glm::vec3(0.0f) : m_nodes[0].lo; }
    glm::vec3 boundsMax() const { return m_nodes.empty() ? glm::vec3(0.0f) : m_nodes[0].hi; }

private:
    struct Triangle {
        glm::vec3 v0, e1, e2;
    };
    struct Node {
        glm::vec3 lo{INFINITY};
        uint32_t first = 0; // leaf: first entry in m_order; inner: index of the left child (right = left + 1)
        glm::vec3 hi{-INFINITY};
        uint32_t count = 0; // > 0: leaf
    };
    void subdivide(uint32_t node, std::vector<glm::vec3>& centroids, std::vector<glm::vec3>& boxLo, std::vector<glm::vec3>& boxHi);
    template <bool AnyHit>
    bool traverse(const Ray& ray, RayHit* hit, HitFilter filter, const void* context) const;

    std::vector<Triangle> m_triangles;
    std::vector<uint32_t> m_order;
    std::vector<Node> m_nodes;
};

} // namespace tools
