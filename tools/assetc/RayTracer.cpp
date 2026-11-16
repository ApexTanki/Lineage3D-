#include "RayTracer.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tools {
namespace {

constexpr uint32_t kBins = 12;
constexpr uint32_t kLeafSize = 4;
constexpr float kMinT = 1e-5f;

float surfaceArea(const glm::vec3& lo, const glm::vec3& hi) {
    const glm::vec3 d = glm::max(hi - lo, glm::vec3(0.0f));
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

// Ray–box entry distance, or +inf when the box is missed or lies beyond tMax.
float slab(const glm::vec3& lo, const glm::vec3& hi, const glm::vec3& origin, const glm::vec3& inv, float tMax) {
    const glm::vec3 t0 = (lo - origin) * inv, t1 = (hi - origin) * inv;
    const glm::vec3 tNear = glm::min(t0, t1), tFar = glm::max(t0, t1);
    const float enter = std::max(std::max(tNear.x, tNear.y), std::max(tNear.z, 0.0f));
    const float exit = std::min(std::min(tFar.x, tFar.y), std::min(tFar.z, tMax));
    return enter <= exit ? enter : INFINITY;
}

} // namespace

void RayScene::addTriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
    m_triangles.push_back({a, b - a, c - a});
}

void RayScene::build() {
    const auto count = uint32_t(m_triangles.size());
    m_order.resize(count);
    std::iota(m_order.begin(), m_order.end(), 0u);
    m_nodes.clear();
    if (count == 0) return;
    m_nodes.reserve(size_t(count) * 2);
    std::vector<glm::vec3> centroids(count), boxLo(count), boxHi(count);
    for (uint32_t i = 0; i < count; ++i) {
        const Triangle& t = m_triangles[i];
        const glm::vec3 b = t.v0 + t.e1, c = t.v0 + t.e2;
        boxLo[i] = glm::min(t.v0, glm::min(b, c));
        boxHi[i] = glm::max(t.v0, glm::max(b, c));
        centroids[i] = (boxLo[i] + boxHi[i]) * 0.5f;
    }
    Node root;
    root.first = 0;
    root.count = count;
    m_nodes.push_back(root);
    subdivide(0, centroids, boxLo, boxHi);
}

void RayScene::subdivide(uint32_t rootIndex, std::vector<glm::vec3>& centroids, std::vector<glm::vec3>& boxLo, std::vector<glm::vec3>& boxHi) {
    std::vector<uint32_t> pending{rootIndex};
    while (!pending.empty()) {
        const uint32_t index = pending.back();
        pending.pop_back();
        Node& node = m_nodes[index];
        glm::vec3 cLo(INFINITY), cHi(-INFINITY);
        node.lo = glm::vec3(INFINITY);
        node.hi = glm::vec3(-INFINITY);
        for (uint32_t i = node.first; i < node.first + node.count; ++i) {
            const uint32_t t = m_order[i];
            node.lo = glm::min(node.lo, boxLo[t]);
            node.hi = glm::max(node.hi, boxHi[t]);
            cLo = glm::min(cLo, centroids[t]);
            cHi = glm::max(cHi, centroids[t]);
        }
        if (node.count <= kLeafSize) continue;

        // Binned SAH over the three axes.
        float bestCost = INFINITY;
        int bestAxis = -1;
        uint32_t bestSplit = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const float extent = cHi[axis] - cLo[axis];
            if (extent <= 1e-9f) continue;
            struct Bin {
                glm::vec3 lo{INFINITY}, hi{-INFINITY};
                uint32_t count = 0;
            } bins[kBins];
            for (uint32_t i = node.first; i < node.first + node.count; ++i) {
                const uint32_t t = m_order[i];
                const auto b = std::min(kBins - 1, uint32_t((centroids[t][axis] - cLo[axis]) / extent * float(kBins)));
                bins[b].count++;
                bins[b].lo = glm::min(bins[b].lo, boxLo[t]);
                bins[b].hi = glm::max(bins[b].hi, boxHi[t]);
            }
            float leftArea[kBins - 1], rightArea[kBins - 1];
            uint32_t leftCount[kBins - 1], rightCount[kBins - 1];
            glm::vec3 lo(INFINITY), hi(-INFINITY);
            uint32_t sum = 0;
            for (uint32_t i = 0; i < kBins - 1; ++i) {
                sum += bins[i].count;
                lo = glm::min(lo, bins[i].lo);
                hi = glm::max(hi, bins[i].hi);
                leftCount[i] = sum;
                leftArea[i] = sum ? surfaceArea(lo, hi) : 0.0f;
            }
            lo = glm::vec3(INFINITY);
            hi = glm::vec3(-INFINITY);
            sum = 0;
            for (uint32_t i = kBins - 1; i > 0; --i) {
                sum += bins[i].count;
                lo = glm::min(lo, bins[i].lo);
                hi = glm::max(hi, bins[i].hi);
                rightCount[i - 1] = sum;
                rightArea[i - 1] = sum ? surfaceArea(lo, hi) : 0.0f;
            }
            for (uint32_t i = 0; i < kBins - 1; ++i) {
                if (leftCount[i] == 0 || rightCount[i] == 0) continue;
                const float cost = float(leftCount[i]) * leftArea[i] + float(rightCount[i]) * rightArea[i];
                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestSplit = i;
                }
            }
        }
        if (bestAxis < 0 || bestCost >= float(node.count) * surfaceArea(node.lo, node.hi)) continue; // a leaf is cheaper

        const float extent = cHi[bestAxis] - cLo[bestAxis];
        uint32_t* begin = m_order.data() + node.first;
        uint32_t* mid = std::partition(begin, begin + node.count, [&](uint32_t t) {
            return std::min(kBins - 1, uint32_t((centroids[t][bestAxis] - cLo[bestAxis]) / extent * float(kBins))) <= bestSplit;
        });
        const auto leftCount = uint32_t(mid - begin);
        if (leftCount == 0 || leftCount == node.count) continue;
        const auto left = uint32_t(m_nodes.size());
        const uint32_t first = node.first, total = node.count;
        node.first = left;
        node.count = 0; // `node` may dangle after the pushes below
        Node l, r;
        l.first = first;
        l.count = leftCount;
        r.first = first + leftCount;
        r.count = total - leftCount;
        m_nodes.push_back(l);
        m_nodes.push_back(r);
        pending.push_back(left);
        pending.push_back(left + 1);
    }
}

template <bool AnyHit>
bool RayScene::traverse(const Ray& ray, RayHit* hit, HitFilter filter, const void* context) const {
    if (m_nodes.empty()) return false;
    glm::vec3 inv;
    for (int a = 0; a < 3; ++a) inv[a] = std::fabs(ray.direction[a]) > 1e-30f ? 1.0f / ray.direction[a] : std::copysign(1e30f, ray.direction[a]);
    float best = ray.tMax;
    bool found = false;
    struct Entry {
        uint32_t node;
        float t;
    } stack[128];
    int depth = 0;
    if (slab(m_nodes[0].lo, m_nodes[0].hi, ray.origin, inv, best) == INFINITY) return false;
    uint32_t node = 0;
    while (true) {
        const Node& n = m_nodes[node];
        if (n.count > 0) {
            for (uint32_t i = n.first; i < n.first + n.count; ++i) {
                const uint32_t index = m_order[i];
                const Triangle& tri = m_triangles[index];
                const glm::vec3 p = glm::cross(ray.direction, tri.e2);
                const float det = glm::dot(tri.e1, p);
                if (std::fabs(det) < 1e-14f) continue;
                const float invDet = 1.0f / det;
                const glm::vec3 s = ray.origin - tri.v0;
                const float u = glm::dot(s, p) * invDet;
                if (u < 0.0f || u > 1.0f) continue;
                const glm::vec3 q = glm::cross(s, tri.e1);
                const float v = glm::dot(ray.direction, q) * invDet;
                if (v < 0.0f || u + v > 1.0f) continue;
                const float t = glm::dot(tri.e2, q) * invDet;
                if (t <= kMinT || t >= best) continue;
                if (filter && !filter(context, index, u, v)) continue;
                if constexpr (AnyHit) return true;
                best = t;
                found = true;
                hit->t = t;
                hit->triangle = index;
                hit->u = u;
                hit->v = v;
                hit->backFace = det < 0.0f;
            }
        } else {
            uint32_t a = n.first, b = n.first + 1;
            float ta = slab(m_nodes[a].lo, m_nodes[a].hi, ray.origin, inv, best);
            float tb = slab(m_nodes[b].lo, m_nodes[b].hi, ray.origin, inv, best);
            if (tb < ta) {
                std::swap(a, b);
                std::swap(ta, tb);
            }
            if (ta != INFINITY) {
                if (tb != INFINITY && depth < 128) stack[depth++] = {b, tb};
                node = a;
                continue;
            }
        }
        // Pop the next subtree that can still hold a closer hit.
        bool next = false;
        while (depth > 0) {
            const Entry e = stack[--depth];
            if (e.t < best) {
                node = e.node;
                next = true;
                break;
            }
        }
        if (!next) break;
    }
    return found;
}

bool RayScene::intersect(const Ray& ray, RayHit& hit, HitFilter filter, const void* context) const {
    return traverse<false>(ray, &hit, filter, context);
}

bool RayScene::occluded(const Ray& ray, HitFilter filter, const void* context) const {
    return traverse<true>(ray, nullptr, filter, context);
}

} // namespace tools
