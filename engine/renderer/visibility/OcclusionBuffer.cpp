#include "renderer/visibility/OcclusionBuffer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace renderer {
namespace {

constexpr uint32_t kMaxSubmitted = 4096; // occluders considered per frame (prepare keeps the largest kMaxOccluders)
constexpr float kUncovered = FLT_MAX;

// Box faces as corner indices (bit 0 = x, bit 1 = y, bit 2 = z) and one more corner across the box, for the outward
// direction: −x, +x, −y, +y, −z, +z.
constexpr int kFaceCorners[6][4] = {{0, 2, 4, 6}, {1, 3, 5, 7}, {0, 1, 4, 5}, {2, 3, 6, 7}, {0, 1, 2, 3}, {4, 5, 6, 7}};
constexpr int kEdges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

float cross2(const glm::vec2& o, const glm::vec2& a, const glm::vec2& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

// Andrew's monotone chain; returns the hull counter-clockwise in (X, Y) (collinear points dropped).
uint32_t convexHull(glm::vec2* points, uint32_t count, glm::vec2* hull) {
    std::sort(points, points + count, [](const glm::vec2& a, const glm::vec2& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
    uint32_t k = 0;
    for (uint32_t i = 0; i < count; ++i) {
        while (k >= 2 && cross2(hull[k - 2], hull[k - 1], points[i]) <= 0.0f) --k;
        hull[k++] = points[i];
    }
    for (uint32_t i = count - 1, lower = k + 1; i-- > 0;) {
        while (k >= lower && cross2(hull[k - 2], hull[k - 1], points[i]) <= 0.0f) --k;
        hull[k++] = points[i];
    }
    return k > 1 ? k - 1 : k; // the last point repeats the first
}

} // namespace

OcclusionBuffer::OcclusionBuffer() {
    m_occluders.reserve(kMaxSubmitted);
    m_depth.assign(size_t(kWidth) * kHeight, kUncovered);
    m_tileMax.assign(size_t(kTilesX) * kTilesY, kUncovered);
}

void OcclusionBuffer::begin(const glm::mat4& viewProjection, const glm::vec3& eye) {
    m_viewProjection = viewProjection;
    m_toClipPlanes = glm::transpose(glm::inverse(glm::dmat4(viewProjection))); // once per frame, in double
    m_eye = eye;
    // A perspective has clip z = A·w − B (z/w = A − B/w, w = view distance), so its z row is A × its w row minus B in the
    // translation: A, B and the near plane (z = 0 at w = B/A) follow from the matrix.
    const glm::dvec4 zRow(viewProjection[0][2], viewProjection[1][2], viewProjection[2][2], viewProjection[3][2]);
    const glm::dvec4 wRow(viewProjection[0][3], viewProjection[1][3], viewProjection[2][3], viewProjection[3][3]);
    int k = 0;
    for (int i = 1; i < 3; ++i)
        if (std::abs(wRow[i]) > std::abs(wRow[k])) k = i;
    m_depthA = std::abs(wRow[k]) > 1e-12 ? zRow[k] / wRow[k] : 0.0;
    m_depthB = m_depthA * wRow.w - zRow.w;
    m_zNear = m_depthA > 0.0 ? m_depthB / m_depthA : 0.0;
    m_occluders.clear();
    m_stats = Stats{};
}

void OcclusionBuffer::addOccluder(const glm::vec3 corners[8]) {
    ++m_stats.submitted;
    if (m_occluders.size() >= kMaxSubmitted) {
        ++m_stats.overBudget;
        return;
    }
    Prepared p;
    if (setupOccluder(corners, p)) m_occluders.push_back(p);
}

bool OcclusionBuffer::setupOccluder(const glm::vec3 corners[8], Prepared& out) {
    // Outward face planes in world space, in double (their clip-space depth planes feed the occlusion test); an eye
    // inside every one of them is inside the box.
    glm::dvec3 center(0.0);
    for (int k = 0; k < 8; ++k) center += glm::dvec3(corners[k]);
    center *= 0.125;
    glm::dvec4 facePlanes[6];
    bool front[6];
    bool inside = true;
    for (int f = 0; f < 6; ++f) {
        const glm::dvec3 a(corners[kFaceCorners[f][0]]);
        glm::dvec3 n = glm::cross(glm::dvec3(corners[kFaceCorners[f][1]]) - a, glm::dvec3(corners[kFaceCorners[f][2]]) - a);
        const double length = glm::length(n);
        if (!(length > 1e-12)) { // a flat box isn't a solid occluder
            ++m_stats.offscreen;
            return false;
        }
        n /= length;
        if (glm::dot(n, a - center) < 0.0) n = -n;
        facePlanes[f] = glm::dvec4(n, -glm::dot(n, a));
        front[f] = glm::dot(n, glm::dvec3(m_eye)) + facePlanes[f].w > 0.0;
        inside = inside && !front[f];
    }
    Stats& stats = m_stats;
    if (inside) {
        ++stats.offscreen;
        return false;
    }

    // Clip space; trivially outside when every corner lies beyond the same plane.
    glm::vec4 h[8];
    uint32_t outsideAll = 0x3F;
    for (int k = 0; k < 8; ++k) {
        h[k] = m_viewProjection * glm::vec4(corners[k], 1.0f);
        uint32_t bits = 0;
        if (h[k].x < -h[k].w) bits |= 1;
        if (h[k].x > h[k].w) bits |= 2;
        if (h[k].y < -h[k].w) bits |= 4;
        if (h[k].y > h[k].w) bits |= 8;
        if (h[k].z < 0.0f) bits |= 16;
        if (h[k].z > h[k].w) bits |= 32;
        outsideAll &= bits;
    }
    if (outsideAll) {
        ++stats.offscreen;
        return false;
    }

    // The near-clipped box: corners in front of the near plane plus the points where edges cross it. All have w > 0.
    glm::vec4 clipped[20];
    uint32_t n = 0;
    for (int k = 0; k < 8; ++k)
        if (h[k].z >= 0.0f) clipped[n++] = h[k];
    for (const auto& e : kEdges) {
        const glm::vec4& a = h[e[0]];
        const glm::vec4& b = h[e[1]];
        if ((a.z >= 0.0f) != (b.z >= 0.0f)) {
            const float t = a.z / (a.z - b.z);
            glm::vec4 p = a + (b - a) * t;
            p.z = 0.0f;
            clipped[n++] = p;
        }
    }
    glm::vec2 screen[20], hull[41]; // monotone chain: up to 2n − 1 entries
    float depthCap = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        const float invW = 1.0f / clipped[i].w;
        screen[i] = glm::vec2((clipped[i].x * invW * 0.5f + 0.5f) * float(kWidth), (0.5f - clipped[i].y * invW * 0.5f) * float(kHeight));
        depthCap = std::max(depthCap, clipped[i].z * invW);
    }
    const uint32_t hullCount = n >= 3 ? convexHull(screen, n, hull) : 0;
    float area = 0.0f;
    glm::vec2 lo(FLT_MAX), hi(-FLT_MAX);
    for (uint32_t i = 0; i < hullCount; ++i) {
        const glm::vec2& p = hull[i];
        const glm::vec2& q = hull[(i + 1) % hullCount];
        area += p.x * q.y - q.x * p.y;
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
    }
    area *= 0.5f; // counter-clockwise in (X, Y): positive
    if (hullCount < 3 || !(area >= kMinArea)) {
        ++stats.tooSmall;
        return false;
    }
    out.area = area;
    out.depthCap = depthCap;
    out.x0 = int(std::max(0.0f, std::floor(lo.x)));
    out.x1 = int(std::min(float(kWidth - 1), std::ceil(hi.x)));
    out.y0 = int(std::max(0.0f, std::floor(lo.y)));
    out.y1 = int(std::min(float(kHeight - 1), std::ceil(hi.y)));
    if (out.x0 > out.x1 || out.y0 > out.y1) {
        ++stats.offscreen;
        return false;
    }

    // Inner coverage: the whole pixel [X, X+1] × [Y, Y+1] lies left of every edge: E(center) ≥ ½(|a| + |b|).
    out.edgeCount = 0;
    for (uint32_t i = 0; i < hullCount && out.edgeCount < 16; ++i) {
        const glm::vec2& p = hull[i];
        const glm::vec2& q = hull[(i + 1) % hullCount];
        const float a = -(q.y - p.y), b = q.x - p.x;
        const float c = -(a * p.x + b * p.y);
        const float slack = 0.5f * (std::abs(a) + std::abs(b));
        out.edgeA[out.edgeCount] = a;
        out.edgeB[out.edgeCount] = b;
        out.edgeK[out.edgeCount] = c + 0.5f * a + 0.5f * b - slack * 1.002f; // at integer (X, Y) = pixel centers
        ++out.edgeCount;
    }

    // Depth: the entry point of a convex body is on its farthest front-face plane. Each plane, written in pixel
    // coordinates through the inverse-transpose of the view-projection, plus half a pixel of its slope bounds the
    // depth anywhere inside the pixel. A face seen almost edge-on would need huge slopes: then the box's farthest
    // point serves as a constant (still conservative) depth.
    out.planeCount = 0;
    bool constant = false;
    for (int f = 0; f < 6 && !constant; ++f) {
        if (!front[f]) continue;
        const glm::dvec4 P = m_toClipPlanes * facePlanes[f];
        if (!(std::abs(P.z) > 1e-30)) {
            constant = true;
            break;
        }
        const double alpha = -2.0 * P.x / (double(kWidth) * P.z);
        const double beta = 2.0 * P.y / (double(kHeight) * P.z);
        const double gamma = (P.x - P.y - P.w) / P.z;
        if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(gamma) || std::abs(alpha) + std::abs(beta) > 0.5) {
            constant = true;
            break;
        }
        out.planeA[out.planeCount] = float(alpha);
        out.planeB[out.planeCount] = float(beta);
        out.planeK[out.planeCount] = float(gamma + 0.5 * alpha + 0.5 * beta + 0.5 * (std::abs(alpha) + std::abs(beta)));
        ++out.planeCount;
    }
    if (constant || out.planeCount == 0) {
        out.planeA[0] = out.planeB[0] = 0.0f;
        out.planeK[0] = depthCap;
        out.planeCount = 1;
    }
    return true;
}

void OcclusionBuffer::prepare() {
    if (m_occluders.size() > kMaxOccluders) {
        std::nth_element(m_occluders.begin(), m_occluders.begin() + kMaxOccluders, m_occluders.end(),
                         [](const Prepared& a, const Prepared& b) { return a.area > b.area; });
        m_stats.overBudget += uint32_t(m_occluders.size()) - kMaxOccluders;
        m_occluders.resize(kMaxOccluders);
    }
    m_stats.drawn = uint32_t(m_occluders.size());
}

void OcclusionBuffer::rasterizeBand(uint32_t band) {
    if (band >= kBands) return;
    const int rowBegin = int(band * kTile), rowEnd = rowBegin + int(kTile);
    std::fill(m_depth.begin() + ptrdiff_t(rowBegin) * kWidth, m_depth.begin() + ptrdiff_t(rowEnd) * kWidth, kUncovered);
    for (const Prepared& o : m_occluders) {
        const int y0 = std::max(o.y0, rowBegin), y1 = std::min(o.y1, rowEnd - 1);
        for (int y = y0; y <= y1; ++y) {
            // The covered span of this row: every edge constrains X from one side.
            float left = float(o.x0), right = float(o.x1);
            bool empty = false;
            for (uint32_t e = 0; e < o.edgeCount && !empty; ++e) {
                const float a = o.edgeA[e];
                const float rest = o.edgeB[e] * float(y) + o.edgeK[e];
                if (a > 0.0f) left = std::max(left, -rest / a);
                else if (a < 0.0f) right = std::min(right, -rest / a);
                else empty = rest < 0.0f;
            }
            if (empty || !(left <= right)) continue;
            const int xl = int(std::ceil(left)), xr = int(std::floor(right));
            float* row = m_depth.data() + size_t(y) * kWidth;
            for (int x = xl; x <= xr; ++x) {
                float z = 0.0f;
                for (uint32_t p = 0; p < o.planeCount; ++p) z = std::max(z, o.planeA[p] * float(x) + o.planeB[p] * float(y) + o.planeK[p]);
                z = std::min(z, o.depthCap);
                row[x] = std::min(row[x], z);
            }
        }
    }
    for (uint32_t tx = 0; tx < kTilesX; ++tx) {
        float tileMax = 0.0f;
        for (int y = rowBegin; y < rowEnd; ++y) {
            const float* row = m_depth.data() + size_t(y) * kWidth + tx * kTile;
            for (uint32_t x = 0; x < kTile; ++x) tileMax = std::max(tileMax, row[x]);
        }
        m_tileMax[band * kTilesX + tx] = tileMax;
    }
}

void OcclusionBuffer::finishStats() {
    uint32_t covered = 0;
    for (float t : m_tileMax) covered += t < kUncovered ? 1u : 0u;
    m_stats.coveredTiles = covered;
}

bool OcclusionBuffer::boxOccluded(const glm::vec3& boxMin, const glm::vec3& boxMax) const {
    if (m_occluders.empty() || !(m_depthA > 0.0)) return false;
    float x0 = FLT_MAX, x1 = -FLT_MAX, y0 = FLT_MAX, y1 = -FLT_MAX, closest = FLT_MAX;
    for (int k = 0; k < 8; ++k) {
        const glm::vec3 corner((k & 1) ? boxMax.x : boxMin.x, (k & 2) ? boxMax.y : boxMin.y, (k & 4) ? boxMax.z : boxMin.z);
        const glm::vec4 h = m_viewProjection * glm::vec4(corner, 1.0f);
        if (!(h.z >= 0.0f) || !(h.w > 0.0f)) return false; // reaches the near plane: never occluded
        const float invW = 1.0f / h.w;
        const float x = (h.x * invW * 0.5f + 0.5f) * float(kWidth);
        const float y = (0.5f - h.y * invW * 0.5f) * float(kHeight);
        x0 = std::min(x0, x), x1 = std::max(x1, x);
        y0 = std::min(y0, y), y1 = std::max(y1, y);
        closest = std::min(closest, h.w); // clip w is the view distance; a box's nearest point is one of its corners
    }
    // The occluders must be nearer than the object by a margin that grows with distance (float rounding of depth grows
    // with the square of it): compare against the depth of a point that much nearer than the object's closest corner.
    const double shifted = double(closest) - (double(kMarginMeters) + double(kMarginRelative) * double(closest));
    if (!(shifted > m_zNear)) return false;
    const float nearest = float(m_depthA - m_depthB / shifted);
    if (x1 < 0.0f || y1 < 0.0f || x0 >= float(kWidth) || y0 >= float(kHeight)) return false; // off screen: the frustum decides
    const int px0 = int(std::max(0.0f, std::floor(x0))), px1 = int(std::min(float(kWidth - 1), std::floor(x1)));
    const int py0 = int(std::max(0.0f, std::floor(y0))), py1 = int(std::min(float(kHeight - 1), std::floor(y1)));
    for (int ty = py0 / int(kTile); ty <= py1 / int(kTile); ++ty)
        for (int tx = px0 / int(kTile); tx <= px1 / int(kTile); ++tx) {
            if (m_tileMax[size_t(ty) * kTilesX + size_t(tx)] < nearest) continue;
            // A tile that is partly uncovered or deeper somewhere: its pixels under the rectangle decide (objects close
            // to an occluder's edge).
            const int yBegin = std::max(py0, ty * int(kTile)), yEnd = std::min(py1, ty * int(kTile) + int(kTile) - 1);
            const int xBegin = std::max(px0, tx * int(kTile)), xEnd = std::min(px1, tx * int(kTile) + int(kTile) - 1);
            for (int y = yBegin; y <= yEnd; ++y) {
                const float* row = m_depth.data() + size_t(y) * kWidth;
                for (int x = xBegin; x <= xEnd; ++x)
                    if (!(row[x] < nearest)) return false;
            }
        }
    return true;
}

} // namespace renderer
