#include "LightmapUv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <unordered_map>

namespace tools {
namespace {

constexpr float kChartCos = 0.8660254f; // 30°
constexpr uint32_t kMaxLayout = 4096;   // uint16 sizes; also keeps one mesh within one lightmap page at sane densities

struct Chart {
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec3 u{1.0f, 0.0f, 0.0f}, v{0.0f, 0.0f, 1.0f};
    glm::vec2 lo{INFINITY}, hi{-INFINITY}; // projected bounds, meters
    uint32_t width = 0, height = 0;        // texels, padding included
    uint32_t x = 0, y = 0;                 // placement in the layout
};

// Positions welded on a 0.1 mm grid: raw meshes split vertices at UV0 and normal seams, but charts follow geometry.
// The full 64-bit grid coordinates are the key, so meshes of any size never weld distant positions together.
std::array<int64_t, 3> positionKey(const glm::vec3& p) {
    return {std::llround(double(p.x) * 1e4), std::llround(double(p.y) * 1e4), std::llround(double(p.z) * 1e4)};
}

} // namespace

void generateLightmapUvs(RawMesh& mesh, uint16_t& outWidth, uint16_t& outHeight) {
    const size_t triangles = mesh.indices.size() / 3;
    std::vector<glm::vec3> normal(triangles);
    std::vector<float> area(triangles);
    for (size_t t = 0; t < triangles; ++t) {
        const glm::vec3& a = mesh.vertices[mesh.indices[t * 3]].position;
        const glm::vec3& b = mesh.vertices[mesh.indices[t * 3 + 1]].position;
        const glm::vec3& c = mesh.vertices[mesh.indices[t * 3 + 2]].position;
        const glm::vec3 n = glm::cross(b - a, c - a);
        const float len = glm::length(n);
        area[t] = len * 0.5f;
        normal[t] = len > 1e-12f ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // Edge adjacency on welded positions.
    std::map<std::array<int64_t, 3>, uint32_t> positionIds;
    std::vector<uint32_t> positionOf(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i)
        positionOf[i] = positionIds.emplace(positionKey(mesh.vertices[i].position), uint32_t(positionIds.size())).first->second;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edgeTriangles;
    for (size_t t = 0; t < triangles; ++t)
        for (int e = 0; e < 3; ++e) {
            uint32_t a = positionOf[mesh.indices[t * 3 + e]], b = positionOf[mesh.indices[t * 3 + (e + 1) % 3]];
            if (a > b) std::swap(a, b);
            if (a != b) edgeTriangles[{a, b}].push_back(uint32_t(t));
        }
    std::vector<std::vector<uint32_t>> neighbors(triangles);
    for (const auto& [edge, list] : edgeTriangles)
        for (uint32_t t : list)
            for (uint32_t o : list)
                if (o != t) neighbors[t].push_back(o);

    // Flood fill from the largest unassigned triangle.
    std::vector<int32_t> chartOf(triangles, -1);
    std::vector<Chart> charts;
    std::vector<uint32_t> order(triangles);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return area[a] > area[b]; });
    std::vector<uint32_t> queue;
    for (uint32_t seed : order) {
        if (chartOf[seed] >= 0) continue;
        const auto id = int32_t(charts.size());
        Chart chart;
        chart.normal = normal[seed];
        charts.push_back(chart);
        chartOf[seed] = id;
        if (area[seed] <= 1e-12f) continue; // a degenerate triangle gets a tiny chart of its own
        queue.assign(1, seed);
        while (!queue.empty()) {
            const uint32_t t = queue.back();
            queue.pop_back();
            for (uint32_t n : neighbors[t])
                if (chartOf[n] < 0 && area[n] > 1e-12f && glm::dot(normal[n], chart.normal) >= kChartCos) {
                    chartOf[n] = id;
                    queue.push_back(n);
                }
        }
    }

    // Projection basis and bounds per chart.
    for (Chart& c : charts) {
        const glm::vec3 up = std::abs(c.normal.y) > 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
        c.u = glm::normalize(glm::cross(up, c.normal));
        c.v = glm::cross(c.normal, c.u);
    }
    for (size_t t = 0; t < triangles; ++t) {
        Chart& c = charts[size_t(chartOf[t])];
        for (int k = 0; k < 3; ++k) {
            const glm::vec3& p = mesh.vertices[mesh.indices[t * 3 + k]].position;
            const glm::vec2 q(glm::dot(p, c.u), glm::dot(p, c.v));
            c.lo = glm::min(c.lo, q);
            c.hi = glm::max(c.hi, q);
        }
    }

    // Shelf packing at the reference density; halve the density if a huge mesh would exceed the layout limit.
    float density = kLightmapReferenceDensity;
    uint32_t layoutWidth = 0, layoutHeight = 0;
    for (int attempt = 0; attempt < 8; ++attempt) {
        uint64_t total = 0;
        uint32_t widest = 0;
        for (Chart& c : charts) {
            c.width = uint32_t(std::ceil((c.hi.x - c.lo.x) * density)) + 2;
            c.height = uint32_t(std::ceil((c.hi.y - c.lo.y) * density)) + 2;
            total += uint64_t(c.width) * c.height;
            widest = std::max(widest, c.width);
        }
        layoutWidth = std::max(widest, uint32_t(std::ceil(std::sqrt(double(total)) * 1.15)));
        std::vector<uint32_t> byHeight(charts.size());
        std::iota(byHeight.begin(), byHeight.end(), 0u);
        std::stable_sort(byHeight.begin(), byHeight.end(), [&](uint32_t a, uint32_t b) { return charts[a].height > charts[b].height; });
        uint32_t x = 0, y = 0, rowHeight = 0;
        for (uint32_t i : byHeight) {
            Chart& c = charts[i];
            if (x + c.width > layoutWidth) {
                y += rowHeight;
                x = 0;
                rowHeight = 0;
            }
            c.x = x;
            c.y = y;
            x += c.width;
            rowHeight = std::max(rowHeight, c.height);
        }
        layoutHeight = y + rowHeight;
        if (layoutWidth <= kMaxLayout && layoutHeight <= kMaxLayout) break;
        density *= 0.5f;
    }

    // New vertices per (chart, source vertex).
    std::unordered_map<uint64_t, uint32_t> remap;
    std::vector<RawVertex> vertices;
    vertices.reserve(mesh.vertices.size() + mesh.vertices.size() / 4);
    for (size_t t = 0; t < triangles; ++t) {
        const auto chartId = uint32_t(chartOf[t]);
        const Chart& c = charts[chartId];
        for (int k = 0; k < 3; ++k) {
            uint32_t& index = mesh.indices[t * 3 + k];
            const uint64_t key = uint64_t(chartId) << 32 | index;
            auto [it, inserted] = remap.emplace(key, uint32_t(vertices.size()));
            if (inserted) {
                RawVertex v = mesh.vertices[index];
                const glm::vec2 q(glm::dot(v.position, c.u), glm::dot(v.position, c.v));
                const glm::vec2 texel = glm::vec2(float(c.x + 1), float(c.y + 1)) + (q - c.lo) * density;
                v.uv1 = texel / glm::vec2(float(layoutWidth), float(layoutHeight));
                vertices.push_back(v);
            }
            index = it->second;
        }
    }
    mesh.vertices = std::move(vertices);
    mesh.hasUv1 = true;
    // Sizes are recorded at the reference density, as if the layout had been made there.
    const float scale = kLightmapReferenceDensity / density;
    outWidth = uint16_t(std::min<uint32_t>(kMaxLayout, uint32_t(std::ceil(float(layoutWidth) * scale))));
    outHeight = uint16_t(std::min<uint32_t>(kMaxLayout, uint32_t(std::ceil(float(layoutHeight) * scale))));
}

void lightmapSizeFromArea(const RawMesh& mesh, uint16_t& width, uint16_t& height) {
    double total = 0.0;
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        const glm::vec3& a = mesh.vertices[mesh.indices[t]].position;
        const glm::vec3& b = mesh.vertices[mesh.indices[t + 1]].position;
        const glm::vec3& c = mesh.vertices[mesh.indices[t + 2]].position;
        total += 0.5 * double(glm::length(glm::cross(b - a, c - a)));
    }
    const double side = std::clamp(std::ceil(std::sqrt(total) * kLightmapReferenceDensity * 1.25), 8.0, 2048.0);
    width = height = uint16_t(side);
}

} // namespace tools
