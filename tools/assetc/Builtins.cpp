#include "Builtins.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <vector>

namespace tools {
namespace {

constexpr float kPi = 3.14159265358979f;

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, sep)) parts.push_back(part);
    return parts;
}

bool parseColor(const std::string& hex, uint8_t rgb[3]) {
    if (hex.size() != 6) return false;
    char* end = nullptr;
    const unsigned long v = std::strtoul(hex.c_str(), &end, 16);
    if (*end != '\0') return false;
    rgb[0] = uint8_t(v >> 16), rgb[1] = uint8_t(v >> 8), rgb[2] = uint8_t(v);
    return true;
}

// ---------------------------------------------------------------- meshes

// One quad with normal n = u × v, counter-clockwise seen from the side n points to. UVs follow glTF
// (v = 0 at the top), so a texture appears upright.
void addQuad(RawMesh& m, glm::vec3 center, glm::vec3 u, glm::vec3 v, glm::vec2 uvScale) {
    const glm::vec3 n = glm::normalize(glm::cross(u, v));
    const auto base = uint32_t(m.vertices.size());
    const glm::vec3 corners[4] = {center - u * 0.5f - v * 0.5f, center + u * 0.5f - v * 0.5f, center + u * 0.5f + v * 0.5f,
                                  center - u * 0.5f + v * 0.5f};
    const glm::vec2 uvs[4] = {{0, uvScale.y}, {uvScale.x, uvScale.y}, {uvScale.x, 0}, {0, 0}};
    for (int i = 0; i < 4; ++i) {
        RawVertex vtx;
        vtx.position = corners[i];
        vtx.normal = n;
        vtx.uv0 = uvs[i];
        m.vertices.push_back(vtx);
    }
    for (uint32_t i : {0u, 1u, 2u, 0u, 2u, 3u}) m.indices.push_back(base + i);
}

// Box faces in the order +X, −X, +Y, −Y, +Z, −Z; `sides` and `caps` (±Y) pick which ones. With tile > 0 one texture
// repeat covers `tile` meters; otherwise each face maps the whole texture once.
void addBoxFaces(RawMesh& m, glm::vec3 size, float tile, bool sides, bool caps) {
    struct Face { glm::vec3 n, u, v; };
    const Face faces[6] = {{{1, 0, 0}, {0, 0, -1}, {0, 1, 0}}, {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
                           {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}}, {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
                           {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},  {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}}};
    for (int i = 0; i < 6; ++i) {
        const bool cap = i == 2 || i == 3;
        if (!(cap ? caps : sides)) continue;
        const glm::vec3 u = faces[i].u * size, v = faces[i].v * size;
        const glm::vec2 repeats = tile > 0.0f ? glm::vec2(glm::length(u), glm::length(v)) / tile : glm::vec2(1.0f);
        addQuad(m, faces[i].n * size * 0.5f, u, v, repeats);
    }
}

void makeSphere(RawMesh& m, uint32_t rings, uint32_t segments) {
    for (uint32_t i = 0; i <= rings; ++i) {
        const float theta = kPi * float(i) / float(rings); // 0 at the top
        for (uint32_t j = 0; j <= segments; ++j) {
            const float phi = 2.0f * kPi * float(j) / float(segments);
            RawVertex v;
            v.normal = glm::vec3(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));
            v.position = v.normal * 0.5f;
            v.uv0 = glm::vec2(float(j) / float(segments), float(i) / float(rings));
            m.vertices.push_back(v);
        }
    }
    for (uint32_t i = 0; i < rings; ++i)
        for (uint32_t j = 0; j < segments; ++j) {
            const uint32_t a = i * (segments + 1) + j, d = a + 1, b = a + segments + 1, c = b + 1;
            for (uint32_t idx : {a, d, c, a, c, b}) m.indices.push_back(idx); // CCW seen from outside
        }
}

void makeCylinder(RawMesh& m, uint32_t segments) {
    for (uint32_t j = 0; j <= segments; ++j) { // side: top row then bottom row per column
        const float phi = 2.0f * kPi * float(j) / float(segments);
        const glm::vec3 n(std::cos(phi), 0.0f, std::sin(phi));
        for (float y : {0.5f, -0.5f}) {
            RawVertex v;
            v.position = glm::vec3(n.x * 0.5f, y, n.z * 0.5f);
            v.normal = n;
            v.uv0 = glm::vec2(float(j) / float(segments), y > 0.0f ? 0.0f : 1.0f);
            m.vertices.push_back(v);
        }
    }
    for (uint32_t j = 0; j < segments; ++j) {
        const uint32_t a = j * 2, b = a + 1, d = a + 2, c = a + 3;
        for (uint32_t idx : {a, d, c, a, c, b}) m.indices.push_back(idx);
    }
    for (float y : {0.5f, -0.5f}) { // caps
        const auto center = uint32_t(m.vertices.size());
        RawVertex cv;
        cv.position = glm::vec3(0, y, 0);
        cv.normal = glm::vec3(0, y > 0.0f ? 1.0f : -1.0f, 0);
        cv.uv0 = glm::vec2(0.5f);
        m.vertices.push_back(cv);
        for (uint32_t j = 0; j <= segments; ++j) {
            const float phi = 2.0f * kPi * float(j) / float(segments);
            RawVertex v = cv;
            v.position = glm::vec3(std::cos(phi) * 0.5f, y, std::sin(phi) * 0.5f);
            v.uv0 = glm::vec2(0.5f + std::cos(phi) * 0.5f, 0.5f + std::sin(phi) * 0.5f);
            m.vertices.push_back(v);
        }
        for (uint32_t j = 0; j < segments; ++j) {
            const uint32_t p0 = center + 1 + j, p1 = p0 + 1;
            if (y > 0.0f) for (uint32_t idx : {center, p1, p0}) m.indices.push_back(idx); // faces +Y
            else for (uint32_t idx : {center, p0, p1}) m.indices.push_back(idx);          // faces −Y
        }
    }
}

// Appends `part` transformed by `m` (normals by its inverse transpose, uv0 scaled) as material `slot`; consecutive
// parts with the same slot share one submesh.
void appendPart(RawMesh& mesh, const RawMesh& part, const glm::mat4& m, glm::vec2 uvScale, uint32_t slot) {
    const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(m)));
    const auto base = uint32_t(mesh.vertices.size());
    const auto first = uint32_t(mesh.indices.size());
    for (RawVertex v : part.vertices) {
        v.position = glm::vec3(m * glm::vec4(v.position, 1.0f));
        v.normal = glm::normalize(normalMatrix * v.normal);
        v.uv0 *= uvScale;
        mesh.vertices.push_back(v);
    }
    for (uint32_t i : part.indices) mesh.indices.push_back(base + i);
    const auto count = uint32_t(part.indices.size());
    if (!mesh.submeshes.empty() && mesh.submeshes.back().materialSlot == slot) mesh.submeshes.back().indexCount += count;
    else mesh.submeshes.push_back({first, count, slot});
}

glm::mat4 placed(glm::vec3 translation, glm::vec3 scale) {
    return glm::translate(glm::mat4(1.0f), translation) * glm::scale(glm::mat4(1.0f), scale);
}

// A unit-diameter sphere pushed in and out by a smooth, seeded pattern (tree crowns, rocks, bushes; M5 plan D12). The
// normals come from the displaced surface itself (central differences on the ring/segment grid, wrapping around), so
// the mesh is smooth-shaded and seam-free apart from the UV seam: what LOD simplification needs.
void makeLumpySphere(RawMesh& m, uint32_t rings, uint32_t segments, float amplitude, float seed) {
    auto radiusAt = [&](const glm::vec3& d) {
        const float n = std::sin(2.1f * d.x + seed) * std::sin(1.7f * d.y + 2.0f * seed) * std::sin(2.6f * d.z + 0.5f * seed) +
                        0.5f * std::sin(4.3f * d.x + 3.1f * d.y + seed) * std::sin(3.7f * d.z - 1.3f * seed) +
                        0.25f * std::sin(7.9f * d.y + 6.1f * d.z + 2.7f * seed);
        return 0.5f * (1.0f + amplitude * n);
    };
    auto pointAt = [&](int i, int j) {
        i = std::clamp(i, 0, int(rings));
        const float theta = kPi * float(i) / float(rings);
        const float phi = 2.0f * kPi * float(j) / float(segments);
        const glm::vec3 d(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));
        return d * radiusAt(d);
    };
    const auto base = uint32_t(m.vertices.size());
    for (uint32_t i = 0; i <= rings; ++i)
        for (uint32_t j = 0; j <= segments; ++j) {
            RawVertex v;
            v.position = pointAt(int(i), int(j));
            glm::vec3 n;
            if (i == 0 || i == rings) {
                n = glm::vec3(0.0f, i == 0 ? 1.0f : -1.0f, 0.0f); // the poles: straight up and down
            } else {
                const glm::vec3 alongRing = pointAt(int(i), int(j) + 1) - pointAt(int(i), int(j) - 1);
                const glm::vec3 alongMeridian = pointAt(int(i) + 1, int(j)) - pointAt(int(i) - 1, int(j));
                n = glm::normalize(glm::cross(alongRing, alongMeridian));
                if (glm::dot(n, v.position) < 0.0f) n = -n;
            }
            v.normal = n;
            v.uv0 = glm::vec2(float(j) / float(segments) * 2.0f, float(i) / float(rings));
            m.vertices.push_back(v);
        }
    for (uint32_t i = 0; i < rings; ++i)
        for (uint32_t j = 0; j < segments; ++j) {
            const uint32_t a = base + i * (segments + 1) + j, d = a + 1, b = a + segments + 1, c = b + 1;
            for (uint32_t idx : {a, d, c, a, c, b}) m.indices.push_back(idx); // CCW seen from outside
        }
}

void addTriangle(RawMesh& m, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec2 ta, glm::vec2 tb, glm::vec2 tc) {
    const glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
    const auto base = uint32_t(m.vertices.size());
    const glm::vec3 p[3] = {a, b, c};
    const glm::vec2 t[3] = {ta, tb, tc};
    for (int i = 0; i < 3; ++i) {
        RawVertex v;
        v.position = p[i];
        v.normal = n;
        v.uv0 = t[i];
        m.vertices.push_back(v);
        m.indices.push_back(base + uint32_t(i));
    }
}

// A ramp: rises along +Z from the bottom edge at −Z to the full height at +Z; centered like a box.
void makeWedge(RawMesh& m, glm::vec3 size, float tile) {
    const glm::vec3 h = size * 0.5f;
    const float slope = std::hypot(size.y, size.z);
    addQuad(m, glm::vec3(0.0f), glm::vec3(-size.x, 0, 0), glm::vec3(0, size.y, size.z), glm::vec2(size.x, slope) / tile);  // slope
    addQuad(m, glm::vec3(0, 0, h.z), glm::vec3(size.x, 0, 0), glm::vec3(0, size.y, 0), glm::vec2(size.x, size.y) / tile);  // back
    addQuad(m, glm::vec3(0, -h.y, 0), glm::vec3(size.x, 0, 0), glm::vec3(0, 0, size.z), glm::vec2(size.x, size.z) / tile); // bottom
    for (float side : {1.0f, -1.0f}) {
        const glm::vec3 a(side * h.x, -h.y, -h.z), b(side * h.x, -h.y, h.z), c(side * h.x, h.y, h.z);
        const glm::vec2 ta(0, size.y / tile), tb(size.z / tile, size.y / tile), tc(size.z / tile, 0);
        if (side > 0.0f) addTriangle(m, a, c, b, ta, tc, tb);
        else addTriangle(m, a, b, c, ta, tb, tc);
    }
}


// ---------------------------------------------------------------- procedural textures

uint32_t hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float hash01(uint32_t x) {
    return float(hash(x) & 0xFFFFFFu) / float(0xFFFFFFu);
}

// Tiling value noise: `period` lattice cells across the texture.
float lattice(int x, int y, int period, uint32_t seed) {
    x = ((x % period) + period) % period;
    y = ((y % period) + period) % period;
    return hash01(uint32_t(x) * 73856093u ^ uint32_t(y) * 19349663u ^ seed * 83492791u);
}

float valueNoise(float x, float y, int period, uint32_t seed) {
    const float fx = std::floor(x), fy = std::floor(y);
    const int x0 = int(fx), y0 = int(fy);
    float tx = x - fx, ty = y - fy;
    tx = tx * tx * (3.0f - 2.0f * tx);
    ty = ty * ty * (3.0f - 2.0f * ty);
    const float a = lattice(x0, y0, period, seed), b = lattice(x0 + 1, y0, period, seed);
    const float c = lattice(x0, y0 + 1, period, seed), d = lattice(x0 + 1, y0 + 1, period, seed);
    const float top = a + (b - a) * tx, bottom = c + (d - c) * tx;
    return top + (bottom - top) * ty;
}

// Octaves of tiling noise in [0, 1]; u, v run 0–1 across the texture.
float fbm(float u, float v, int period, uint32_t seed, int octaves = 4) {
    float sum = 0.0f, amplitude = 0.5f, norm = 0.0f;
    for (int o = 0; o < octaves; ++o) {
        const int p = period << o;
        sum += amplitude * valueNoise(u * float(p), v * float(p), p, seed + uint32_t(o));
        norm += amplitude;
        amplitude *= 0.5f;
    }
    return sum / norm;
}

float smooth01(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Non-tiling value-noise octaves in world space (period far beyond any map).
float worldNoise(float x, float z, uint32_t seed) {
    float sum = 0.0f, amplitude = 0.5f, norm = 0.0f, f = 1.0f / 16.0f;
    for (int o = 0; o < 4; ++o) {
        sum += amplitude * valueNoise(x * f + 512.0f, z * f + 512.0f, 1 << 20, seed + uint32_t(o));
        norm += amplitude;
        amplitude *= 0.5f;
        f *= 2.0f;
    }
    return sum / norm;
}

// The demo arena's ground height (m): a flat 40 m square plaza at y = 0, hills rising toward the map edge and a pond
// basin around (14, −14). Terrain chunks sample it in world coordinates, so neighbouring chunks meet exactly.
float arenaHeight(float x, float z) {
    const float r = std::max(std::fabs(x), std::fabs(z));
    const float rim = smooth01((r - 20.0f) / 12.0f);
    const float pond = 1.3f * (1.0f - smooth01((std::hypot(x - 14.0f, z + 14.0f) - 3.5f) / 3.0f));
    return rim * (2.0f + 2.6f * worldNoise(x, z, 5)) - pond;
}

// A heightfield chunk of `size` m and cells² quads covering the world square centered on (cx, cz), with positions
// relative to that center (place the instance at (cx, 0, cz)). UV0 = world xz / tile, so textures run on seamlessly.
void makeTerrain(RawMesh& m, float size, uint32_t cells, float tile, float cx, float cz) {
    const float step = size / float(cells), e = step * 0.5f;
    for (uint32_t j = 0; j <= cells; ++j)
        for (uint32_t i = 0; i <= cells; ++i) {
            const float x = cx - size * 0.5f + float(i) * step, z = cz - size * 0.5f + float(j) * step;
            RawVertex v;
            v.position = glm::vec3(x - cx, arenaHeight(x, z), z - cz);
            v.normal = glm::normalize(glm::vec3((arenaHeight(x - e, z) - arenaHeight(x + e, z)) / (2.0f * e), 1.0f,
                                                (arenaHeight(x, z - e) - arenaHeight(x, z + e)) / (2.0f * e)));
            v.uv0 = glm::vec2(x, z) / tile;
            m.vertices.push_back(v);
        }
    for (uint32_t j = 0; j < cells; ++j)
        for (uint32_t i = 0; i < cells; ++i) {
            const uint32_t v00 = j * (cells + 1) + i, v10 = v00 + 1, v01 = v00 + cells + 1, v11 = v01 + 1;
            for (uint32_t idx : {v00, v01, v11, v00, v11, v10}) m.indices.push_back(idx); // counter-clockwise from above
        }
}

// One texel of a procedural material; colors are sRGB 0–1, specular and gloss linear.
struct Surface {
    float height = 0.0f;
    float albedo[3] = {0.5f, 0.5f, 0.5f};
    float alpha = 1.0f;
    float specular = 0.04f;
    float gloss = 0.25f;
    float emission[3] = {0.0f, 0.0f, 0.0f};
};
using SurfaceFn = void (*)(float x, float y, float size, Surface& s);

// Running bond: 4 bricks across, 8 courses, mortar joints; per-brick tone, a few dark bricks, grain.
void bricks(float x, float y, float size, Surface& s) {
    const float bw = size / 4.0f, bh = size / 8.0f;
    const int row = int(y / bh);
    const float xs = x + ((row & 1) ? bw * 0.5f : 0.0f);
    const int rawCol = int(std::floor(xs / bw));
    const float u = xs - float(rawCol) * bw, v = y - float(row) * bh;
    const int col = rawCol & 3; // the brick across the seam keeps one id
    const float edge = std::min(std::min(u, bw - u), std::min(v, bh - v));
    const uint32_t id = uint32_t(row * 16 + col);
    const float n = fbm(x / size, y / size, 16, 7), grain = fbm(x / size, y / size, 64, 11);
    const float face = smooth01((edge - size / 200.0f) / (size / 170.0f)); // 0 in the joint, 1 on the brick face
    s.height = face * (0.8f + 0.2f * n) + (1.0f - face) * 0.2f * grain;
    const float tone = 0.82f + 0.36f * hash01(id * 2654435761u);
    const float dark = hash01(id + 99u) < 0.15f ? 0.62f : 1.0f;
    const float brick[3] = {0.66f * tone * dark, 0.33f * tone * dark, 0.24f * tone * dark};
    const float mortar[3] = {0.68f, 0.66f, 0.62f};
    for (int c = 0; c < 3; ++c) s.albedo[c] = std::clamp((brick[c] * face + mortar[c] * (1.0f - face)) * (0.82f + 0.36f * grain), 0.0f, 1.0f);
    s.specular = 0.02f + 0.05f * face;
    s.gloss = std::clamp((0.12f + 0.3f * face) * (0.7f + 0.6f * n), 0.0f, 1.0f);
}

// Steel floor: 4×4 plates with grooves, corner rivets and a diamond tread; grime lowers specular and gloss.
void plates(float x, float y, float size, Surface& s) {
    const float cell = size / 4.0f;
    const float u = std::fmod(x, cell), v = std::fmod(y, cell);
    const uint32_t id = uint32_t(int(y / cell) * 4 + int(x / cell));
    const float edge = std::min(std::min(u, cell - u), std::min(v, cell - v));
    const float seam = smooth01((edge - size / 400.0f) / (size / 220.0f));
    const float inset = cell * 0.09f, radius = cell * 0.04f;
    float rivet = 0.0f;
    for (float cx : {inset, cell - inset})
        for (float cy : {inset, cell - inset}) {
            const float d = std::hypot(u - cx, v - cy) / radius;
            if (d < 1.0f) rivet = std::max(rivet, std::sqrt(1.0f - d * d));
        }
    const float g = cell / 7.0f; // diamond tread on a 45° grid, alternating orientation
    const float a = (u + v) / g, b = (u - v) / g;
    const float fa = a - std::floor(a) - 0.5f, fb = b - std::floor(b) - 0.5f;
    const bool flip = (int(std::floor(a)) + int(std::floor(b))) & 1;
    const float t = 1.0f - (flip ? std::fabs(fa) / 0.11f + std::fabs(fb) / 0.34f : std::fabs(fb) / 0.11f + std::fabs(fa) / 0.34f);
    const float interior = smooth01((edge - cell * 0.14f) / (cell * 0.02f));
    const float tread = smooth01(t * 3.0f) * interior;
    s.height = 0.45f * seam + 0.22f * tread + 0.55f * rivet;
    const float grime = fbm(x / size, y / size, 8, 23), scratch = fbm(x / size * 4.0f, y / size * 0.25f, 32, 31);
    const float tint = 0.9f + 0.2f * hash01(id * 977u);
    const float steel[3] = {0.42f * tint, 0.45f * tint, 0.48f * tint};
    const float dirt = std::clamp(grime * 1.4f - 0.35f, 0.0f, 1.0f);
    for (int c = 0; c < 3; ++c)
        s.albedo[c] = std::clamp(steel[c] * (0.85f + 0.3f * scratch) * (1.0f - 0.45f * dirt) * (0.35f + 0.65f * seam) + 0.12f * rivet, 0.0f, 1.0f);
    s.specular = (0.12f + 0.5f * (1.0f - dirt)) * (0.15f + 0.85f * seam);
    s.gloss = std::clamp((0.35f + 0.5f * (1.0f - dirt) + 0.1f * scratch) * (0.3f + 0.7f * seam), 0.0f, 1.0f);
}

// Sci-fi wall panel: 2×2 panels, a raised frame around a recessed field, glowing stripes and indicator lights.
void panel(float x, float y, float size, Surface& s) {
    const float cell = size / 2.0f;
    const float u = std::fmod(x, cell), v = std::fmod(y, cell);
    const float edge = std::min(std::min(u, cell - u), std::min(v, cell - v));
    const float rim = smooth01(edge / (cell * 0.012f));
    const float field = smooth01((edge - cell * 0.07f) / (cell * 0.015f));
    s.height = rim * (0.75f - 0.4f * field);
    const float n = fbm(x / size, y / size, 16, 41);
    const float frame[3] = {0.36f, 0.38f, 0.41f}, inner[3] = {0.11f, 0.12f, 0.14f};
    for (int c = 0; c < 3; ++c) s.albedo[c] = (frame[c] * (1.0f - field) + inner[c] * field) * (0.85f + 0.3f * n) * (0.3f + 0.7f * rim);
    s.specular = 0.25f;
    s.gloss = 0.55f;
    float glow = 0.0f;
    for (float stripe : {0.32f, 0.5f, 0.68f}) glow = std::max(glow, 1.0f - smooth01((std::fabs(v - stripe * cell) - cell * 0.012f) / (cell * 0.01f)));
    glow *= smooth01((u - cell * 0.15f) / (cell * 0.01f)) * smooth01((cell * 0.85f - u) / (cell * 0.01f));
    const float cyan[3] = {0.25f, 0.9f, 1.0f}, orange[3] = {1.0f, 0.55f, 0.12f}, green[3] = {0.35f, 1.0f, 0.3f};
    for (int c = 0; c < 3; ++c) s.emission[c] = cyan[c] * glow;
    for (int i = 0; i < 5; ++i) {
        const float d = std::hypot(u - cell * (0.3f + 0.1f * float(i)), v - cell * 0.84f) / (cell * 0.022f);
        const float dot = 1.0f - smooth01((d - 0.7f) / 0.3f);
        for (int c = 0; c < 3; ++c) s.emission[c] = std::max(s.emission[c], ((i & 1) ? green[c] : orange[c]) * dot);
    }
    for (float& e : s.emission) e *= field;
}

// Chain-link fence: two families of diagonal wires in alpha, antialiased over ~1.5 texels.
void fence(float x, float y, float size, Surface& s) {
    const float period = size / 5.0f, halfWidth = size / 140.0f;
    const float d1 = std::fmod(x + y, period), d2 = std::fmod(x - y + size * 4.0f, period);
    const float dist = std::min(std::min(d1, period - d1), std::min(d2, period - d2));
    s.alpha = std::clamp(1.0f - (dist - halfWidth) / 1.5f, 0.0f, 1.0f);
    const float n = fbm(x / size, y / size, 16, 53);
    for (int c = 0; c < 3; ++c) s.albedo[c] = (0.56f + 0.02f * float(c)) * (0.8f + 0.4f * n);
    s.specular = 0.5f;
    s.gloss = 0.5f;
}

// Ground: grass and dirt patches, pebbles in the dirt.
void ground(float x, float y, float size, Surface& s) {
    const float u = x / size, v = y / size;
    const float patch = fbm(u, v, 4, 61), detail = fbm(u, v, 32, 67), fine = fbm(u, v, 128, 71, 2);
    const float grassAmount = smooth01((patch - 0.42f) / 0.16f);
    const float cell = size / 48.0f;
    const int cx = int(x / cell), cy = int(y / cell);
    const uint32_t id = uint32_t((cy % 48) * 48 + (cx % 48));
    const float px = (float(cx) + 0.25f + 0.5f * hash01(id * 3u)) * cell, py = (float(cy) + 0.25f + 0.5f * hash01(id * 3u + 1u)) * cell;
    const float d = std::hypot(x - px, y - py) / (cell * (0.18f + 0.2f * hash01(id * 3u + 2u)));
    const float pebble = d < 1.0f ? std::sqrt(1.0f - d * d) * (1.0f - grassAmount) : 0.0f;
    const float grass[3] = {0.18f, 0.27f, 0.09f}, dirt[3] = {0.35f, 0.28f, 0.19f}, stone[3] = {0.48f, 0.47f, 0.44f};
    for (int c = 0; c < 3; ++c) {
        const float g = grass[c] * (0.7f + 0.6f * detail), t = dirt[c] * (0.8f + 0.4f * fine);
        s.albedo[c] = std::clamp((t * (1.0f - grassAmount) + g * grassAmount) * (1.0f - pebble) + stone[c] * pebble, 0.0f, 1.0f);
    }
    s.height = 0.3f * detail + 0.25f * fine + 0.6f * pebble + 0.15f * grassAmount;
    s.specular = 0.03f;
    s.gloss = 0.12f + 0.15f * pebble;
}

// Concrete: 2×2 cast slabs with seams, pores and stains.
void concrete(float x, float y, float size, Surface& s) {
    const float cell = size / 2.0f;
    const float u = std::fmod(x, cell), v = std::fmod(y, cell);
    const float edge = std::min(std::min(u, cell - u), std::min(v, cell - v));
    const float seam = smooth01((edge - size / 512.0f) / (size / 300.0f));
    const float stain = fbm(x / size, y / size, 6, 81), pores = fbm(x / size, y / size, 96, 83, 2);
    const float tone = 0.62f * (0.86f + 0.22f * stain) * (0.92f + 0.12f * pores) * (0.55f + 0.45f * seam);
    s.albedo[0] = tone;
    s.albedo[1] = tone * 0.98f;
    s.albedo[2] = tone * 0.94f;
    s.height = 0.6f * seam + 0.25f * pores + 0.1f * stain;
    s.specular = 0.035f;
    s.gloss = 0.18f + 0.1f * stain;
}

// Water: ripples from a few tiling sine waves and noise; the albedo is the deep color under the reflections.
void water(float x, float y, float size, Surface& s) {
    const float u = x / size, v = y / size;
    const float t = 2.0f * kPi;
    s.height = 0.25f * std::sin(t * (3.0f * u + 2.0f * v)) + 0.2f * std::sin(t * (5.0f * u - 3.0f * v)) +
               0.15f * std::sin(t * (-7.0f * u + 6.0f * v)) + 0.5f * fbm(u, v, 16, 91);
    const float n = fbm(u, v, 8, 93);
    s.albedo[0] = 0.05f + 0.03f * n;
    s.albedo[1] = 0.13f + 0.04f * n;
    s.albedo[2] = 0.15f + 0.04f * n;
    s.specular = 0.4f;
    s.gloss = 0.9f;
}

// Heat haze (M8 plan D11): turbulent, tileable noise; its normals become the distortion that distort emitters apply.
void haze(float x, float y, float size, Surface& s) {
    const float u = x / size, v = y / size;
    s.height = 0.6f * fbm(u, v, 4, 131) + 0.4f * fbm(u, v, 8, 137);
}

// Hazard stripes with grime, for the conveyor belt (scrolled along v).
void stripes(float x, float y, float size, Surface& s) {
    const float band = std::fmod(x + y, size / 4.0f) / (size / 4.0f);
    const float yellow = smooth01((std::fabs(band - 0.5f) - 0.22f) / 0.02f);
    const float grime = fbm(x / size, y / size, 8, 101), scratches = fbm(x / size * 8.0f, y / size * 0.5f, 64, 103, 2);
    const float dirt = std::clamp(grime * 1.5f - 0.45f, 0.0f, 0.8f);
    const float y0[3] = {0.85f, 0.62f, 0.08f}, b0[3] = {0.07f, 0.07f, 0.07f};
    for (int c = 0; c < 3; ++c) s.albedo[c] = (y0[c] * (1.0f - yellow) + b0[c] * yellow) * (1.0f - dirt) * (0.85f + 0.3f * scratches);
    s.height = 0.5f + 0.3f * scratches;
    s.specular = 0.06f;
    s.gloss = 0.3f;
}

// Fan: four blades and a hub inside a ring, in alpha (alpha test); rotated through the material's UV animation.
void fan(float x, float y, float size, Surface& s) {
    const float cx = x - size * 0.5f, cy = y - size * 0.5f;
    const float r = std::hypot(cx, cy) / (size * 0.5f);
    const float angle = std::atan2(cy, cx);
    const float blade = std::fabs(std::sin(2.0f * angle + 0.6f * r)); // 4 blades with a little sweep
    const float inBlade = r < 0.86f && blade > 0.72f ? 1.0f : 0.0f;
    const float hub = r < 0.18f ? 1.0f : 0.0f, ring = r > 0.9f && r < 0.98f ? 1.0f : 0.0f;
    s.alpha = std::max(std::max(inBlade, hub), ring);
    const float n = fbm(x / size, y / size, 16, 111);
    for (int c = 0; c < 3; ++c) s.albedo[c] = (0.3f + 0.02f * float(c)) * (0.8f + 0.4f * n) * (hub > 0.0f ? 0.6f : 1.0f);
    s.specular = 0.3f;
    s.gloss = 0.45f;
}

// Tangent space: +X = image right, +Y = image up (toward row 0) — the convention of the engine's MikkTSpace tangents,
// which are computed with v flipped, and of glTF normal maps. n ∝ (−∂h/∂x, +∂h/∂row, 1) with wrap-around.
void normalFromHeight(const std::vector<float>& h, uint32_t size, float strength, Image& out) {
    auto at = [&](int x, int y) { return h[size_t((y + int(size)) % int(size)) * size + size_t((x + int(size)) % int(size))]; };
    for (int y = 0; y < int(size); ++y)
        for (int x = 0; x < int(size); ++x) {
            const float dx = (at(x + 1, y) - at(x - 1, y)) * 0.5f * strength;
            const float dRow = (at(x, y + 1) - at(x, y - 1)) * 0.5f * strength;
            const glm::vec3 n = glm::normalize(glm::vec3(-dx, dRow, 1.0f));
            uint8_t* px = &out.rgba[(size_t(y) * size + size_t(x)) * 4];
            for (int c = 0; c < 3; ++c) px[c] = uint8_t(std::clamp(std::round((n[c] * 0.5f + 0.5f) * 255.0f), 0.0f, 255.0f));
            px[3] = 255;
        }
}

uint8_t unitByte(float v) {
    return uint8_t(std::clamp(std::round(v * 255.0f), 0.0f, 255.0f));
}

bool renderSurface(SurfaceFn fn, const std::string& map, uint32_t size, float normalStrength, Image& out, std::string& error) {
    if (map != "albedo" && map != "normal" && map != "specgloss" && map != "emission") {
        error = "unknown map '" + map + "' (albedo, normal, specgloss, emission)";
        return false;
    }
    out.width = out.height = size;
    out.rgba.assign(size_t(size) * size * 4, 255);
    std::vector<float> heights(map == "normal" ? size_t(size) * size : 0);
    for (uint32_t y = 0; y < size; ++y)
        for (uint32_t x = 0; x < size; ++x) {
            Surface s;
            fn(float(x) + 0.5f, float(y) + 0.5f, float(size), s);
            const size_t i = size_t(y) * size + x;
            uint8_t* px = &out.rgba[i * 4];
            if (map == "albedo") {
                for (int c = 0; c < 3; ++c) px[c] = unitByte(s.albedo[c]);
                px[3] = unitByte(s.alpha);
            } else if (map == "specgloss") {
                px[0] = px[1] = px[2] = linearToSrgb(s.specular);
                px[3] = unitByte(s.gloss);
            } else if (map == "emission") {
                for (int c = 0; c < 3; ++c) px[c] = unitByte(s.emission[c]);
            } else {
                heights[i] = s.height;
            }
        }
    if (map == "normal") normalFromHeight(heights, size, normalStrength, out);
    return true;
}

} // namespace

bool makeBuiltinMesh(const std::string& spec, RawMesh& out, std::string& error) {
    out = RawMesh{};
    const std::vector<std::string> p = split(spec, ':');
    if (p.empty()) {
        error = "empty builtin mesh name";
        return false;
    }
    if (p[0] == "crate") {
        RawMesh sides, caps;
        addBoxFaces(sides, glm::vec3(1.0f), 0.0f, true, false);
        addBoxFaces(caps, glm::vec3(1.0f), 0.0f, false, true);
        appendPart(out, sides, glm::mat4(1.0f), {1, 1}, 0);
        appendPart(out, caps, glm::mat4(1.0f), {1, 1}, 1);
    } else if (p[0] == "tree") {
        // Trunk (slot 0, bark) under a lumpy crown (slot 1, leaves): 3.2 m wide, 5 m tall, standing on y = 0.
        RawMesh trunk, crown;
        makeCylinder(trunk, 10);
        makeLumpySphere(crown, 14, 28, 0.22f, 1.3f);
        appendPart(out, trunk, placed({0.0f, 1.2f, 0.0f}, {0.3f, 2.4f, 0.3f}), {1, 2}, 0);
        appendPart(out, crown, placed({0.0f, 3.5f, 0.0f}, {3.2f, 2.8f, 3.2f}), {2, 2}, 1);
    } else if (p[0] == "rock" || p[0] == "bush") {
        // A boulder (about 1.6 × 1.0 × 1.4 m, partly sunk) or a bush (1.6 m, slot 0 either way).
        const bool rock = p[0] == "rock";
        RawMesh shape;
        makeLumpySphere(shape, rock ? 18 : 12, rock ? 36 : 24, rock ? 0.3f : 0.18f, rock ? 4.7f : 2.2f);
        appendPart(out, shape, rock ? placed({0.0f, 0.3f, 0.0f}, {1.6f, 1.0f, 1.4f}) : placed({0.0f, 0.6f, 0.0f}, {1.6f, 1.3f, 1.6f}),
                   {2, 1}, 0);
    } else if (p[0] == "lamp") {
        RawMesh cylinder, sphere;
        makeCylinder(cylinder, 24);
        makeSphere(sphere, 16, 32);
        appendPart(out, cylinder, placed({0.0f, 1.25f, 0.0f}, {0.12f, 2.3f, 0.12f}), {1, 4}, 0);  // pole, y 0.1–2.4
        appendPart(out, cylinder, placed({0.0f, 0.06f, 0.0f}, {0.6f, 0.12f, 0.6f}), {2, 0.5f}, 0); // base, y 0–0.12
        appendPart(out, sphere, placed({0.0f, 2.45f, 0.0f}, glm::vec3(0.4f)), {1, 1}, 1);          // bulb, y 2.25–2.65
    } else {
        if (p[0] == "cube") addBoxFaces(out, glm::vec3(1.0f), 0.0f, true, true);
        else if (p[0] == "box") {
            const float sx = p.size() > 1 ? std::strtof(p[1].c_str(), nullptr) : 0.0f;
            const float sy = p.size() > 2 ? std::strtof(p[2].c_str(), nullptr) : 0.0f;
            const float sz = p.size() > 3 ? std::strtof(p[3].c_str(), nullptr) : 0.0f;
            const float tile = p.size() > 4 ? std::strtof(p[4].c_str(), nullptr) : 0.0f;
            if (!(sx > 0.0f && sy > 0.0f && sz > 0.0f && tile > 0.0f)) {
                error = "box:<x>:<y>:<z>:<tile m> needs positive numbers";
                return false;
            }
            addBoxFaces(out, glm::vec3(sx, sy, sz), tile, true, true);
        }
        else if (p[0] == "wedge") {
            const float sx = p.size() > 1 ? std::strtof(p[1].c_str(), nullptr) : 0.0f;
            const float sy = p.size() > 2 ? std::strtof(p[2].c_str(), nullptr) : 0.0f;
            const float sz = p.size() > 3 ? std::strtof(p[3].c_str(), nullptr) : 0.0f;
            const float tile = p.size() > 4 ? std::strtof(p[4].c_str(), nullptr) : 0.0f;
            if (!(sx > 0.0f && sy > 0.0f && sz > 0.0f && tile > 0.0f)) {
                error = "wedge:<x>:<y>:<z>:<tile m> needs positive numbers";
                return false;
            }
            makeWedge(out, glm::vec3(sx, sy, sz), tile);
        }
        else if (p[0] == "terrain") {
            const float size = p.size() == 6 ? std::strtof(p[1].c_str(), nullptr) : 0.0f;
            const int cells = p.size() == 6 ? std::atoi(p[2].c_str()) : 0;
            const float tile = p.size() == 6 ? std::strtof(p[3].c_str(), nullptr) : 0.0f;
            if (!(size > 0.0f) || cells < 1 || cells > 256 || !(tile > 0.0f)) {
                error = "terrain:<size m>:<cells 1-256>:<tile m>:<center x>:<center z> expected";
                return false;
            }
            makeTerrain(out, size, uint32_t(cells), tile, std::strtof(p[4].c_str(), nullptr), std::strtof(p[5].c_str(), nullptr));
        }
        else if (p[0] == "sphere") makeSphere(out, 16, 32);
        else if (p[0] == "cylinder") makeCylinder(out, 24);
        else if (p[0] == "plane") {
            const float size = p.size() > 1 ? std::strtof(p[1].c_str(), nullptr) : 1.0f;
            const float repeats = p.size() > 2 ? std::strtof(p[2].c_str(), nullptr) : 1.0f;
            if (!(size > 0.0f) || !(repeats > 0.0f)) {
                error = "plane:<size>:<repeats> needs positive numbers";
                return false;
            }
            addQuad(out, glm::vec3(0), glm::vec3(size, 0, 0), glm::vec3(0, 0, -size), glm::vec2(repeats));
        } else {
            error = "unknown builtin mesh '" + p[0] + "'";
            return false;
        }
        out.submeshes.push_back({0, uint32_t(out.indices.size()), 0});
    }
    out.hasNormals = true;
    out.hasTangents = false;
    out.hasUv0 = true;
    return true;
}

bool makeBuiltinImage(const std::string& spec, Image& out, std::string& error) {
    const std::vector<std::string> p = split(spec, ':');
    if (p.empty()) {
        error = "empty builtin texture name";
        return false;
    }
    if (p[0] == "bricks" || p[0] == "plates" || p[0] == "panel") {
        if (p.size() != 2) {
            error = p[0] + ":<map> expected";
            return false;
        }
        const SurfaceFn fn = p[0] == "bricks" ? bricks : p[0] == "plates" ? plates : panel;
        return renderSurface(fn, p[1], 1024, p[0] == "bricks" ? 9.0f : 7.0f, out, error);
    }
    if (p[0] == "fence") return renderSurface(fence, "albedo", 512, 1.0f, out, error);
    if (p[0] == "ground" || p[0] == "concrete" || p[0] == "water") {
        if (p.size() != 2) {
            error = p[0] + ":<map> expected";
            return false;
        }
        const SurfaceFn fn = p[0] == "ground" ? ground : p[0] == "concrete" ? concrete : water;
        return renderSurface(fn, p[1], p[0] == "water" ? 512 : 1024, p[0] == "water" ? 4.0f : 6.0f, out, error);
    }
    if (p[0] == "stripes") return renderSurface(stripes, "albedo", 512, 1.0f, out, error);
    if (p[0] == "fan") return renderSurface(fan, "albedo", 256, 1.0f, out, error);
    if (p[0] == "haze") return renderSurface(haze, "normal", 128, 6.0f, out, error);
    if (p[0] == "foam") {
        // White streaks in alpha, for a water detail layer.
        constexpr uint32_t kSize = 512;
        out.width = out.height = kSize;
        out.rgba.resize(size_t(kSize) * kSize * 4);
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                const float u = (float(x) + 0.5f) / kSize, v = (float(y) + 0.5f) / kSize;
                const float n = fbm(u, v, 6, 121), m = fbm(u * 2.0f, v, 24, 123);
                const float foam = smooth01((n * 0.7f + m * 0.3f - 0.52f) / 0.12f);
                uint8_t* px = &out.rgba[(size_t(y) * kSize + x) * 4];
                px[0] = 225, px[1] = 236, px[2] = 240;
                px[3] = unitByte(foam);
            }
        return true;
    }
    if (p[0] == "fire" || p[0] == "smoke" || p[0] == "explosion") {
        // Sprite flipbooks (M6 plan D11): 4 × 4 cells of 128², row-major from the top left, straight alpha.
        //   fire       a looping flame: noise scrolls up one full period over the 16 frames, so the loop is seamless
        //   smoke      a looping puff that billows (the same seamless scroll)
        //   explosion  a one-shot fireball: it expands, cools from white through orange to dark red and fades to smoke
        constexpr uint32_t kCell = 128, kGrid = 4, kSize = kCell * kGrid;
        out.width = out.height = kSize;
        out.rgba.resize(size_t(kSize) * kSize * 4);
        const int kind = p[0] == "fire" ? 0 : p[0] == "smoke" ? 1 : 2;
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                const uint32_t frame = (y / kCell) * kGrid + x / kCell;
                const float t = float(frame) / 16.0f;
                const float cx = (float(x % kCell) + 0.5f) / kCell * 2.0f - 1.0f, cy = (float(y % kCell) + 0.5f) / kCell * 2.0f - 1.0f;
                glm::vec3 color(0.0f);
                float alpha = 0.0f;
                if (kind == 0) {
                    const float h = 1.0f - (cy + 1.0f) * 0.5f; // 0 at the cell's bottom, 1 at its top
                    const float n = fbm((cx + 1.0f) * 0.25f, h * 0.5f - t, 8, 97);
                    const float width = 0.55f * std::pow(std::max(0.0f, 1.0f - h), 0.7f) + 0.04f;
                    const float body = std::clamp((width * (0.55f + 0.9f * n) - std::fabs(cx + 0.15f * (n - 0.5f))) / (0.3f * width), 0.0f, 1.0f);
                    const float intensity = body * (1.0f - smooth01((h - 0.7f) / 0.3f)) * smooth01(h / 0.08f);
                    const glm::vec3 core(1.0f, 0.95f, 0.72f), mid(1.0f, 0.55f, 0.12f), edge(0.75f, 0.14f, 0.03f);
                    color = intensity > 0.6f ? glm::mix(mid, core, (intensity - 0.6f) / 0.4f) : glm::mix(edge, mid, intensity / 0.6f);
                    alpha = intensity;
                } else if (kind == 1) {
                    const float r = std::hypot(cx, cy);
                    const float n = fbm((cx + 1.0f) * 0.25f, (cy + 1.0f) * 0.25f + t, 8, 113);
                    const float density = (1.0f - smooth01((r - 0.25f) / 0.7f)) * (0.45f + 0.8f * n);
                    alpha = std::clamp(density, 0.0f, 1.0f) * 0.85f;
                    color = glm::vec3(0.42f + 0.22f * n + 0.1f * (1.0f - (cy + 1.0f) * 0.5f));
                } else {
                    const float progress = float(frame) / 15.0f;
                    const float r = std::hypot(cx, cy);
                    const float n = fbm((cx + 1.0f) * 0.25f, (cy + 1.0f) * 0.25f, 8, 131 + frame);
                    const float radius = 0.22f + 0.68f * std::sqrt(progress);
                    const float body = 1.0f - smooth01((r - radius * (0.75f + 0.35f * n)) / 0.18f);
                    const float heat = std::clamp(1.0f - progress * 1.15f + 0.3f * (n - 0.5f) - 0.4f * r / radius, 0.0f, 1.0f);
                    const glm::vec3 hot(1.0f, 0.95f, 0.8f), orange(1.0f, 0.5f, 0.1f), cool(0.35f, 0.08f, 0.03f), smoke(0.2f, 0.18f, 0.17f);
                    color = heat > 0.5f ? glm::mix(orange, hot, (heat - 0.5f) / 0.5f) : glm::mix(progress > 0.6f ? smoke : cool, orange, heat / 0.5f);
                    alpha = std::clamp(body, 0.0f, 1.0f) * (1.0f - smooth01((progress - 0.55f) / 0.45f));
                }
                uint8_t* px = &out.rgba[(size_t(y) * kSize + x) * 4];
                for (int c = 0; c < 3; ++c) px[c] = unitByte(std::pow(std::clamp(color[c], 0.0f, 1.0f), 1.0f / 2.2f)); // sRGB-ish
                px[3] = unitByte(std::clamp(alpha, 0.0f, 1.0f));
            }
        return true;
    }
    if (p[0] == "debris" || p[0] == "ring" || p[0] == "flash") {
        // Particle atlases (M7 plan D12), straight alpha:
        //   debris  2 × 2 cells of 256²: irregular dark chunks (rock and scorched metal), lit from the top left
        //   ring    one 512² shockwave: a bright band near the edge over a faint, fading inner disc
        //   flash   one 256² starburst: a hot core with eight soft spikes (muzzle flashes, explosion flashes)
        const int kind = p[0] == "debris" ? 0 : p[0] == "ring" ? 1 : 2;
        const uint32_t kSize = kind == 1 ? 512 : kind == 0 ? 512 : 256;
        const uint32_t grid = kind == 0 ? 2 : 1, cell = kSize / grid;
        out.width = out.height = kSize;
        out.rgba.resize(size_t(kSize) * kSize * 4);
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                const uint32_t index = (y / cell) * grid + x / cell;
                const float cx = (float(x % cell) + 0.5f) / float(cell) * 2.0f - 1.0f, cy = (float(y % cell) + 0.5f) / float(cell) * 2.0f - 1.0f;
                const float r = std::hypot(cx, cy), a = std::atan2(cy, cx);
                glm::vec3 color(1.0f);
                float alpha = 0.0f;
                if (kind == 0) {
                    // The outline: a radius that wanders with angle (low-frequency noise around the circle).
                    const float u = (a + kPi) / (2.0f * kPi);
                    const float edge = 0.48f + 0.3f * (fbm(u * 2.0f, 0.37f * float(index + 1), 2, 211 + index, 3) - 0.5f) * 2.0f;
                    alpha = std::clamp((edge - r) / 0.03f, 0.0f, 1.0f);
                    const float light = std::clamp(0.6f - 0.35f * (cx + cy), 0.25f, 1.0f); // image rows grow down: top left is lit
                    const float grain = 0.8f + 0.4f * fbm(cx * 0.5f + 0.5f, cy * 0.5f + 0.5f, 8, 223 + index);
                    const glm::vec3 base = index % 2 ? glm::vec3(0.22f, 0.21f, 0.2f) : glm::vec3(0.3f, 0.25f, 0.19f);
                    color = base * light * grain * 1.6f;
                } else if (kind == 1) {
                    const float band = std::exp(-std::pow((r - 0.78f) / 0.07f, 2.0f));
                    const float disc = r < 0.78f ? 0.18f * smooth01(r / 0.78f) : 0.0f;
                    const float n = 0.75f + 0.5f * fbm((a + kPi) / (2.0f * kPi) * 4.0f, r, 4, 241, 3);
                    alpha = std::clamp((band + disc) * n, 0.0f, 1.0f) * (1.0f - smooth01((r - 0.92f) / 0.08f));
                    color = glm::vec3(1.0f, 0.93f, 0.82f);
                } else {
                    const float core = std::exp(-r * r * 14.0f);
                    const float spikes = std::pow(std::max(0.0f, std::cos(a * 4.0f)), 24.0f) * std::max(0.0f, 1.0f - r) * 0.9f +
                                         std::pow(std::max(0.0f, std::cos(a * 4.0f + kPi * 0.25f * 2.0f)), 40.0f) * std::max(0.0f, 1.0f - r * 1.4f) * 0.6f;
                    const float glow = std::max(0.0f, 1.0f - r) * std::max(0.0f, 1.0f - r) * 0.5f;
                    alpha = std::clamp(core + spikes + glow, 0.0f, 1.0f);
                    color = glm::mix(glm::vec3(1.0f, 0.62f, 0.25f), glm::vec3(1.0f, 0.97f, 0.88f), std::clamp(core * 1.5f, 0.0f, 1.0f));
                }
                uint8_t* px = &out.rgba[(size_t(y) * kSize + x) * 4];
                for (int c = 0; c < 3; ++c) px[c] = unitByte(std::pow(std::clamp(color[c], 0.0f, 1.0f), 1.0f / 2.2f)); // sRGB-ish
                px[3] = unitByte(alpha);
            }
        return true;
    }
    if (p[0] == "radar") {
        // A 4 × 4 flipbook of a radar sweep: frame k (row-major) has the beam at k/16 of a turn, a fading trail and blips
        // that light up as the beam passes.
        constexpr uint32_t kCell = 128, kGrid = 4, kSize = kCell * kGrid;
        out.width = out.height = kSize;
        out.rgba.resize(size_t(kSize) * kSize * 4);
        const float blips[4][2] = {{0.35f, 0.6f}, {0.6f, 2.2f}, {0.75f, 3.6f}, {0.5f, 5.1f}}; // radius, angle
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                const uint32_t frame = (y / kCell) * kGrid + x / kCell;
                const float cx = (float(x % kCell) + 0.5f) / kCell * 2.0f - 1.0f, cy = (float(y % kCell) + 0.5f) / kCell * 2.0f - 1.0f;
                const float r = std::hypot(cx, cy), a = std::atan2(cy, cx) + kPi; // 0 – 2π
                const float beam = 2.0f * kPi * float(frame) / 16.0f;
                const float behind = std::fmod(beam - a + 4.0f * kPi, 2.0f * kPi); // how far the beam has passed
                float g = 0.06f;
                if (r < 0.95f) {
                    g += 0.85f * std::exp(-behind * 2.2f);
                    for (float ring : {0.33f, 0.66f, 0.95f}) g += 0.25f * (1.0f - smooth01(std::fabs(r - ring) * 60.0f));
                    for (const auto& b : blips) {
                        const float d = std::hypot(cx - b[0] * std::cos(b[1] - kPi), cy - b[0] * std::sin(b[1] - kPi));
                        const float passed = std::fmod(beam - b[1] + 4.0f * kPi, 2.0f * kPi);
                        g += (1.0f - smooth01(d * 25.0f)) * std::exp(-passed * 1.2f) * 1.5f;
                    }
                }
                uint8_t* px = &out.rgba[(size_t(y) * kSize + x) * 4];
                px[0] = unitByte(std::min(1.0f, g * 0.25f));
                px[1] = unitByte(std::min(1.0f, g));
                px[2] = unitByte(std::min(1.0f, g * 0.45f));
                px[3] = 255;
            }
        return true;
    }
    if (p[0] == "glow") {
        uint8_t rgb[3];
        if (p.size() != 2 || !parseColor(p[1], rgb)) {
            error = "glow:<RRGGBB> expected";
            return false;
        }
        constexpr uint32_t kSize = 256;
        out.width = out.height = kSize;
        out.rgba.resize(size_t(kSize) * kSize * 4);
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                const float r = std::hypot(float(x) + 0.5f - kSize * 0.5f, float(y) + 0.5f - kSize * 0.5f) / (kSize * 0.5f);
                const float a = std::clamp(1.0f - r, 0.0f, 1.0f);
                uint8_t* px = &out.rgba[(size_t(y) * kSize + x) * 4];
                px[0] = rgb[0], px[1] = rgb[1], px[2] = rgb[2];
                px[3] = unitByte(a * a);
            }
        return true;
    }
    if (p[0] == "ramp") {
        const int steps = p.size() == 2 ? std::atoi(p[1].c_str()) : 0;
        if (steps < 2 || steps > 64) {
            error = "ramp:<steps> needs 2–64 steps";
            return false;
        }
        constexpr uint32_t kWidth = 512, kHeight = 32;
        out.width = kWidth;
        out.height = kHeight;
        out.rgba.resize(size_t(kWidth) * kHeight * 4);
        for (uint32_t y = 0; y < kHeight; ++y)
            for (uint32_t x = 0; x < kWidth; ++x) {
                const uint32_t k = std::min(uint32_t(steps - 1), x * uint32_t(steps) / kWidth);
                const auto value = uint8_t(std::lround(255.0 * double(k) / double(steps - 1)));
                uint8_t* px = &out.rgba[(size_t(y) * kWidth + x) * 4];
                px[0] = px[1] = px[2] = value;
                px[3] = 255;
            }
        return true;
    }
    uint8_t a[3], b[3];
    if (p.size() != 4 || !parseColor(p[1], a) || !parseColor(p[2], b)) {
        error = "expected <pattern>:<RRGGBB>:<RRGGBB>:<count> or a procedural material (see Builtins.h)";
        return false;
    }
    const int count = std::atoi(p[3].c_str());
    if (count < 1 || count > 256) {
        error = "count must be 1..256";
        return false;
    }
    if (p[0] != "checker" && p[0] != "grid") {
        error = "unknown builtin texture '" + p[0] + "'";
        return false;
    }
    constexpr uint32_t kSize = 1024;
    out.width = out.height = kSize;
    out.rgba.assign(size_t(kSize) * kSize * 4, 255);
    const uint32_t cell = kSize / uint32_t(count);
    const uint32_t line = std::max(2u, cell / 16);
    for (uint32_t y = 0; y < kSize; ++y)
        for (uint32_t x = 0; x < kSize; ++x) {
            const bool useB = p[0] == "checker" ? ((x / cell) + (y / cell)) % 2 == 1 : (x % cell) < line || (y % cell) < line;
            uint8_t* px = &out.rgba[(size_t(y) * kSize + x) * 4];
            const uint8_t* c = useB ? b : a;
            px[0] = c[0], px[1] = c[1], px[2] = c[2];
        }
    return true;
}

} // namespace tools
