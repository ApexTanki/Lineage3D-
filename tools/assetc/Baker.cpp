#include "Baker.h"

#include <algorithm>
#include <atomic>
#include <cmath>

namespace tools {
namespace {

constexpr float kPi = 3.14159265358979f;
constexpr float kInsideThreshold = 0.3f; // fraction of rays reaching back faces above which a point is inside geometry
constexpr float kY0 = 0.282095f;         // SH basis constants (L0, L1)
constexpr float kY1 = 0.488603f;
// Indirect samples are clamped to this radiance (largest channel). A small bright emitter (a lamp's bulb) hit by an
// occasional hemisphere ray would otherwise speckle the lightmap; its light is baked as a point light anyway.
constexpr float kMaxIndirectRadiance = 2.0f;

glm::vec3 clampRadiance(const glm::vec3& c) {
    const float m = std::max(c.r, std::max(c.g, c.b));
    return m > kMaxIndirectRadiance ? c * (kMaxIndirectRadiance / m) : c;
}

float cross2(const glm::vec2& a, const glm::vec2& b) {
    return a.x * b.y - a.y * b.x;
}

// R2 low-discrepancy sequence (the plastic constant), shifted per texel (Cranley–Patterson rotation).
glm::vec2 r2(uint32_t k, const glm::vec2& offset) {
    const float x = 0.5f + 0.7548776662f * float(k) + offset.x, y = 0.5f + 0.5698402910f * float(k) + offset.y;
    return glm::vec2(x - std::floor(x), y - std::floor(y));
}

void basis(const glm::vec3& n, glm::vec3& t, glm::vec3& b) {
    const glm::vec3 a = std::fabs(n.y) < 0.999f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    t = glm::normalize(glm::cross(a, n));
    b = glm::cross(n, t);
}

glm::vec3 cosineDirection(const glm::vec3& n, const glm::vec2& u) {
    glm::vec3 t, b;
    basis(n, t, b);
    const float r = std::sqrt(u.x), phi = 2.0f * kPi * u.y;
    return glm::normalize(t * (r * std::cos(phi)) + b * (r * std::sin(phi)) + n * std::sqrt(std::max(0.0f, 1.0f - u.x)));
}

glm::vec3 coneDirection(const glm::vec3& axis, float cosMax, const glm::vec2& u) {
    glm::vec3 t, b;
    basis(axis, t, b);
    const float cosT = 1.0f - u.x * (1.0f - cosMax), sinT = std::sqrt(std::max(0.0f, 1.0f - cosT * cosT));
    const float phi = 2.0f * kPi * u.y;
    return glm::normalize(t * (sinT * std::cos(phi)) + b * (sinT * std::sin(phi)) + axis * cosT);
}

// Keeps a sample direction on the geometric front side (shading normals can lean past the surface).
glm::vec3 aboveSurface(glm::vec3 d, const glm::vec3& ng) {
    const float dn = glm::dot(d, ng);
    if (dn <= 1e-4f) d = glm::normalize(d - ng * (2.0f * dn) + ng * 1e-3f);
    return d;
}

float rayEpsilon(const glm::vec3& p) {
    return 0.003f + 1e-5f * std::max(std::fabs(p.x), std::max(std::fabs(p.y), std::fabs(p.z)));
}

uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

float attenuation(float distance, float range) {
    const float x = std::clamp(1.0f - (distance * distance) / (range * range), 0.0f, 1.0f);
    return x * x;
}

float smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Closest point of triangle (a, b, c) to p in 2D, as barycentrics (weights of b and c).
void closestBarycentrics(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, float& u, float& v, float& distance) {
    distance = INFINITY;
    const glm::vec2 corners[3] = {a, b, c};
    for (int e = 0; e < 3; ++e) {
        const glm::vec2 p0 = corners[e], p1 = corners[(e + 1) % 3];
        const glm::vec2 d = p1 - p0;
        const float len2 = glm::dot(d, d);
        const float t = len2 > 0.0f ? std::clamp(glm::dot(p - p0, d) / len2, 0.0f, 1.0f) : 0.0f;
        const float dist = glm::length(p - (p0 + d * t));
        if (dist < distance) {
            distance = dist;
            float w[3] = {0.0f, 0.0f, 0.0f};
            w[e] = 1.0f - t;
            w[(e + 1) % 3] = t;
            u = w[1];
            v = w[2];
        }
    }
}

} // namespace

glm::vec4 BakeTexture::sample(const glm::vec2& uv) const {
    if (width == 0 || height == 0) return glm::vec4(1.0f);
    const float x = (uv.x - std::floor(uv.x)) * float(width) - 0.5f, y = (uv.y - std::floor(uv.y)) * float(height) - 0.5f;
    const float fx = std::floor(x), fy = std::floor(y);
    const float ax = x - fx, ay = y - fy;
    auto at = [&](int ix, int iy) {
        ix = ((ix % int(width)) + int(width)) % int(width);
        iy = ((iy % int(height)) + int(height)) % int(height);
        return texels[size_t(iy) * width + size_t(ix)];
    };
    const int ix = int(fx), iy = int(fy);
    return glm::mix(glm::mix(at(ix, iy), at(ix + 1, iy), ax), glm::mix(at(ix, iy + 1), at(ix + 1, iy + 1), ax), ay);
}

Baker::Rng::Rng(uint64_t seed) : state(mix64(seed + 0x9E3779B97F4A7C15ull)) {}

uint32_t Baker::Rng::next() {
    const uint64_t old = state;
    state = old * 6364136223846793005ull + 1442695040888963407ull;
    const auto xorshifted = uint32_t(((old >> 18u) ^ old) >> 27u);
    const auto rot = uint32_t(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

float Baker::Rng::uniform() {
    return float(next() >> 8) * (1.0f / 16777216.0f);
}

Baker::Baker(const BakeSettings& settings, const SkySettings& sky, const glm::vec3& sunDirection, const glm::vec3& sunColor)
    : m_settings(settings), m_skySettings(sky), m_sunDirection(glm::normalize(sunDirection)), m_sunColor(sunColor) {
    m_skyCube = renderSkyCube(sky, 64);
    m_ambientUp = skyIrradiance(sky, glm::vec3(0.0f, 1.0f, 0.0f));
    m_ambientDown = skyIrradiance(sky, glm::vec3(0.0f, -1.0f, 0.0f));
}

uint32_t Baker::addTexture(BakeTexture texture) {
    m_textures.push_back(std::move(texture));
    return uint32_t(m_textures.size() - 1);
}

uint32_t Baker::addMaterial(const BakeMaterial& material) {
    m_materials.push_back(material);
    return uint32_t(m_materials.size() - 1);
}

void Baker::addMesh(BakeMesh mesh) {
    const auto meshIndex = uint32_t(m_meshes.size());
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        Triangle tri;
        tri.mesh = meshIndex;
        tri.i0 = mesh.indices[t];
        tri.i1 = mesh.indices[t + 1];
        tri.i2 = mesh.indices[t + 2];
        tri.material = mesh.materials[t / 3];
        const glm::vec3 n = glm::cross(mesh.positions[tri.i1] - mesh.positions[tri.i0], mesh.positions[tri.i2] - mesh.positions[tri.i0]);
        const float len = glm::length(n);
        if (len < 1e-12f) continue; // degenerate: never hit, never lightmapped
        tri.normal = n / len;
        m_triangles.push_back(tri);
        m_scene.addTriangle(mesh.positions[tri.i0], mesh.positions[tri.i1], mesh.positions[tri.i2]);
    }
    m_meshes.push_back(std::move(mesh));
}

void Baker::addLight(const BakeLight& light) {
    m_lights.push_back(light);
}

void Baker::build() {
    m_scene.build();
}

bool Baker::alphaFilter(const void* context, uint32_t triangle, float u, float v) {
    const Baker& self = *static_cast<const Baker*>(context);
    const Triangle& t = self.m_triangles[triangle];
    const BakeMaterial& m = self.m_materials[t.material];
    if (!m.alphaTest) return true;
    const BakeMesh& mesh = self.m_meshes[t.mesh];
    const glm::vec2 uv = mesh.uv0[t.i0] * (1.0f - u - v) + mesh.uv0[t.i1] * u + mesh.uv0[t.i2] * v;
    const float alpha = m.baseColor.a * (m.albedo >= 0 ? self.m_textures[size_t(m.albedo)].sample(uv).a : 1.0f);
    return alpha >= m.alphaCutoff;
}

bool Baker::sunShadowFilter(const void* context, uint32_t triangle, float u, float v) {
    const Baker& self = *static_cast<const Baker*>(context);
    if (!self.m_materials[self.m_triangles[triangle].material].castShadows) return false;
    return alphaFilter(context, triangle, u, v);
}

bool Baker::lightShadowFilter(const void* context, uint32_t triangle, float u, float v) {
    const Baker& self = *static_cast<const Baker*>(context);
    const BakeMaterial& m = self.m_materials[self.m_triangles[triangle].material];
    if (m.emission.r > 0.0f || m.emission.g > 0.0f || m.emission.b > 0.0f) return false;
    return sunShadowFilter(context, triangle, u, v);
}

Baker::Surface Baker::surface(uint32_t triangle, float u, float v) const {
    const Triangle& t = m_triangles[triangle];
    const BakeMesh& m = m_meshes[t.mesh];
    const float w = 1.0f - u - v;
    Surface s;
    s.position = m.positions[t.i0] * w + m.positions[t.i1] * u + m.positions[t.i2] * v;
    const glm::vec3 n = m.normals[t.i0] * w + m.normals[t.i1] * u + m.normals[t.i2] * v;
    const float len = glm::length(n);
    s.normal = len > 1e-8f ? n / len : t.normal;
    s.geometricNormal = t.normal;
    if (glm::dot(s.normal, s.geometricNormal) < 0.0f) s.normal = s.geometricNormal; // a broken shading normal
    s.uv0 = m.uv0[t.i0] * w + m.uv0[t.i1] * u + m.uv0[t.i2] * v;
    s.uv1 = m.uv1[t.i0] * w + m.uv1[t.i1] * u + m.uv1[t.i2] * v;
    s.material = &m_materials[t.material];
    s.rect = m.lightmapRect;
    return s;
}

glm::vec3 Baker::albedo(const Surface& s) const {
    glm::vec3 a = glm::vec3(s.material->baseColor);
    if (s.material->albedo >= 0) a *= glm::vec3(m_textures[size_t(s.material->albedo)].sample(s.uv0));
    return glm::min(a, glm::vec3(0.95f)); // no surface reflects everything
}

glm::vec3 Baker::emission(const Surface& s) const {
    glm::vec3 e = s.material->emission;
    if (s.material->emissionMap >= 0) e *= glm::vec3(m_textures[size_t(s.material->emissionMap)].sample(s.uv0));
    return e;
}

glm::vec3 Baker::sky(const glm::vec3& direction) const {
    uint32_t face = 0;
    float s = 0.0f, t = 0.0f;
    cubeFaceCoordinates(direction, face, s, t);
    const HdrImage& image = m_skyCube[face];
    const float x = std::clamp(s * float(image.width) - 0.5f, 0.0f, float(image.width - 1));
    const float y = std::clamp(t * float(image.height) - 0.5f, 0.0f, float(image.height - 1));
    const auto x0 = uint32_t(x), y0 = uint32_t(y);
    const uint32_t x1 = std::min(x0 + 1, image.width - 1), y1 = std::min(y0 + 1, image.height - 1);
    const float fx = x - float(x0), fy = y - float(y0);
    return glm::mix(glm::mix(image.at(x0, y0), image.at(x1, y0), fx), glm::mix(image.at(x0, y1), image.at(x1, y1), fx), fy);
}

float Baker::sunVisibility(const glm::vec3& origin, uint32_t samples, float offset0, float offset1) const {
    const float cosMax = std::cos(m_settings.sunAngularRadius);
    uint32_t visible = 0;
    for (uint32_t k = 0; k < samples; ++k) {
        Ray ray;
        ray.origin = origin;
        ray.direction = coneDirection(m_sunDirection, cosMax, r2(k, glm::vec2(offset0, offset1)));
        if (!m_scene.occluded(ray, sunShadowFilter, this)) ++visible;
    }
    return float(visible) / float(std::max(1u, samples));
}

glm::vec3 Baker::lightDirect(const BakeLight& light, const glm::vec3& origin, const glm::vec3& normal, uint32_t samples, Rng& rng) const {
    const glm::vec3 toLight = light.position - origin;
    const float distance = glm::length(toLight);
    if (distance >= light.range || distance < 1e-4f) return glm::vec3(0.0f);
    const glm::vec3 l = toLight / distance;
    const float ndl = glm::dot(normal, l);
    if (ndl <= 0.0f) return glm::vec3(0.0f);
    float strength = attenuation(distance, light.range);
    if (light.spot) strength *= smoothstep(light.cosOuter, light.cosInner, glm::dot(-l, light.direction));
    if (strength <= 0.0f) return glm::vec3(0.0f);
    uint32_t visible = 0;
    for (uint32_t k = 0; k < samples; ++k) {
        // A point on the source sphere, for soft shadows.
        glm::vec3 jitter;
        do {
            jitter = glm::vec3(rng.uniform(), rng.uniform(), rng.uniform()) * 2.0f - 1.0f;
        } while (glm::dot(jitter, jitter) > 1.0f);
        const glm::vec3 target = light.position + jitter * light.radius;
        const glm::vec3 d = target - origin;
        const float len = glm::length(d);
        Ray ray;
        ray.origin = origin;
        ray.direction = d / len;
        ray.tMax = std::max(0.0f, len - light.radius - 1e-3f);
        if (!m_scene.occluded(ray, lightShadowFilter, this)) ++visible;
    }
    return light.color * (strength * ndl * float(visible) / float(std::max(1u, samples)));
}

glm::vec3 Baker::directAt(const Surface& s, const glm::vec3& origin, Rng& rng) const {
    glm::vec3 e(0.0f);
    const float ndl = glm::dot(s.normal, m_sunDirection);
    if (ndl > 0.0f) e += m_sunColor * (ndl * sunVisibility(origin, 1, rng.uniform(), rng.uniform()));
    for (const BakeLight& light : m_lights) e += lightDirect(light, origin, s.normal, 1, rng);
    return e;
}

glm::vec3 Baker::trace(const Ray& ray, uint32_t depth, Rng& rng, bool& backFace) const {
    backFace = false;
    RayHit hit;
    if (!m_scene.intersect(ray, hit, alphaFilter, this)) return sky(ray.direction);
    Surface s = surface(hit.triangle, hit.u, hit.v);
    if (hit.backFace) {
        if (!s.material->doubleSided) {
            backFace = true;
            return glm::vec3(0.0f);
        }
        s.normal = -s.normal;
        s.geometricNormal = -s.geometricNormal;
    }
    const glm::vec3 origin = s.position + s.geometricNormal * rayEpsilon(s.position);
    glm::vec3 e = directAt(s, origin, rng);
    if (depth < m_settings.bounces) {
        Ray next;
        next.origin = origin;
        next.direction = aboveSurface(cosineDirection(s.normal, glm::vec2(rng.uniform(), rng.uniform())), s.geometricNormal);
        bool ignored = false;
        e += trace(next, depth + 1, rng, ignored);
    }
    return emission(s) + albedo(s) * e;
}

glm::vec3 Baker::lightmapAt(const Surface& s) const {
    if (s.rect < 0 || size_t(s.rect) >= m_rects.size() || m_light.empty()) return glm::mix(m_ambientDown, m_ambientUp, s.normal.y * 0.5f + 0.5f);
    const LightmapRect& r = m_rects[size_t(s.rect)];
    const HdrImage& page = m_light[r.page];
    const float x = std::clamp(s.uv1.x * float(r.width) - 0.5f, 0.0f, float(r.width - 1)) + float(r.x);
    const float y = std::clamp(s.uv1.y * float(r.height) - 0.5f, 0.0f, float(r.height - 1)) + float(r.y);
    const auto x0 = uint32_t(x), y0 = uint32_t(y);
    const uint32_t x1 = std::min(x0 + 1, r.x + r.width - 1), y1 = std::min(y0 + 1, r.y + r.height - 1);
    const float fx = x - float(x0), fy = y - float(y0);
    return glm::mix(glm::mix(page.at(x0, y0), page.at(x1, y0), fx), glm::mix(page.at(x0, y1), page.at(x1, y1), fx), fy);
}

glm::vec3 Baker::shadeWithLightmap(const Ray& ray, Rng& rng, bool& backFace) const {
    backFace = false;
    RayHit hit;
    if (!m_scene.intersect(ray, hit, alphaFilter, this)) return sky(ray.direction);
    Surface s = surface(hit.triangle, hit.u, hit.v);
    if (hit.backFace) {
        if (!s.material->doubleSided) {
            backFace = true;
            return glm::vec3(0.0f);
        }
        s.normal = -s.normal;
        s.geometricNormal = -s.geometricNormal;
    }
    const glm::vec3 origin = s.position + s.geometricNormal * rayEpsilon(s.position);
    glm::vec3 e = lightmapAt(s);
    const float ndl = glm::dot(s.normal, m_sunDirection);
    if (ndl > 0.0f) e += m_sunColor * (ndl * sunVisibility(origin, 2, rng.uniform(), rng.uniform()));
    return emission(s) + albedo(s) * e;
}

void Baker::bakeLightmaps(const std::vector<LightmapRect>& rects, uint32_t pageSize, uint32_t pages) {
    m_rects = rects;
    m_pageSize = pageSize;
    m_light.assign(pages, HdrImage(pageSize, pageSize));
    m_shadow.assign(pages, std::vector<float>(size_t(pageSize) * pageSize, 1.0f));

    // 1. Which triangle point each texel samples: texel centers inside a triangle, then texels within 0.75 texel of
    //    one (chart borders) at their closest point, so bilinear filtering at the border reads real light.
    struct Coverage {
        uint32_t triangle = ~0u;
        float u = 0.0f, v = 0.0f;
        float distance = INFINITY;
    };
    std::vector<std::vector<Coverage>> coverage(pages, std::vector<Coverage>(size_t(pageSize) * pageSize));
    for (uint32_t ti = 0; ti < m_triangles.size(); ++ti) {
        const Triangle& t = m_triangles[ti];
        const BakeMesh& mesh = m_meshes[t.mesh];
        if (mesh.lightmapRect < 0) continue;
        const LightmapRect& r = rects[size_t(mesh.lightmapRect)];
        const glm::vec2 size(float(r.width), float(r.height)), origin(float(r.x), float(r.y));
        const glm::vec2 q0 = mesh.uv1[t.i0] * size + origin, q1 = mesh.uv1[t.i1] * size + origin, q2 = mesh.uv1[t.i2] * size + origin;
        const float area = cross2(q1 - q0, q2 - q0);
        if (std::fabs(area) < 1e-10f) continue;
        const glm::vec2 lo = glm::min(q0, glm::min(q1, q2)) - 1.0f, hi = glm::max(q0, glm::max(q1, q2)) + 1.0f;
        const int x0 = std::max(int(r.x), int(std::floor(lo.x))), x1 = std::min(int(r.x + r.width) - 1, int(std::ceil(hi.x)));
        const int y0 = std::max(int(r.y), int(std::floor(lo.y))), y1 = std::min(int(r.y + r.height) - 1, int(std::ceil(hi.y)));
        std::vector<Coverage>& page = coverage[r.page];
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const glm::vec2 c(float(x) + 0.5f, float(y) + 0.5f);
                const float w0 = cross2(q1 - c, q2 - c) / area, w1 = cross2(q2 - c, q0 - c) / area, w2 = 1.0f - w0 - w1;
                float u = w1, v = w2, distance = 0.0f;
                if (std::min(w0, std::min(w1, w2)) < -1e-5f) {
                    closestBarycentrics(c, q0, q1, q2, u, v, distance);
                    if (distance > 0.75f) continue;
                }
                Coverage& cell = page[size_t(y) * pageSize + size_t(x)];
                if (distance < cell.distance) cell = {ti, u, v, distance};
            }
    }

    // 2. Integrate every covered texel.
    struct Job {
        uint32_t page, x, y;
    };
    std::vector<Job> jobs;
    for (uint32_t p = 0; p < pages; ++p)
        for (uint32_t i = 0; i < pageSize * pageSize; ++i)
            if (coverage[p][i].triangle != ~0u) jobs.push_back({p, i % pageSize, i / pageSize});
    std::vector<std::vector<uint8_t>> valid(pages, std::vector<uint8_t>(size_t(pageSize) * pageSize, 0));
    // The indirect part and the normal of every texel, for the denoising filter below.
    std::vector<std::vector<glm::vec3>> indirectLight(pages, std::vector<glm::vec3>(size_t(pageSize) * pageSize, glm::vec3(0.0f)));
    std::vector<std::vector<glm::vec3>> normals(pages, std::vector<glm::vec3>(size_t(pageSize) * pageSize, glm::vec3(0.0f)));
    std::atomic<uint32_t> invalid{0};
    constexpr uint32_t kChunk = 64;
    parallelFor(uint32_t((jobs.size() + kChunk - 1) / kChunk), [&](uint32_t chunk) {
        for (size_t j = size_t(chunk) * kChunk; j < std::min(jobs.size(), size_t(chunk + 1) * kChunk); ++j) {
            const Job& job = jobs[j];
            const size_t index = size_t(job.y) * pageSize + job.x;
            const Coverage& c = coverage[job.page][index];
            Rng rng((uint64_t(job.page) << 40) ^ (uint64_t(job.y) << 20) ^ job.x);
            const Surface s = surface(c.triangle, c.u, c.v);
            const glm::vec3 origin = s.position + s.geometricNormal * rayEpsilon(s.position);
            glm::vec3 light(0.0f);
            for (const BakeLight& l : m_lights) light += lightDirect(l, origin, s.normal, m_settings.lightSamples, rng);
            const glm::vec2 offset(rng.uniform(), rng.uniform());
            glm::vec3 indirect(0.0f);
            uint32_t back = 0;
            for (uint32_t k = 0; k < m_settings.samples; ++k) {
                Ray ray;
                ray.origin = origin;
                ray.direction = aboveSurface(cosineDirection(s.normal, r2(k, offset)), s.geometricNormal);
                bool backFace = false;
                indirect += clampRadiance(trace(ray, 1, rng, backFace));
                back += backFace ? 1u : 0u;
            }
            indirect /= float(std::max(1u, m_settings.samples));
            indirectLight[job.page][index] = indirect;
            normals[job.page][index] = s.normal;
            m_light[job.page].pixels[index] = light; // direct only for now; the filtered indirect part is added below
            m_shadow[job.page][index] = sunVisibility(origin, m_settings.sunSamples, rng.uniform(), rng.uniform());
            const bool inside = float(back) > kInsideThreshold * float(m_settings.samples);
            valid[job.page][index] = inside ? 0 : 1;
            if (inside) invalid.fetch_add(1, std::memory_order_relaxed);
        }
    });
    m_bakedTexels = uint32_t(jobs.size());
    m_invalidTexels = invalid.load();

    // 3. Indirect light is smooth but carries Monte Carlo noise: a 5×5 filter within each rect over valid texels whose
    //    normals agree (so creases stay sharp) removes most of it. Direct light from baked lights stays unfiltered.
    filterIndirect(rects, valid, indirectLight, normals);

    // 4. Texels inside geometry and the padding take their valid neighbours' light.
    dilate(rects, valid);
}

void Baker::filterIndirect(const std::vector<LightmapRect>& rects, const std::vector<std::vector<uint8_t>>& valid,
                           const std::vector<std::vector<glm::vec3>>& indirect, const std::vector<std::vector<glm::vec3>>& normals) {
    static constexpr float kWeights[5] = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f}; // separable tent, applied as a 5×5 product
    for (const LightmapRect& r : rects) {
        const std::vector<uint8_t>& mask = valid[r.page];
        const std::vector<glm::vec3>& source = indirect[r.page];
        const std::vector<glm::vec3>& n = normals[r.page];
        HdrImage& light = m_light[r.page];
        for (uint32_t y = r.y; y < r.y + r.height; ++y)
            for (uint32_t x = r.x; x < r.x + r.width; ++x) {
                const size_t index = size_t(y) * m_pageSize + x;
                if (!mask[index]) continue;
                glm::vec3 sum(0.0f);
                float total = 0.0f;
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) {
                        const int nx = int(x) + dx, ny = int(y) + dy;
                        if (nx < int(r.x) || ny < int(r.y) || nx >= int(r.x + r.width) || ny >= int(r.y + r.height)) continue;
                        const size_t ni = size_t(ny) * m_pageSize + size_t(nx);
                        if (!mask[ni] || glm::dot(n[ni], n[index]) < 0.9f) continue; // another face or chart: keep the crease
                        const float w = kWeights[dx + 2] * kWeights[dy + 2];
                        sum += source[ni] * w;
                        total += w;
                    }
                light.pixels[index] += sum / total; // total > 0: the texel itself always counts
            }
    }
}

void Baker::dilate(const std::vector<LightmapRect>& rects, std::vector<std::vector<uint8_t>>& valid) {
    for (const LightmapRect& r : rects) {
        std::vector<uint8_t>& mask = valid[r.page];
        HdrImage& light = m_light[r.page];
        std::vector<float>& shadow = m_shadow[r.page];
        std::vector<std::pair<uint32_t, glm::vec4>> updates;
        for (int pass = 0; pass < 6; ++pass) {
            updates.clear();
            for (uint32_t y = r.y; y < r.y + r.height; ++y)
                for (uint32_t x = r.x; x < r.x + r.width; ++x) {
                    const size_t index = size_t(y) * m_pageSize + x;
                    if (mask[index]) continue;
                    glm::vec3 sum(0.0f);
                    float shadowSum = 0.0f;
                    uint32_t n = 0;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nx = int(x) + dx, ny = int(y) + dy;
                            if ((dx == 0 && dy == 0) || nx < int(r.x) || ny < int(r.y) || nx >= int(r.x + r.width) || ny >= int(r.y + r.height))
                                continue;
                            const size_t ni = size_t(ny) * m_pageSize + size_t(nx);
                            if (!mask[ni]) continue;
                            sum += light.pixels[ni];
                            shadowSum += shadow[ni];
                            ++n;
                        }
                    if (n) updates.push_back({uint32_t(index), glm::vec4(sum / float(n), shadowSum / float(n))});
                }
            if (updates.empty()) break;
            for (const auto& [index, value] : updates) {
                light.pixels[index] = glm::vec3(value);
                shadow[index] = value.w;
                mask[index] = 1;
            }
        }
    }
}

void Baker::bakeProbes(assets::ProbeGridData& grid) const {
    const uint32_t count = grid.dims[0] * grid.dims[1] * grid.dims[2];
    grid.probes.assign(count, assets::ProbeGridData::Probe{});
    const uint32_t rays = std::max(16u, m_settings.probeRays);
    std::vector<glm::vec3> directions(rays);
    for (uint32_t k = 0; k < rays; ++k) { // Fibonacci sphere: uniform, deterministic
        const float z = 1.0f - 2.0f * (float(k) + 0.5f) / float(rays);
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z)), phi = float(k) * 2.39996323f;
        directions[k] = glm::vec3(r * std::cos(phi), z, r * std::sin(phi));
    }
    parallelFor(count, [&](uint32_t i) {
        const uint32_t x = i % grid.dims[0], y = (i / grid.dims[0]) % grid.dims[1], z = i / (grid.dims[0] * grid.dims[1]);
        const glm::vec3 p = grid.origin + grid.spacing * glm::vec3(float(x), float(y), float(z));
        Rng rng(0xC0FFEEull + i);
        glm::vec3 c[4] = {glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(0.0f)};
        uint32_t back = 0;
        for (const glm::vec3& d : directions) {
            Ray ray;
            ray.origin = p;
            ray.direction = d;
            bool backFace = false;
            const glm::vec3 radiance = clampRadiance(shadeWithLightmap(ray, rng, backFace));
            back += backFace ? 1u : 0u;
            c[0] += radiance * kY0;
            c[1] += radiance * (kY1 * d.y);
            c[2] += radiance * (kY1 * d.z);
            c[3] += radiance * (kY1 * d.x);
        }
        for (glm::vec3& v : c) v *= 4.0f * kPi / float(rays);
        // Baked lights reach the probe directly: a delta of E/π = Li at normal incidence is π·Li·Y(l) in SH.
        for (const BakeLight& light : m_lights) {
            const glm::vec3 toLight = light.position - p;
            const float distance = glm::length(toLight);
            if (distance >= light.range || distance < 1e-4f) continue;
            const glm::vec3 l = toLight / distance;
            float strength = attenuation(distance, light.range);
            if (light.spot) strength *= smoothstep(light.cosOuter, light.cosInner, glm::dot(-l, light.direction));
            if (strength <= 0.0f) continue;
            // Visibility from the probe (lightDirect with a normal facing the light).
            const glm::vec3 li = lightDirect(light, p, l, 8, rng);
            c[0] += li * (kPi * kY0);
            c[1] += li * (kPi * kY1 * l.y);
            c[2] += li * (kPi * kY1 * l.z);
            c[3] += li * (kPi * kY1 * l.x);
        }
        assets::ProbeGridData::Probe& probe = grid.probes[i];
        // Irradiance form: E/π(n) = Y0·c0 + (2/3)·Y1·(c11·n.x + c1−1·n.y + c10·n.z).
        const glm::vec3 a = c[0] * kY0, bx = c[3] * (kY1 * 2.0f / 3.0f), by = c[1] * (kY1 * 2.0f / 3.0f), bz = c[2] * (kY1 * 2.0f / 3.0f);
        for (int ch = 0; ch < 3; ++ch) probe.sh[ch] = glm::vec4(bx[ch], by[ch], bz[ch], a[ch]);
        probe.sunOcclusion = sunVisibility(p, m_settings.sunSamples, rng.uniform(), rng.uniform());
        probe.valid = float(back) > kInsideThreshold * float(rays) ? 0.0f : 1.0f;
    });
}

CubeFaces Baker::renderCube(const glm::vec3& position, uint32_t faceSize) const {
    CubeFaces faces;
    for (HdrImage& f : faces) f = HdrImage(faceSize, faceSize);
    parallelFor(6 * faceSize, [&](uint32_t row) {
        const uint32_t face = row / faceSize, y = row % faceSize;
        Rng rng(0xBEEFull + row);
        for (uint32_t x = 0; x < faceSize; ++x) {
            glm::vec3 sum(0.0f);
            for (uint32_t k = 0; k < 4; ++k) {
                const float s = (float(x) + 0.25f + 0.5f * float(k & 1)) / float(faceSize);
                const float t = (float(y) + 0.25f + 0.5f * float(k >> 1)) / float(faceSize);
                Ray ray;
                ray.origin = position;
                ray.direction = glm::normalize(cubeFaceDirection(face, s, t));
                bool backFace = false;
                sum += shadeWithLightmap(ray, rng, backFace);
            }
            faces[face].at(x, y) = sum * 0.25f;
        }
    });
    return faces;
}

} // namespace tools
