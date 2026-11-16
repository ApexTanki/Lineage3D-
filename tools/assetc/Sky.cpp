#include "Sky.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>

namespace tools {
namespace {

constexpr float kPi = 3.14159265358979f;

uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float lattice(int x, int y, uint32_t seed) {
    return float(hash32(uint32_t(x) * 73856093u ^ uint32_t(y) * 19349663u ^ seed * 83492791u) & 0xFFFFFFu) / 16777215.0f;
}

float smoothstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float valueNoise(float x, float y, uint32_t seed) {
    const float fx = std::floor(x), fy = std::floor(y);
    const int xi = int(fx), yi = int(fy);
    float u = x - fx, v = y - fy;
    u = u * u * (3.0f - 2.0f * u);
    v = v * v * (3.0f - 2.0f * v);
    const float a = lattice(xi, yi, seed), b = lattice(xi + 1, yi, seed);
    const float c = lattice(xi, yi + 1, seed), d = lattice(xi + 1, yi + 1, seed);
    return (a + (b - a) * u) * (1.0f - v) + (c + (d - c) * u) * v;
}

float fbm(float x, float y, uint32_t seed) {
    float sum = 0.0f, amplitude = 0.5f, total = 0.0f;
    for (int octave = 0; octave < 5; ++octave) {
        sum += valueNoise(x, y, seed + uint32_t(octave) * 101u) * amplitude;
        total += amplitude;
        x *= 2.03f;
        y *= 2.03f;
        amplitude *= 0.5f;
    }
    return sum / total;
}

float radicalInverse(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

void basis(const glm::vec3& n, glm::vec3& t, glm::vec3& b) {
    const glm::vec3 a = std::fabs(n.y) < 0.999f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    t = glm::normalize(glm::cross(a, n));
    b = glm::cross(n, t);
}

glm::vec3 sampleFace(const HdrImage& image, float s, float t) {
    const float x = std::clamp(s * float(image.width) - 0.5f, 0.0f, float(image.width - 1));
    const float y = std::clamp(t * float(image.height) - 0.5f, 0.0f, float(image.height - 1));
    const auto x0 = uint32_t(x), y0 = uint32_t(y);
    const uint32_t x1 = std::min(x0 + 1, image.width - 1), y1 = std::min(y0 + 1, image.height - 1);
    const float fx = x - float(x0), fy = y - float(y0);
    return glm::mix(glm::mix(image.at(x0, y0), image.at(x1, y0), fx), glm::mix(image.at(x0, y1), image.at(x1, y1), fx), fy);
}

glm::vec3 sampleCube(const CubeFaces& cube, const glm::vec3& direction) {
    uint32_t face = 0;
    float s = 0.0f, t = 0.0f;
    cubeFaceCoordinates(direction, face, s, t);
    return sampleFace(cube[face], s, t);
}

CubeFaces downsample(const CubeFaces& src) {
    CubeFaces dst;
    const uint32_t size = std::max(1u, src[0].width / 2);
    for (uint32_t f = 0; f < 6; ++f) {
        dst[f] = HdrImage(size, size);
        for (uint32_t y = 0; y < size; ++y)
            for (uint32_t x = 0; x < size; ++x) {
                glm::vec3 sum(0.0f);
                for (uint32_t k = 0; k < 4; ++k)
                    sum += src[f].at(std::min(src[f].width - 1, x * 2 + (k & 1)), std::min(src[f].height - 1, y * 2 + (k >> 1)));
                dst[f].at(x, y) = sum * 0.25f;
            }
    }
    return dst;
}

} // namespace

void parallelFor(uint32_t count, const std::function<void(uint32_t)>& fn) {
    const uint32_t threads = std::max(1u, std::min(count, std::thread::hardware_concurrency()));
    std::atomic<uint32_t> next{0};
    auto worker = [&] {
        for (uint32_t i = next.fetch_add(1); i < count; i = next.fetch_add(1)) fn(i);
    };
    std::vector<std::thread> pool;
    for (uint32_t t = 1; t < threads; ++t) pool.emplace_back(worker);
    worker();
    for (std::thread& t : pool) t.join();
}

glm::vec3 skyRadiance(const SkySettings& sky, const glm::vec3& direction) {
    const glm::vec3 d = glm::normalize(direction);
    const glm::vec3 sun = glm::normalize(sky.sunDirection);
    const float cosSun = std::max(0.0f, glm::dot(d, sun));
    glm::vec3 c;
    if (d.y >= 0.0f) {
        c = glm::mix(sky.zenith, sky.horizon, std::pow(1.0f - d.y, 2.5f));
        c += sky.glow * (std::pow(cosSun, sky.glowExponent) + 0.15f * std::pow(cosSun, 3.0f));
        if (sky.clouds > 0.0f) {
            // Clouds on a plane above the camera; they thin out toward the horizon, where the projection would alias.
            const float k = sky.cloudScale / (d.y + 0.08f);
            const float n = fbm(d.x * k, d.z * k, sky.seed);
            const float cover = smoothstep(1.0f - sky.clouds, 1.0f - sky.clouds + 0.3f, n) * smoothstep(0.0f, 0.2f, d.y);
            const glm::vec3 lit = sky.cloudColor * (0.72f + 0.28f * (1.0f - n)) * (0.85f + 0.35f * cosSun * cosSun);
            c = glm::mix(c, lit, cover * 0.9f);
        }
    } else {
        c = glm::mix(sky.horizon * 0.75f, sky.ground, smoothstep(0.0f, 0.12f, -d.y));
    }
    return c * sky.intensity;
}

glm::vec3 cubeFaceDirection(uint32_t face, float s, float t) {
    const float sc = s * 2.0f - 1.0f, tc = t * 2.0f - 1.0f;
    switch (face) {
    case 0: return glm::vec3(1.0f, -tc, -sc);
    case 1: return glm::vec3(-1.0f, -tc, sc);
    case 2: return glm::vec3(sc, 1.0f, tc);
    case 3: return glm::vec3(sc, -1.0f, -tc);
    case 4: return glm::vec3(sc, -tc, 1.0f);
    default: return glm::vec3(-sc, -tc, -1.0f);
    }
}

void cubeFaceCoordinates(const glm::vec3& d, uint32_t& face, float& s, float& t) {
    const glm::vec3 a = glm::abs(d);
    float sc, tc, ma;
    if (a.x >= a.y && a.x >= a.z) {
        face = d.x > 0.0f ? 0 : 1;
        ma = a.x;
        sc = d.x > 0.0f ? -d.z : d.z;
        tc = -d.y;
    } else if (a.y >= a.z) {
        face = d.y > 0.0f ? 2 : 3;
        ma = a.y;
        sc = d.x;
        tc = d.y > 0.0f ? d.z : -d.z;
    } else {
        face = d.z > 0.0f ? 4 : 5;
        ma = a.z;
        sc = d.z > 0.0f ? d.x : -d.x;
        tc = -d.y;
    }
    ma = std::max(ma, 1e-20f);
    s = 0.5f * (sc / ma + 1.0f);
    t = 0.5f * (tc / ma + 1.0f);
}

CubeFaces renderSkyCube(const SkySettings& sky, uint32_t faceSize) {
    CubeFaces faces;
    for (HdrImage& f : faces) f = HdrImage(faceSize, faceSize);
    parallelFor(6 * faceSize, [&](uint32_t row) {
        const uint32_t face = row / faceSize, y = row % faceSize;
        for (uint32_t x = 0; x < faceSize; ++x) {
            glm::vec3 sum(0.0f);
            for (uint32_t k = 0; k < 4; ++k) {
                const float s = (float(x) + 0.25f + 0.5f * float(k & 1)) / float(faceSize);
                const float t = (float(y) + 0.25f + 0.5f * float(k >> 1)) / float(faceSize);
                sum += skyRadiance(sky, cubeFaceDirection(face, s, t));
            }
            faces[face].at(x, y) = sum * 0.25f;
        }
    });
    return faces;
}

float environmentPhongExponent(uint32_t mip, uint32_t mips) {
    const float gloss = mips > 1 ? 1.0f - float(mip) / float(mips - 1) : 1.0f;
    return std::max(0.25f, std::exp2(1.0f + 10.0f * gloss) * 0.25f);
}

CubeMips prefilterCube(const CubeFaces& base, uint32_t mips) {
    std::vector<CubeFaces> pyramid{base};
    while (pyramid.back()[0].width > 1) pyramid.push_back(downsample(pyramid.back()));
    const uint32_t size = base[0].width;
    const float texelSolidAngle = 4.0f * kPi / (6.0f * float(size) * float(size));
    CubeMips out(mips);
    out[0] = base;
    constexpr uint32_t kSamples = 128;
    for (uint32_t m = 1; m < mips; ++m) {
        const uint32_t faceSize = std::max(1u, size >> m);
        const float n = environmentPhongExponent(m, mips);
        for (HdrImage& f : out[m]) f = HdrImage(faceSize, faceSize);
        parallelFor(6 * faceSize, [&](uint32_t row) {
            const uint32_t face = row / faceSize, y = row % faceSize;
            for (uint32_t x = 0; x < faceSize; ++x) {
                const glm::vec3 r = glm::normalize(cubeFaceDirection(face, (float(x) + 0.5f) / float(faceSize), (float(y) + 0.5f) / float(faceSize)));
                glm::vec3 t, b;
                basis(r, t, b);
                const float rotation = float(hash32(face * 7919u + y * 104729u + x) & 0xFFFFu) / 65536.0f;
                glm::vec3 sum(0.0f);
                for (uint32_t k = 0; k < kSamples; ++k) {
                    const float u1 = (float(k) + 0.5f) / float(kSamples);
                    const float u2 = std::fmod(radicalInverse(k) + rotation, 1.0f);
                    const float cosT = std::pow(u1, 1.0f / (n + 1.0f)), sinT = std::sqrt(std::max(0.0f, 1.0f - cosT * cosT));
                    const float phi = 2.0f * kPi * u2;
                    const glm::vec3 l = t * (sinT * std::cos(phi)) + b * (sinT * std::sin(phi)) + r * cosT;
                    const float pdf = (n + 1.0f) / (2.0f * kPi) * std::pow(cosT, n);
                    const float level = std::clamp(0.5f * std::log2(1.0f / (float(kSamples) * pdf * texelSolidAngle)) + 1.0f, 0.0f,
                                                   float(pyramid.size() - 1));
                    const auto l0 = uint32_t(level);
                    const uint32_t l1 = std::min(l0 + 1, uint32_t(pyramid.size() - 1));
                    sum += glm::mix(sampleCube(pyramid[l0], l), sampleCube(pyramid[l1], l), level - float(l0));
                }
                out[m][face].at(x, y) = sum / float(kSamples);
            }
        });
    }
    return out;
}

glm::vec3 skyIrradiance(const SkySettings& sky, const glm::vec3& normal) {
    glm::vec3 t, b;
    const glm::vec3 n = glm::normalize(normal);
    basis(n, t, b);
    constexpr uint32_t kSamples = 4096;
    glm::vec3 sum(0.0f);
    for (uint32_t k = 0; k < kSamples; ++k) {
        // Cosine-weighted: the average radiance is E/π.
        const float u1 = (float(k) + 0.5f) / float(kSamples), u2 = radicalInverse(k);
        const float r = std::sqrt(u1), phi = 2.0f * kPi * u2;
        const glm::vec3 d = t * (r * std::cos(phi)) + b * (r * std::sin(phi)) + n * std::sqrt(std::max(0.0f, 1.0f - u1));
        sum += skyRadiance(sky, d);
    }
    return sum / float(kSamples);
}

} // namespace tools
