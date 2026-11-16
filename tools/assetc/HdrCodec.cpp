#include "HdrCodec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace tools {
namespace {

constexpr int kWeights[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

// The one-region BC6H modes. After the 5 mode bits come the low 10 bits of the base endpoint W (R, G, B); then, per
// channel, 10 positions hold X bits 0..xBits−1 followed by W bits epb−1 down to 10 (stored highest first). Untransformed
// mode 11 stores the second endpoint in X; the others store a signed delta from W. Index bits start at 65; texel 0 is
// the anchor and stores 3 bits (its index is always below 8).
struct Mode {
    uint8_t value;    // the 5 mode bits
    int epb;          // endpoint precision
    bool transformed; // X is a delta from W
    int xBits;
};
constexpr Mode kModes[4] = {{0x03, 10, false, 10}, {0x07, 11, true, 9}, {0x0B, 12, true, 8}, {0x0F, 16, true, 4}};

struct BitStream {
    uint8_t* data;
    void set(int pos, uint32_t bit) const {
        if (bit & 1u) data[pos >> 3] = uint8_t(data[pos >> 3] | (1u << (pos & 7)));
    }
    uint32_t get(int pos) const { return (uint32_t(data[pos >> 3]) >> (pos & 7)) & 1u; }
};

// The decode pipeline of the unsigned format: unquantize the endpoints to 16 bits, interpolate with 6-bit weights,
// then scale by 31/64 into half-float bits.
int unquantize(int comp, int epb) {
    if (epb >= 15) return comp;
    if (comp == 0) return 0;
    if (comp == (1 << epb) - 1) return 0xFFFF;
    return ((comp << 16) + 0x8000) >> epb;
}
int interpolate(int e0, int e1, int weight) {
    return (e0 * (64 - weight) + e1 * weight + 32) >> 6;
}
int finish(int c) {
    return (c * 31) >> 6;
}

int quantizeComponent(float value, int epb) {
    const int maxValue = (1 << epb) - 1;
    int q = epb >= 15 ? int(std::lround(value)) : int(std::lround(value / float(1 << (16 - epb)) - 0.5f));
    q = std::clamp(q, 0, maxValue);
    int best = q;
    float bestDistance = std::fabs(float(unquantize(q, epb)) - value);
    for (int candidate : {q - 1, q + 1}) {
        if (candidate < 0 || candidate > maxValue) continue;
        const float d = std::fabs(float(unquantize(candidate, epb)) - value);
        if (d < bestDistance) best = candidate, bestDistance = d;
    }
    return best;
}

struct Candidate {
    int q0[3] = {};
    int q1[3] = {};
    uint8_t index[16] = {};
    int64_t error = std::numeric_limits<int64_t>::max();
};

// Exact decode of (q0, q1) for this mode, best index per texel, total squared error in half-bit units.
int64_t evaluate(const Mode& mode, Candidate& c, const int target[16][3]) {
    int palette[16][3];
    for (int ch = 0; ch < 3; ++ch) {
        const int e0 = unquantize(c.q0[ch], mode.epb), e1 = unquantize(c.q1[ch], mode.epb);
        for (int i = 0; i < 16; ++i) palette[i][ch] = finish(interpolate(e0, e1, kWeights[i]));
    }
    int64_t total = 0;
    for (int t = 0; t < 16; ++t) {
        int64_t best = std::numeric_limits<int64_t>::max();
        for (int i = 0; i < 16; ++i) {
            int64_t e = 0;
            for (int ch = 0; ch < 3; ++ch) {
                const int64_t d = palette[i][ch] - target[t][ch];
                e += d * d;
            }
            if (e < best) best = e, c.index[t] = uint8_t(i);
        }
        total += best;
    }
    return total;
}

// Quantizes float endpoints (interpolation domain, 0–65535) for `mode`; false if the delta can't hold them.
void quantize(const Mode& mode, const float e0[3], const float e1[3], Candidate& c) {
    const int lo = -(1 << (mode.xBits - 1)), hi = (1 << (mode.xBits - 1)) - 1;
    for (int ch = 0; ch < 3; ++ch) {
        c.q0[ch] = quantizeComponent(e0[ch], mode.epb);
        const int q1 = quantizeComponent(e1[ch], mode.epb);
        // A delta that doesn't fit is clamped toward the base; the error decides whether the mode is any good.
        c.q1[ch] = mode.transformed ? c.q0[ch] + std::clamp(q1 - c.q0[ch], lo, hi) : q1;
    }
}

// Makes texel 0's index fit its 3 bits by swapping the endpoints, which mirrors every index exactly (the weights are
// symmetric). Returns false when a transformed mode's delta can't be negated (only the most negative delta).
bool fixAnchor(const Mode& mode, Candidate& c) {
    if (c.index[0] < 8) return true;
    if (mode.transformed) {
        const int hi = (1 << (mode.xBits - 1)) - 1;
        for (int ch = 0; ch < 3; ++ch)
            if (c.q0[ch] - c.q1[ch] > hi) return false;
    }
    for (int ch = 0; ch < 3; ++ch) std::swap(c.q0[ch], c.q1[ch]);
    for (uint8_t& i : c.index) i = uint8_t(15 - i);
    return true;
}

void pack(const Mode& mode, const Candidate& c, uint8_t out[16]) {
    std::memset(out, 0, 16);
    const BitStream bits{out};
    for (int i = 0; i < 5; ++i) bits.set(i, uint32_t(mode.value >> i));
    int pos = 5;
    for (int ch = 0; ch < 3; ++ch)
        for (int i = 0; i < 10; ++i) bits.set(pos++, uint32_t(c.q0[ch] >> i));
    for (int ch = 0; ch < 3; ++ch) {
        const int x = mode.transformed ? (c.q1[ch] - c.q0[ch]) & ((1 << mode.xBits) - 1) : c.q1[ch];
        for (int i = 0; i < mode.xBits; ++i) bits.set(pos++, uint32_t(x >> i));
        for (int bit = mode.epb - 1; bit >= 10; --bit) bits.set(pos++, uint32_t(c.q0[ch] >> bit));
    }
    for (int i = 0; i < 3; ++i) bits.set(65 + i, uint32_t(c.index[0] >> i));
    for (int t = 1; t < 16; ++t)
        for (int i = 0; i < 4; ++i) bits.set(68 + (t - 1) * 4 + i, uint32_t(c.index[t] >> i));
}

// Least-squares endpoints for fixed indices, per channel, in the interpolation domain.
bool refine(const Candidate& c, const float x[16][3], float e0[3], float e1[3]) {
    double a00 = 0, a01 = 0, a11 = 0, b0[3] = {}, b1[3] = {};
    for (int t = 0; t < 16; ++t) {
        const double a = kWeights[c.index[t]] / 64.0, b = 1.0 - a;
        a00 += b * b;
        a01 += a * b;
        a11 += a * a;
        for (int ch = 0; ch < 3; ++ch) {
            b0[ch] += b * x[t][ch];
            b1[ch] += a * x[t][ch];
        }
    }
    const double det = a00 * a11 - a01 * a01;
    if (std::fabs(det) < 1e-9) return false;
    for (int ch = 0; ch < 3; ++ch) {
        e0[ch] = float(std::clamp((b0[ch] * a11 - b1[ch] * a01) / det, 0.0, 65535.0));
        e1[ch] = float(std::clamp((b1[ch] * a00 - b0[ch] * a01) / det, 0.0, 65535.0));
    }
    return true;
}

} // namespace

uint16_t floatToHalf(float value) {
    uint32_t x;
    std::memcpy(&x, &value, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exponent = (x >> 23) & 0xFFu;
    uint32_t mantissa = x & 0x007FFFFFu;
    if (exponent == 0xFF) return uint16_t(sign | (mantissa ? 0x7E00u : 0x7C00u));
    const int e = int(exponent) - 127 + 15;
    if (e >= 31) return uint16_t(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return uint16_t(sign);
        mantissa |= 0x00800000u;
        const uint32_t shift = uint32_t(14 - e);
        uint32_t half = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        if (rest > halfway || (rest == halfway && (half & 1u))) ++half;
        return uint16_t(sign | half);
    }
    uint32_t half = sign | (uint32_t(e) << 10) | (mantissa >> 13);
    const uint32_t rest = mantissa & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) ++half; // a carry into the exponent is correct rounding
    return uint16_t(half);
}

float halfToFloat(uint16_t half) {
    const uint32_t sign = uint32_t(half & 0x8000u) << 16;
    const uint32_t exponent = (half >> 10) & 0x1Fu;
    const uint32_t mantissa = half & 0x3FFu;
    uint32_t x;
    if (exponent == 0) {
        if (mantissa == 0) {
            x = sign;
        } else {
            const float f = float(mantissa) * (1.0f / 16777216.0f); // subnormal: m · 2^−24
            return sign ? -f : f;
        }
    } else if (exponent == 31) {
        x = sign | 0x7F800000u | (mantissa << 13);
    } else {
        x = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

void encodeBc6hBlock(const glm::vec3 texels[16], uint8_t out[16]) {
    int target[16][3];
    float x[16][3];
    float mean[3] = {};
    for (int t = 0; t < 16; ++t)
        for (int ch = 0; ch < 3; ++ch) {
            float v = texels[t][ch];
            v = std::isfinite(v) ? std::clamp(v, 0.0f, 65504.0f) : 0.0f;
            target[t][ch] = floatToHalf(v);
            x[t][ch] = std::min(65535.0f, float(target[t][ch]) * (64.0f / 31.0f));
            mean[ch] += x[t][ch] / 16.0f;
        }

    // Principal axis of the block in the interpolation domain (power iteration on the covariance).
    double cov[3][3] = {};
    for (int t = 0; t < 16; ++t)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) cov[i][j] += double(x[t][i] - mean[i]) * double(x[t][j] - mean[j]);
    double axis[3] = {1.0, 1.0, 1.0};
    for (int iteration = 0; iteration < 8; ++iteration) {
        double next[3] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) next[i] += cov[i][j] * axis[j];
        const double len = std::sqrt(next[0] * next[0] + next[1] * next[1] + next[2] * next[2]);
        if (len < 1e-12) break;
        for (int i = 0; i < 3; ++i) axis[i] = next[i] / len;
    }
    const double axisLength = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    for (double& a : axis) a /= axisLength;
    double tMin = 0.0, tMax = 0.0;
    for (int t = 0; t < 16; ++t) {
        const double p = (x[t][0] - mean[0]) * axis[0] + (x[t][1] - mean[1]) * axis[1] + (x[t][2] - mean[2]) * axis[2];
        tMin = std::min(tMin, p);
        tMax = std::max(tMax, p);
    }
    float e0[3], e1[3];
    for (int ch = 0; ch < 3; ++ch) {
        e0[ch] = float(std::clamp(mean[ch] + tMin * axis[ch], 0.0, 65535.0));
        e1[ch] = float(std::clamp(mean[ch] + tMax * axis[ch], 0.0, 65535.0));
    }

    // A second starting line: the per-channel bounding box (its diagonal follows blocks whose channels change together).
    float b0[3], b1[3];
    for (int ch = 0; ch < 3; ++ch) {
        b0[ch] = 65535.0f;
        b1[ch] = 0.0f;
        for (int t = 0; t < 16; ++t) {
            b0[ch] = std::min(b0[ch], x[t][ch]);
            b1[ch] = std::max(b1[ch], x[t][ch]);
        }
    }

    const Mode* bestMode = nullptr;
    Candidate best;
    for (const Mode& mode : kModes)
        for (int start = 0; start < 2; ++start) {
            float a[3], b[3];
            std::copy(start ? b0 : e0, (start ? b0 : e0) + 3, a);
            std::copy(start ? b1 : e1, (start ? b1 : e1) + 3, b);
            for (int pass = 0; pass < 4; ++pass) { // least-squares refinement converges in a few passes
                Candidate c;
                quantize(mode, a, b, c);
                c.error = evaluate(mode, c, target);
                if (fixAnchor(mode, c) && (!bestMode || c.error < best.error)) {
                    best = c;
                    bestMode = &mode;
                }
                if (pass == 3 || !refine(c, x, a, b)) break;
            }
        }
    pack(*bestMode, best, out); // mode 11 is untransformed, so its anchor fix never fails: bestMode is always set
}

bool decodeBc6hBlock(const uint8_t block[16], glm::vec3 out[16]) {
    const BitStream bits{const_cast<uint8_t*>(block)};
    uint32_t value = 0;
    for (int i = 0; i < 5; ++i) value |= bits.get(i) << i;
    const Mode* mode = nullptr;
    for (const Mode& m : kModes)
        if (m.value == value) mode = &m;
    if (!mode) return false;
    int w[3] = {}, x[3] = {};
    int pos = 5;
    for (int ch = 0; ch < 3; ++ch)
        for (int i = 0; i < 10; ++i) w[ch] |= int(bits.get(pos++)) << i;
    for (int ch = 0; ch < 3; ++ch) {
        for (int i = 0; i < mode->xBits; ++i) x[ch] |= int(bits.get(pos++)) << i;
        for (int bit = mode->epb - 1; bit >= 10; --bit) w[ch] |= int(bits.get(pos++)) << bit;
    }
    int e0[3], e1[3];
    for (int ch = 0; ch < 3; ++ch) {
        int second = x[ch];
        if (mode->transformed) {
            if (second & (1 << (mode->xBits - 1))) second -= 1 << mode->xBits; // deltas are signed
            second = (w[ch] + second) & ((1 << mode->epb) - 1);
        }
        e0[ch] = unquantize(w[ch], mode->epb);
        e1[ch] = unquantize(second, mode->epb);
    }
    for (int t = 0; t < 16; ++t) {
        int index = 0;
        if (t == 0) {
            for (int i = 0; i < 3; ++i) index |= int(bits.get(65 + i)) << i;
        } else {
            for (int i = 0; i < 4; ++i) index |= int(bits.get(68 + (t - 1) * 4 + i)) << i;
        }
        for (int ch = 0; ch < 3; ++ch) out[t][ch] = halfToFloat(uint16_t(finish(interpolate(e0[ch], e1[ch], kWeights[index]))));
    }
    return true;
}

std::vector<uint8_t> encodeBc6hImage(const HdrImage& image) {
    const uint32_t bw = (image.width + 3) / 4, bh = (image.height + 3) / 4;
    std::vector<uint8_t> blocks(size_t(bw) * bh * 16);
    glm::vec3 texels[16];
    for (uint32_t by = 0; by < bh; ++by)
        for (uint32_t bx = 0; bx < bw; ++bx) {
            for (uint32_t t = 0; t < 16; ++t) {
                const uint32_t px = std::min(image.width - 1, bx * 4 + t % 4), py = std::min(image.height - 1, by * 4 + t / 4);
                texels[t] = image.at(px, py);
            }
            encodeBc6hBlock(texels, &blocks[(size_t(by) * bw + bx) * 16]);
        }
    return blocks;
}

HdrImage decodeBc6hImage(const std::vector<uint8_t>& blocks, uint32_t width, uint32_t height) {
    HdrImage image(width, height);
    const uint32_t bw = (width + 3) / 4, bh = (height + 3) / 4;
    glm::vec3 texels[16];
    for (uint32_t by = 0; by < bh; ++by)
        for (uint32_t bx = 0; bx < bw; ++bx) {
            const size_t offset = (size_t(by) * bw + bx) * 16;
            if (offset + 16 > blocks.size() || !decodeBc6hBlock(&blocks[offset], texels)) continue;
            for (uint32_t t = 0; t < 16; ++t) {
                const uint32_t px = bx * 4 + t % 4, py = by * 4 + t / 4;
                if (px < width && py < height) image.at(px, py) = texels[t];
            }
        }
    return image;
}

void encodeRgbm(const glm::vec3& color, uint8_t out[4]) {
    glm::vec3 c;
    for (int i = 0; i < 3; ++i) c[i] = std::isfinite(color[i]) ? std::clamp(color[i], 0.0f, kRgbmRange) : 0.0f;
    const float m = std::max(c.r, std::max(c.g, c.b)) / kRgbmRange;
    const int a = std::clamp(int(std::ceil(m * 255.0f)), 1, 255);
    const float scale = 255.0f / (float(a) / 255.0f * kRgbmRange);
    for (int i = 0; i < 3; ++i) out[i] = uint8_t(std::clamp(std::lround(c[i] * scale), 0l, 255l));
    out[3] = uint8_t(a);
}

glm::vec3 decodeRgbm(const uint8_t rgba[4]) {
    const float m = float(rgba[3]) / 255.0f * kRgbmRange;
    return glm::vec3(rgba[0], rgba[1], rgba[2]) / 255.0f * m;
}

} // namespace tools
