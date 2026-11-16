#pragma once

#include "core/Math.h"
#include "renderer/post/PostSettings.h"

#include <algorithm>
#include <cmath>

// CPU mirror of the tone curves in shaders/include/fl_post.glsl (M8 plan D5): the unit tests and --tonemap-test
// compare the GPU's output against these. Both take linear scene color (exposure already applied) and return linear
// display color in [0, 1].
namespace renderer::tone {

// Stephen Hill's fit of the ACES reference rendering transform and the sRGB output transform. The input and output
// matrices convert between linear sRGB and the fit's working space; rows multiply the color.
inline glm::vec3 aces(glm::vec3 c) {
    const glm::vec3 in = {0.59719f * c.r + 0.35458f * c.g + 0.04823f * c.b, 0.07600f * c.r + 0.90834f * c.g + 0.01566f * c.b,
                          0.02840f * c.r + 0.13383f * c.g + 0.83777f * c.b};
    const glm::vec3 a = in * (in + 0.0245786f) - 0.000090537f;
    const glm::vec3 b = in * (0.983729f * in + 0.4329510f) + 0.238081f;
    const glm::vec3 v = a / b;
    const glm::vec3 out = {1.60475f * v.r - 0.53108f * v.g - 0.07367f * v.b, -0.10208f * v.r + 1.10813f * v.g - 0.00605f * v.b,
                           -0.00327f * v.r - 0.07276f * v.g + 1.07602f * v.b};
    return glm::clamp(out, glm::vec3(0.0f), glm::vec3(1.0f));
}

// John Hable's filmic curve (Uncharted 2): exposure bias 2, linear white point 11.2.
inline glm::vec3 hableCurve(glm::vec3 x) {
    constexpr float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
    return (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F) - E / F;
}

inline glm::vec3 hable(glm::vec3 c) {
    const glm::vec3 white = hableCurve(glm::vec3(11.2f));
    return glm::clamp(hableCurve(2.0f * glm::max(c, glm::vec3(0.0f))) / white, glm::vec3(0.0f), glm::vec3(1.0f));
}

inline glm::vec3 apply(PostSettings::Tonemapper tonemapper, glm::vec3 c) {
    switch (tonemapper) {
    case PostSettings::Tonemapper::Aces: return aces(c);
    case PostSettings::Tonemapper::Hable: return hable(c);
    case PostSettings::Tonemapper::Clamp: break;
    }
    return glm::clamp(c, glm::vec3(0.0f), glm::vec3(1.0f));
}

// The display encoding (IEC 61966-2-1), as the output stage applies it.
inline float srgbEncode(float linear) {
    linear = std::clamp(linear, 0.0f, 1.0f);
    return linear <= 0.0031308f ? linear * 12.92f : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
}

} // namespace renderer::tone
