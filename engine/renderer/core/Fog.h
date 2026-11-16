#pragma once

#include "assets/MapFormat.h"

#include <algorithm>
#include <cmath>

namespace renderer {

using FogSettings = assets::FogSettings;

// CPU mirror of shaders/include/fl_fog.glsl (M3 plan D10), for unit tests and CPU-side effects (particles, M7).

// ∫ ρ0·e^(−k(y_c + t·Δy/d − base)) dt over [0, d].
inline float heightFogIntegral(const FogSettings& f, float distance, float cameraY, float deltaY) {
    const float x = f.heightFalloff * deltaY;
    const float ratio = std::fabs(x) > 1e-4f ? (1.0f - std::exp(-x)) / x : 1.0f - 0.5f * x;
    return f.heightDensity * std::exp(-f.heightFalloff * (cameraY - f.heightBase)) * distance * ratio;
}

inline float fogFactor(const FogSettings& f, const glm::vec3& camera, const glm::vec3& point) {
    if (f.mode == FogSettings::Mode::Off && !f.height) return 0.0f;
    const glm::vec3 d = point - camera;
    const float distance = glm::length(d);
    float result = 0.0f;
    switch (f.mode) {
    case FogSettings::Mode::Linear: result = std::clamp((distance - f.start) / std::max(f.end - f.start, 1e-4f), 0.0f, 1.0f); break;
    case FogSettings::Mode::Exp: result = 1.0f - std::exp(-f.density * distance); break;
    case FogSettings::Mode::Exp2: result = 1.0f - std::exp(-(f.density * distance) * (f.density * distance)); break;
    case FogSettings::Mode::Off: break;
    }
    if (f.height) result = 1.0f - (1.0f - result) * std::exp(-heightFogIntegral(f, distance, camera.y, d.y));
    return std::min(result, f.maxOpacity);
}

inline float skyFogFactor(const FogSettings& f, float cameraY, const glm::vec3& direction) {
    if (f.mode == FogSettings::Mode::Off && !f.height) return 0.0f;
    float result = 0.0f;
    if (f.mode != FogSettings::Mode::Off) {
        const float t = std::clamp(direction.y / std::max(f.skyHorizon, 1e-4f), 0.0f, 1.0f);
        result = 1.0f - t * t * (3.0f - 2.0f * t);
    }
    if (f.height) {
        const float h = direction.y > 1e-3f && f.heightFalloff > 1e-6f
                            ? f.heightDensity * std::exp(-f.heightFalloff * (cameraY - f.heightBase)) / (f.heightFalloff * direction.y)
                            : 1e4f;
        result = 1.0f - (1.0f - result) * std::exp(-h);
    }
    return std::min(result, f.maxOpacity);
}

} // namespace renderer
