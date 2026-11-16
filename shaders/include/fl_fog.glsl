#ifndef FL_FOG_GLSL
#define FL_FOG_GLSL

// Distance and height fog (M3 plan D10), a uniform branch on FrameUniforms. Mirrored on the CPU by renderer/core/Fog.h
// (unit tests). Distance fog: linear between start and end, 1 − e^(−density·d) or 1 − e^(−(density·d)²). Height fog:
// density ρ(y) = ρ0·e^(−k·(y − base)) integrated along the view ray in closed form. Combined as 1 − (1 − f_d)(1 − f_h)
// and clamped by the max opacity.

float flHeightFogIntegral(float distance, float cameraY, float deltaY) {
    // ∫ ρ0·e^(−k(y_c + t·Δy/d − base)) dt over [0, d] = ρ0·e^(−k(y_c − base))·d·(1 − e^(−kΔy))/(kΔy)
    float k = frame.fogHeight.z;
    float x = k * deltaY;
    float ratio = abs(x) > 1e-4 ? (1.0 - exp(-x)) / x : 1.0 - 0.5 * x;
    return frame.fogHeight.y * exp(-k * (cameraY - frame.fogHeight.x)) * distance * ratio;
}

float flFogFactor(vec3 worldPosition) {
    int mode = int(frame.fogColor.w + 0.5);
    bool height = frame.fogHeight.w > 0.5;
    if (mode == 0 && !height) return 0.0;
    vec3 toPoint = worldPosition - frame.cameraPosition.xyz;
    float d = length(toPoint);
    float f = 0.0;
    if (mode == 1) f = clamp((d - frame.fogParams.x) / max(frame.fogParams.y - frame.fogParams.x, 1e-4), 0.0, 1.0);
    else if (mode == 2) f = 1.0 - exp(-frame.fogParams.z * d);
    else if (mode == 3) f = 1.0 - exp(-(frame.fogParams.z * d) * (frame.fogParams.z * d));
    if (height) f = 1.0 - (1.0 - f) * exp(-flHeightFogIntegral(d, frame.cameraPosition.y, toPoint.y));
    return min(f, frame.fogParams.w);
}

// The sky is infinitely far: distance fog covers a band above the horizon (and everything below it), height fog is
// integrated to infinity, which is finite only for rays going up.
float flSkyFogFactor(vec3 direction) {
    int mode = int(frame.fogColor.w + 0.5);
    bool height = frame.fogHeight.w > 0.5;
    if (mode == 0 && !height) return 0.0;
    float f = mode == 0 ? 0.0 : 1.0 - smoothstep(0.0, max(frame.fogSky.x, 1e-4), direction.y);
    if (height) {
        float k = frame.fogHeight.z;
        float h = direction.y > 1e-3 && k > 1e-6 ? frame.fogHeight.y * exp(-k * (frame.cameraPosition.y - frame.fogHeight.x)) / (k * direction.y)
                                                 : 1e4;
        f = 1.0 - (1.0 - f) * exp(-h);
    }
    return min(f, frame.fogParams.w);
}

// `blend`: 0 opaque / alpha blend (mix toward the fog color), 1 additive (fade out), 2 premultiplied (fog × alpha).
vec3 flApplyFog(vec3 color, float alpha, vec3 worldPosition, float blend) {
    float f = flFogFactor(worldPosition);
    if (blend > 1.5) return mix(color, frame.fogColor.rgb * alpha, f);
    if (blend > 0.5) return color * (1.0 - f);
    return mix(color, frame.fogColor.rgb, f);
}

#endif
