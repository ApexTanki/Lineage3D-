#ifndef FL_POST_GLSL
#define FL_POST_GLSL

// Tone curves (M8 plan D5), mirrored on the CPU by renderer/post/ToneCurves.h (the tonemap test compares them).
// Input: linear scene color with exposure applied; output: linear display color in [0, 1]. Needs fl_bindings.glsl.

// Stephen Hill's fit of the ACES reference rendering transform and the sRGB output transform; the matrices convert
// between linear sRGB and the fit's working space (rows multiply the color).
vec3 flAces(vec3 c) {
    vec3 v = vec3(dot(vec3(0.59719, 0.35458, 0.04823), c), dot(vec3(0.07600, 0.90834, 0.01566), c),
                  dot(vec3(0.02840, 0.13383, 0.83777), c));
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    v = a / b;
    return clamp(vec3(dot(vec3(1.60475, -0.53108, -0.07367), v), dot(vec3(-0.10208, 1.10813, -0.00605), v),
                      dot(vec3(-0.00327, -0.07276, 1.07602), v)),
                 0.0, 1.0);
}

// John Hable's filmic curve (Uncharted 2): exposure bias 2, linear white point 11.2.
vec3 flHableCurve(vec3 x) {
    const float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
    return (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F) - E / F;
}

vec3 flHable(vec3 c) {
    return clamp(flHableCurve(2.0 * max(c, vec3(0.0))) / flHableCurve(vec3(11.2)), 0.0, 1.0);
}

// tonemapper: 0 ACES, 1 Hable, 2 clamp (renderer::PostSettings::Tonemapper).
vec3 flTonemap(vec3 c, float tonemapper) {
    if (tonemapper < 0.5) return flAces(c);
    if (tonemapper < 1.5) return flHable(c);
    return clamp(c, 0.0, 1.0);
}

// The forward shaders' last step (M8 plan D8): on the LDR path the scene target is sRGB8, so each surface applies
// exposure and the curve before it blends; on the HDR path the tonemap pass does it once for the frame.
vec3 flForwardOutput(vec3 color) {
    return frame.post.y > 0.5 ? flTonemap(color * frame.post.x, frame.post.z) : color;
}

// The same for blended surfaces. blend: 0 alpha, 1 additive, 2 premultiplied (as flApplyFog). A premultiplied color is
// the surface's color times its coverage: the curve applies to the color, so faint smoke isn't crushed into the toe.
vec3 flForwardOutputBlend(vec3 rgb, float alpha, float blend) {
    if (frame.post.y < 0.5) return rgb;
    if (blend > 1.5) return flTonemap(rgb / max(alpha, 1e-4) * frame.post.x, frame.post.z) * alpha;
    return flTonemap(rgb * frame.post.x, frame.post.z);
}

// Perceptual luma (Rec. 601 weights on linear color).
float flLuma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

// The largest finite half float: post targets in RGBA16F (the fallback for R11G11B10F, R9) hold nothing larger.
const float kFlMaxHalf = 65504.0;

// A scene color as a post filter reads it: an RGBA16F target turns values past the largest half float into +Inf, and a
// single Inf times a zero weight is a NaN that a blur spreads into a black block (bloom, god rays, depth of field). Clamp
// to the largest half float and drop NaNs.
vec3 flFiniteColor(vec3 c) {
    return any(isnan(c)) ? vec3(0.0) : min(c, vec3(kFlMaxHalf));
}

#endif
