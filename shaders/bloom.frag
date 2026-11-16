#version 450
// Pass 9c (M8 plan D6): Jimenez's bloom chain. PREFILTER downsamples the scene into mip 0 with a Karis-weighted 13-tap
// filter (no single bright texel flickers) and keeps what lies above the soft-knee threshold, with exposure applied;
// DOWN is the plain 13-tap downsample of the mip above; UP is a 3×3 tent of the mip below, added (blend One/One) into
// the mip it is drawn into.
//   MODE  PREFILTER | DOWN | UP
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_post.glsl"

layout(set = 1, binding = 1) uniform sampler2D u_input;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

vec3 tap(vec2 uv) {
#if defined(MODE_PREFILTER)
    return flFiniteColor(textureLod(u_input, uv, 0.0).rgb); // the scene: an RGBA16F target may hold Inf
#else
    return textureLod(u_input, uv, 0.0).rgb; // the chain: finite since the prefilter
#endif
}

#if defined(MODE_UP)
// Per-draw: modelRows[0].x = tent radius in source texels.
void main() {
    vec2 r = pd.modelRows[0].x / vec2(textureSize(u_input, 0));
    vec3 c = tap(v_uv + vec2(-r.x, -r.y)) + 2.0 * tap(v_uv + vec2(0.0, -r.y)) + tap(v_uv + vec2(r.x, -r.y)) +
             2.0 * tap(v_uv + vec2(-r.x, 0.0)) + 4.0 * tap(v_uv) + 2.0 * tap(v_uv + vec2(r.x, 0.0)) +
             tap(v_uv + vec2(-r.x, r.y)) + 2.0 * tap(v_uv + vec2(0.0, r.y)) + tap(v_uv + vec2(r.x, r.y));
    o_color = vec4(c * (1.0 / 16.0), 1.0);
}
#else
float karisWeight(vec3 c) {
    return 1.0 / (1.0 + flLuma(c));
}

// Per-draw (PREFILTER): modelRows[0] = (threshold, knee width, exposure, unused).
void main() {
    vec2 t = 1.0 / vec2(textureSize(u_input, 0)); // source texels: the target is half the size
    vec3 a = tap(v_uv + t * vec2(-2.0, -2.0)), b = tap(v_uv + t * vec2(0.0, -2.0)), c = tap(v_uv + t * vec2(2.0, -2.0));
    vec3 d = tap(v_uv + t * vec2(-1.0, -1.0)), e = tap(v_uv + t * vec2(1.0, -1.0));
    vec3 f = tap(v_uv + t * vec2(-2.0, 0.0)), g = tap(v_uv), h = tap(v_uv + t * vec2(2.0, 0.0));
    vec3 i = tap(v_uv + t * vec2(-1.0, 1.0)), j = tap(v_uv + t * vec2(1.0, 1.0));
    vec3 k = tap(v_uv + t * vec2(-2.0, 2.0)), l = tap(v_uv + t * vec2(0.0, 2.0)), m = tap(v_uv + t * vec2(2.0, 2.0));
    // Five overlapping 2×2 groups: the inner one weighs 0.5, each corner one 0.125.
    vec3 g0 = (d + e + i + j) * 0.25, g1 = (a + b + f + g) * 0.25, g2 = (b + c + g + h) * 0.25;
    vec3 g3 = (f + g + k + l) * 0.25, g4 = (g + h + l + m) * 0.25;
#if defined(MODE_PREFILTER)
    vec4 p = pd.modelRows[0];
    g0 *= p.z; g1 *= p.z; g2 *= p.z; g3 *= p.z; g4 *= p.z;
    float w0 = 0.5 * karisWeight(g0), w1 = 0.125 * karisWeight(g1), w2 = 0.125 * karisWeight(g2);
    float w3 = 0.125 * karisWeight(g3), w4 = 0.125 * karisWeight(g4);
    vec3 color = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    // Soft knee: a quadratic ramp over [threshold − knee, threshold + knee], linear above.
    float brightness = max(color.r, max(color.g, color.b));
    float knee = max(p.y, 1e-4);
    float soft = clamp(brightness - p.x + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    color *= max(soft, brightness - p.x) / max(brightness, 1e-4);
    color = min(color, vec3(kFlMaxHalf)); // exposure may push it past what an RGBA16F mip holds
#else
    vec3 color = g0 * 0.5 + (g1 + g2 + g3 + g4) * 0.125;
#endif
    o_color = vec4(color, 1.0);
}
#endif
