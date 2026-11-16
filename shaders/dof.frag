#version 450
// Depth of field (P2, M8 plan D12), before bloom and the tonemap:
//   MODE  COC        half resolution: the scene color and its circle of confusion (alpha) from linear depth
//         BLUR       half resolution: a 16-tap golden-angle disk around each texel, sized by its CoC; a tap counts by its
//                    own CoC, so sharp foreground objects don't smear into the blurred background
//         COMPOSITE  full resolution: the scene mixed toward the blur by the full-resolution CoC
// CoC = clamp(|depth − focus| / range, 0, 1); both nearer and farther than the focus blur.
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_post.glsl"

layout(set = 1, binding = 1) uniform sampler2D u_color; // COC, COMPOSITE: the scene; BLUR: the half-res color + CoC
layout(set = 1, binding = 2) uniform sampler2D u_depth; // linear view depth in meters (COC, COMPOSITE)
layout(set = 1, binding = 3) uniform sampler2D u_blur;  // COMPOSITE: the blurred half-res color

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

// Per-draw: modelRows[0] = (focus distance m, focus range m, max blur radius in half-res texels, unused).
float coc(float depth) {
    return clamp(abs(depth - pd.modelRows[0].x) / max(pd.modelRows[0].y, 1e-3), 0.0, 1.0);
}

void main() {
#if defined(MODE_COC)
    // The 2×2 block's average color (one bilinear tap); its CoC from the block's farthest-from-focus depth, so edges blur
    // generously rather than leave sharp halos.
    vec3 color = flFiniteColor(textureLod(u_color, v_uv, 0.0).rgb); // the scene: an RGBA16F target may hold Inf
    vec2 texel = 1.0 / vec2(textureSize(u_depth, 0));
    float c = 0.0;
    for (int i = 0; i < 4; ++i) {
        vec2 offset = (vec2(i & 1, i >> 1) - 0.5) * texel;
        c = max(c, coc(textureLod(u_depth, v_uv + offset, 0.0).r));
    }
    o_color = vec4(color, c);
#elif defined(MODE_BLUR)
    vec4 center = textureLod(u_color, v_uv, 0.0);
    vec2 texel = 1.0 / vec2(textureSize(u_color, 0));
    float radius = center.a * pd.modelRows[0].z;
    vec3 sum = center.rgb;
    float total = 1.0;
    for (int i = 1; i < 16; ++i) {
        float r = sqrt(float(i) / 15.0) * radius;
        float a = float(i) * 2.39996323; // the golden angle
        vec4 s = textureLod(u_color, v_uv + vec2(cos(a), sin(a)) * r * texel, 0.0);
        float w = smoothstep(0.0, 1.0, s.a * 4.0); // in-focus taps hardly count
        sum += s.rgb * w;
        total += w;
    }
    o_color = vec4(sum / total, center.a);
#else
    vec3 sharp = flFiniteColor(textureLod(u_color, v_uv, 0.0).rgb);
    vec3 blurred = textureLod(u_blur, v_uv, 0.0).rgb;
    float c = coc(textureLod(u_depth, v_uv, 0.0).r);
    o_color = vec4(mix(sharp, blurred, smoothstep(0.0, 1.0, c)), 1.0);
#endif
}
