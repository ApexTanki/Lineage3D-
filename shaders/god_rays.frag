#version 450
// God rays (P2 "cheap volumetric light", M8 plan D12), at half resolution; the tonemap adds the result:
//   MODE  MASK  the sky's color where depth reaches the far plane, weighted toward the sun's screen position
//         BLUR  32 taps from each texel toward the sun, decaying with distance, so occluders cast shafts of shadow
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_post.glsl"

layout(set = 1, binding = 1) uniform sampler2D u_input; // MASK: the scene color; BLUR: the mask
layout(set = 1, binding = 2) uniform sampler2D u_depth; // MASK: linear view depth in meters

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

// Per-draw: modelRows[0] = (sun screen UV xy, far plane m, fade: 0 off-screen … 1 on screen).
void main() {
    vec4 p = pd.modelRows[0];
#ifdef MODE_MASK
    vec2 texel = 1.0 / vec2(textureSize(u_depth, 0));
    float sky = 0.0;
    for (int i = 0; i < 4; ++i) // the 2×2 block's share of sky
        sky += textureLod(u_depth, v_uv + (vec2(i & 1, i >> 1) - 0.5) * texel, 0.0).r >= p.z * 0.999 ? 0.25 : 0.0;
    vec3 color = flFiniteColor(textureLod(u_input, v_uv, 0.0).rgb) * frame.post.x; // exposed, like the bloom
    float nearSun = 1.0 - smoothstep(0.0, 0.35, length((v_uv - p.xy) * vec2(frame.screen.x * frame.screen.w, 1.0)));
    // Squared: the shafts come from the bright core. Clamped: an RGBA16F mask holds nothing past the largest half float.
    o_color = vec4(min(color * sky * (nearSun * nearSun) * p.w, vec3(kFlMaxHalf)), 1.0);
#else
    vec2 step = (p.xy - v_uv) / 32.0;
    vec2 uv = v_uv;
    float decay = 1.0;
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 32; ++i) {
        sum += textureLod(u_input, uv, 0.0).rgb * decay;
        uv += step;
        decay *= 0.96;
    }
    o_color = vec4(sum / 32.0, 1.0);
#endif
}
