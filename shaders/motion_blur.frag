#version 450
// Camera motion blur (P2, M8 plan D12), before bloom and the tonemap: each pixel's world position, rebuilt from linear
// depth along its view ray, projected with the previous frame's view-projection gives the screen motion since then; 8
// taps along it (centered, scaled by the shutter fraction, clamped) average the color. Camera motion only: objects
// moving on their own don't blur.
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_post.glsl"

layout(set = 1, binding = 1) uniform sampler2D u_color;
layout(set = 1, binding = 2) uniform sampler2D u_depth; // linear view depth in meters

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

// Per-draw: the previous frame's view-projection as four rows (modelRows[0..2], lightmapST), and
// sh[0] = (shutter fraction, longest blur as a fraction of the screen, unused, unused).
void main() {
    mat4 previous = transpose(mat4(pd.modelRows[0], pd.modelRows[1], pd.modelRows[2], pd.lightmapST));
    vec4 h = frame.skyFromClip * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 dir = h.xyz / h.w;
    float depth = textureLod(u_depth, v_uv, 0.0).r;
    vec3 world = frame.cameraPosition.xyz + dir * (depth / dot(dir, frame.cameraForward.xyz));
    vec4 clip = previous * vec4(world, 1.0);
    vec2 motion = clip.w > 0.0 ? v_uv - (clip.xy / clip.w * 0.5 + 0.5) : vec2(0.0);
    motion *= pd.sh[0].x;
    float len = length(motion);
    if (len > pd.sh[0].y) motion *= pd.sh[0].y / len;
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 8; ++i) sum += flFiniteColor(textureLod(u_color, v_uv + motion * (float(i) / 7.0 - 0.5), 0.0).rgb);
    o_color = vec4(sum * 0.125, 1.0);
}
