#version 450
// Pass 9d (M8 plan D5): scene color × exposure, plus the bloom chain and the god rays, through the tone curve. Writes
// the backbuffer (linear; flEncodeOutput encodes where the backbuffer doesn't), or the sRGB8 ldrColor target that FXAA
// reads next (its format encodes).
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_post.glsl"

layout(set = 1, binding = 1) uniform sampler2D u_scene;
layout(set = 1, binding = 2) uniform sampler2D u_bloom;   // bloom mip 0, or a black texel without bloom
layout(set = 1, binding = 3) uniform sampler2D u_godRays; // half-res god rays, or a black texel without them

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

// Per-draw: modelRows[0] = (bloom intensity, god-ray intensity, 1: the target encodes sRGB itself, unused).
void main() {
    vec4 params = pd.modelRows[0];
    vec3 color = flFiniteColor(texture(u_scene, v_uv).rgb) * frame.post.x; // no NaN pixel from an Inf on RGBA16F
    color += texture(u_bloom, v_uv).rgb * params.x; // already exposed (bloom.frag's prefilter)
    color += texture(u_godRays, v_uv).rgb * params.y;
    color = flTonemap(color, frame.post.z);
    o_color = params.z > 0.5 ? vec4(color, 1.0) : flEncodeOutput(vec4(color, 1.0));
}
