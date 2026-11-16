#version 450
// Sprites (M6 plan D11): unlit atlas texel × color, with fog: blended sprites fade into the fog color, additive ones fade
// out. Premultiplied sprites premultiply here, so any straight-alpha atlas works with every blend mode.
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_fog.glsl"
#include "fl_post.glsl"

layout(set = 2, binding = 1) uniform sampler2D u_atlas; // binding 0 of set 2 is a uniform-buffer slot on GL

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_world;
layout(location = 2) in vec4 v_color;
layout(location = 3) flat in float v_blend;
layout(location = 0) out vec4 o_color;

void main() {
    vec4 c = texture(u_atlas, v_uv) * v_color;
    if (v_blend > 1.5) c.rgb *= c.a;
    o_color = vec4(flForwardOutputBlend(flApplyFog(c.rgb, c.a, v_world, v_blend), c.a, v_blend), c.a); // LDR path: the curve (M8 plan D8)
}
