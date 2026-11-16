#version 450
#include "fl_common.glsl"

// Set 2 (per material), first texture slot (ARCHITECTURE.md §4.6).
layout(set = 2, binding = 1) uniform sampler2D u_albedo;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 0) out vec4 o_color;

void main() {
    o_color = v_color * texture(u_albedo, v_uv);
}
