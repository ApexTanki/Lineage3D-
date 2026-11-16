#version 450
// The scene grab (passes 5d and 8, M8 plan D10, D11): the resolved scene downsampled to half resolution. One bilinear tap
// at the center of each 2×2 block averages it.
#include "fl_common.glsl"

layout(set = 1, binding = 1) uniform sampler2D u_input;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main() {
    o_color = vec4(textureLod(u_input, v_uv, 0.0).rgb, 1.0);
}
