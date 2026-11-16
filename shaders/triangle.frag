#version 450
#include "fl_common.glsl"
#include "fl_bindings.glsl"

layout(location = 0) in vec3 v_color;
layout(location = 0) out vec4 o_color;

void main() {
    o_color = flEncodeOutput(vec4(v_color, 1.0));
}
