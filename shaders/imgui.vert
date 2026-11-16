#version 450
// Dear ImGui through the RHI. Locations follow ARCHITECTURE.md §4.7: 0 position, 3 uv0, 7 color.
#include "fl_common.glsl"

layout(location = 0) in vec2 a_position;
layout(location = 3) in vec2 a_uv;
layout(location = 7) in vec4 a_color;

layout(push_constant, std430) uniform PerDraw {
    mat4 projection; // canonical UI ortho fixed by renderer::ClipSpace
    vec4 params;     // x = 1 when the target encodes sRGB (ImGui colors are sRGB, so linearize first)
} pd;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;

void main() {
    v_uv = a_uv;
    v_color = a_color;
    if (pd.params.x > 0.5) v_color.rgb = flSrgbToLinear(v_color.rgb);
    gl_Position = pd.projection * vec4(a_position, 0.0, 1.0);
}
