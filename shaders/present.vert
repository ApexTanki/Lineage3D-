#version 450
// Full-screen triangle (ARCHITECTURE.md §4.2 rules C2 and C8): emitted directly in clip space, so the pipeline uses
// CullMode::None, and the UV comes from each backend's own NDC, which samples the matching texel on both.
#include "fl_common.glsl"

layout(location = 0) out vec2 v_uv;

void main() {
    vec2 ndc = vec2(gl_VertexIndex == 1 ? 3.0 : -1.0, gl_VertexIndex == 2 ? 3.0 : -1.0);
    v_uv = ndc * 0.5 + 0.5;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
