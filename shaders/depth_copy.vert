#version 450
// Pass 5c (ARCHITECTURE.md §6, M7 plan D8): a full-screen triangle, emitted directly in clip space (rules C2, C8), so the
// pipeline uses CullMode::None. The fragment shader addresses texels by gl_FragCoord, so no UV is needed.
#include "fl_common.glsl"

void main() {
    vec2 ndc = vec2(gl_VertexIndex == 1 ? 3.0 : -1.0, gl_VertexIndex == 2 ? 3.0 : -1.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
