#version 450
// Skybox (ARCHITECTURE.md §6 pass 5, M3 plan D9): one full-screen triangle on the far plane. Emitted directly in clip
// space with z = w, which is depth 1 on both backends, so it fills only pixels no opaque geometry covered (LessOrEqual,
// no depth write). Clip-space geometry bypasses the projection, so the pipeline uses CullMode::None (rule C8).
#include "fl_common.glsl"
#include "fl_bindings.glsl"

layout(location = 0) out vec4 v_direction; // homogeneous: linear across the screen, divided per pixel

void main() {
    vec2 ndc = vec2(gl_VertexIndex == 1 ? 3.0 : -1.0, gl_VertexIndex == 2 ? 3.0 : -1.0);
    v_direction = frame.skyFromClip * vec4(ndc, 1.0, 1.0);
    gl_Position = vec4(ndc, 1.0, 1.0);
}
