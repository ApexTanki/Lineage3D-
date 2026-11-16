#version 450
// M0 parity triangle. It goes through ClipSpace with back-face culling on, so a winding or Y-flip mistake on
// either backend makes it vanish or turn upside down. The red vertex must be at the top.
#include "fl_common.glsl"
#include "fl_bindings.glsl"

layout(location = 0) out vec3 v_color;

void main() {
    // Counter-clockwise as seen from +Z (the camera side).
    const vec3 kPositions[3] = vec3[3](vec3(0.0, 0.6, 0.0), vec3(-0.6, -0.45, 0.0), vec3(0.6, -0.45, 0.0));
    const vec3 kColors[3] = vec3[3](vec3(1.0, 0.15, 0.1), vec3(0.1, 0.9, 0.2), vec3(0.15, 0.3, 1.0));
    vec3 world = flTransformPoint(pd.modelRows, kPositions[gl_VertexIndex]);
    v_color = kColors[gl_VertexIndex];
    gl_Position = frame.viewProj * vec4(world, 1.0);
}
