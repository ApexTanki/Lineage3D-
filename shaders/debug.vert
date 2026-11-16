#version 450
// Debug geometry passes (M5 plan D10): every scene draw's triangles in one flat color, as lines over the lit frame
// (wireframe view) or added up per pixel (overdraw view). Positions only.
//   OBJECT_DYNAMIC    transform in the per-draw block (the object → world rows)
//   OBJECT_INSTANCED  static instance runs: transform from attributes 8–10 (the scene pass's instance data, 80 B each)
//   OBJECT_SKINNED    the transform blended from the bone palette (set 3; M6 plan D9)
#include "fl_common.glsl"
#ifdef OBJECT_SKINNED
#include "fl_skinning.glsl"
#endif

// Set 1, binding 0: the pass (a dynamic offset into the frame's uniform ring). Mirrored by renderer::DebugPassUniforms.
layout(std140, set = 1, binding = 0) uniform DebugPass {
    mat4 viewProj; // backend clip space: the scene camera's
    vec4 color;    // linear; overdraw adds it once per covering draw
} pass;

layout(push_constant, std430) uniform PerDraw {
    vec4 modelRows[3];
} pd;

layout(location = 0) in ivec4 a_position; // int16 grid coordinates; dequantization is folded into the rows
#ifdef OBJECT_INSTANCED
layout(location = 8) in vec4 i_row0;
layout(location = 9) in vec4 i_row1;
layout(location = 10) in vec4 i_row2;
#endif

void main() {
#if defined(OBJECT_SKINNED)
    vec4 rows[3];
    flSkinRows(rows);
#elif defined(OBJECT_INSTANCED)
    vec4 rows[3] = vec4[3](i_row0, i_row1, i_row2);
#else
    vec4 rows[3] = pd.modelRows;
#endif
    gl_Position = pass.viewProj * vec4(flTransformPoint(rows, vec3(a_position.xyz)), 1.0);
}
