#version 450
// Blob shadows (ARCHITECTURE.md §6 pass 5b, DEV-3; M4 plan D9): a horizontal quad under each flagged model, all of them
// in one instanced draw after the sky. Six vertices per instance from gl_VertexIndex, no vertex buffer.
#include "fl_common.glsl"
#include "fl_bindings.glsl"

layout(location = 8) in vec4 i_blob;   // xyz = center on the ground, w = radius along its X axis
layout(location = 9) in vec4 i_params; // x = darkening at the center (0–1), y = radius along its Z axis, zw = X axis (unit, xz)

layout(location = 0) out vec2 v_local; // −1..1 across the quad
layout(location = 1) out vec3 v_world;
layout(location = 2) flat out float v_strength;

void main() {
    // Two triangles: (−1,−1) (1,−1) (1,1) and (−1,−1) (1,1) (−1,1).
    int i = gl_VertexIndex;
    vec2 corner = vec2(i == 1 || i == 2 || i == 4 ? 1.0 : -1.0, i == 2 || i == 4 || i == 5 ? 1.0 : -1.0);
    v_local = corner;
    vec3 axisX = vec3(i_params.z, 0.0, i_params.w), axisZ = vec3(-i_params.w, 0.0, i_params.z); // turned with the model
    v_world = i_blob.xyz + axisX * (corner.x * i_blob.w) + axisZ * (corner.y * i_params.y);
    v_strength = i_params.x;
    gl_Position = frame.viewProj * vec4(v_world, 1.0);
}
