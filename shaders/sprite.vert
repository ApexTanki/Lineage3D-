#version 450
// Sprites (M6 plan D11): flipbook billboards, one instanced draw per run of sprites that share an atlas and a blend mode,
// sorted back to front. Six vertices per instance from gl_VertexIndex; no vertex buffer.
#include "fl_common.glsl"
#include "fl_bindings.glsl"

layout(location = 8) in vec4 i_centerRotation; // xyz = center (world), w = rotation (radians, camera-facing sprites)
layout(location = 9) in vec4 i_sizeFacing;     // xy = size (m), z = 1: upright about Y, w = blend for fog
layout(location = 10) in vec4 i_uvRect;        // the flipbook cell: UV offset xy, scale zw
layout(location = 11) in vec4 i_color;         // linear RGB × opacity

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec3 v_world;
layout(location = 2) out vec4 v_color;
layout(location = 3) flat out float v_blend;

void main() {
    // Two triangles: (−1,−1) (1,−1) (1,1) and (−1,−1) (1,1) (−1,1).
    int i = gl_VertexIndex;
    vec2 corner = vec2(i == 1 || i == 2 || i == 4 ? 1.0 : -1.0, i == 2 || i == 4 || i == 5 ? 1.0 : -1.0);
    vec3 center = i_centerRotation.xyz;
    vec3 toCamera = frame.cameraPosition.xyz - center;
    vec3 right, up;
    if (i_sizeFacing.z > 0.5) {
        // Upright: turned about Y toward the camera (flames stay vertical as the camera looks down on them).
        vec3 level = vec3(toCamera.x, 0.0, toCamera.z);
        vec3 f = dot(level, level) > 1e-8 ? normalize(level) : vec3(0.0, 0.0, 1.0);
        right = vec3(f.z, 0.0, -f.x);
        up = vec3(0.0, 1.0, 0.0);
    } else {
        vec3 f = dot(toCamera, toCamera) > 1e-8 ? normalize(toCamera) : vec3(0.0, 0.0, 1.0);
        // The view's up axis is never parallel to the direction of anything on screen (a world-up helper flipped the
        // sprite as that direction crossed the vertical; M7 review).
        vec3 r = normalize(cross(frame.cameraUp.xyz, f));
        vec3 u = cross(f, r);
        float c = cos(i_centerRotation.w), s = sin(i_centerRotation.w);
        right = r * c + u * s;
        up = u * c - r * s;
    }
    v_world = center + right * (corner.x * 0.5 * i_sizeFacing.x) + up * (corner.y * 0.5 * i_sizeFacing.y);
    // Corner (−1, −1) is the cell's bottom left; v = 0 is the top image row (the flipbook convention, Flipbook.h).
    v_uv = i_uvRect.xy + vec2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5) * i_uvRect.zw;
    v_color = i_color;
    v_blend = i_sizeFacing.w;
    gl_Position = frame.viewProj * vec4(v_world, 1.0);
}
