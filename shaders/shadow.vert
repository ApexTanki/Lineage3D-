#version 450
// Shadow casters (ARCHITECTURE.md §4.14; M4 plan D7): depth only, into one cascade layer or one spot-atlas tile.
//   OBJECT_DYNAMIC    transform in the per-draw block
//   OBJECT_INSTANCED  static instance runs: transform from attributes 8–10 (the shadow view's rows-only instance data,
//                     48 B per instance; M5 plan D8)
//   OBJECT_SKINNED    the transform blended from the bone palette (set 3; M6 plan D9)
//   ALPHA_TEST        cut out by the albedo's alpha, with the material's UV animation (the fan's shadow spins)
// No set 0: a shadow pass never binds the frame set, which holds the very textures it renders into.
#include "fl_common.glsl"
#include "fl_material.glsl"
#ifdef OBJECT_SKINNED
#include "fl_skinning.glsl"
#endif

// Set 1, binding 0: one shadow pass (a dynamic offset into the frame's uniform ring). Mirrored by
// renderer::ShadowPassUniforms (engine/renderer/core/ShaderTypes.h).
layout(std140, set = 1, binding = 0) uniform ShadowPass {
    mat4 viewProj; // backend clip space, built by renderer::ClipSpace
    vec4 params;   // x = 1: pancake (flatten casters in front of the near plane onto it); y = depth offset added to
                   // NDC z (the per-cascade constant bias); z = time (s, UV animation); w = NDC z of the near plane
} pass;

// Per-draw constants of dynamic casters: the object → world rows only (48 B).
layout(push_constant, std430) uniform PerDraw {
    vec4 modelRows[3];
} pd;

layout(location = 0) in ivec4 a_position; // int16 grid coordinates; dequantization is folded into the rows
#ifdef ALPHA_TEST
layout(location = 3) in vec2 a_uv0;
layout(location = 0) out vec2 v_uv0;
#endif
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
    vec4 clip = pass.viewProj * vec4(flTransformPoint(rows, vec3(a_position.xyz)), 1.0);
    clip.z += pass.params.y * clip.w;
    // Pancaking (§4.14): casters between the sun and the cascade's depth range land on the near plane instead of being
    // clipped, so off-screen casters still cast. Cascades are orthographic (w = 1); spot passes switch it off.
    if (pass.params.x > 0.5) clip.z = max(clip.z, pass.params.w * clip.w);
    gl_Position = clip;
#ifdef ALPHA_TEST
    v_uv0 = flAnimateUv(a_uv0, pass.params.z);
#endif
}
