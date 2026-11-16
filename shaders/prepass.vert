#version 450
// Pass 2, the depth prepass (High; ARCHITECTURE.md §6, M8 plan D9): lit.vert's position and nothing else. The position
// code below is lit.vert's, expression for expression, and gl_Position is invariant in both (fl_common.glsl), so the
// opaque pass that follows tests LessOrEqual against exactly the depths written here.
//   FADE  a draw in a LOD cross-fade (M5 plan D7): its LOD word goes to the fragment stage, which keeps the pixels the
//         scene pass's fade twin keeps (fl_lodfade.glsl)
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#ifdef OBJECT_SKINNED
#include "fl_skinning.glsl"
#endif

#if defined(OBJECT_INSTANCED) || defined(OBJECT_LIGHTMAP_INSTANCED)
#define FL_INSTANCED 1
#endif

layout(location = 0) in ivec4 a_position; // int16 grid coordinates; dequantization is folded into the rows
#ifdef FL_INSTANCED
layout(location = 8) in vec4 i_row0;
layout(location = 9) in vec4 i_row1;
layout(location = 10) in vec4 i_row2;
#endif
#ifdef FADE
#ifdef FL_INSTANCED
layout(location = 12) in uvec4 i_lights; // w: the instance's LOD word
#endif
layout(location = 0) flat out uint v_lodWord;
#endif

void main() {
#if defined(OBJECT_SKINNED)
    vec4 rows[3];
    flSkinRows(rows);
#elif defined(FL_INSTANCED)
    vec4 rows[3] = vec4[3](i_row0, i_row1, i_row2);
#else
    vec4 rows[3] = pd.modelRows;
#endif
#ifdef FADE
#ifdef FL_INSTANCED
    v_lodWord = i_lights.w;
#else
    v_lodWord = pd.lights.w;
#endif
#endif
    vec3 world = flTransformPoint(rows, vec3(a_position.xyz));
    gl_Position = frame.viewProj * vec4(world, 1.0);
}
