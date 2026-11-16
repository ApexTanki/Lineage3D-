#version 450
// Lit template (ARCHITECTURE.md §4.4, §4.7; M2 plan D1, M3 plan D7). The OBJECT axis picks where per-object data lives:
//   OBJECT_DYNAMIC             transform, light list, probe SH and sun visibility in the per-draw block
//   OBJECT_INSTANCED           static instance groups without lightmaps: transform + light list from attributes 8–10, 12
//   OBJECT_LIGHTMAP            a lightmapped static drawn on its own: per-draw block, UV1 → page via lightmapST
//   OBJECT_LIGHTMAP_INSTANCED  lightmapped static instance groups: attribute 11 = lightmapST
//   OBJECT_SKINNED             like OBJECT_DYNAMIC, with the transform blended from the bone palette (M6 plan D9)
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_material.glsl"
#ifdef OBJECT_SKINNED
#include "fl_skinning.glsl"
#endif

#if defined(OBJECT_INSTANCED) || defined(OBJECT_LIGHTMAP_INSTANCED)
#define FL_INSTANCED 1
#endif
#if defined(OBJECT_LIGHTMAP) || defined(OBJECT_LIGHTMAP_INSTANCED)
#define FL_LIGHTMAPPED 1
#endif

layout(location = 0) in ivec4 a_position; // int16 grid coordinates; dequantization is folded into the rows
layout(location = 1) in vec4 a_normal;    // RGBA8_SNORM
#ifdef NORMAL_MAP
layout(location = 2) in vec4 a_tangent;   // RGBA8_SNORM, w = bitangent sign (MikkTSpace)
#endif
layout(location = 3) in vec2 a_uv0;
#ifdef FL_LIGHTMAPPED
layout(location = 4) in vec2 a_uv1;       // RG16_UNORM, the mesh's lightmap layout
#endif
#ifdef FL_INSTANCED
layout(location = 8) in vec4 i_row0;
layout(location = 9) in vec4 i_row1;
layout(location = 10) in vec4 i_row2;
#ifdef OBJECT_LIGHTMAP_INSTANCED
layout(location = 11) in vec4 i_lightmapST;
#endif
layout(location = 12) in uvec4 i_lights;  // light list | lightmap page | reflection slot
#endif

layout(location = 0) out vec3 v_worldPosition;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv0;
layout(location = 3) flat out uvec4 v_lights;
#ifdef NORMAL_MAP
layout(location = 4) out vec4 v_tangent; // xyz world tangent, w = bitangent sign including the transform's mirroring
#endif
#ifdef FL_LIGHTMAPPED
layout(location = 5) out vec3 v_lightmapUv; // xy = lightmap page UV, z = page
#endif
#ifdef DETAIL_LAYER
layout(location = 6) out vec2 v_detailUv;
#endif
#ifdef OBJECT_SKINNED
layout(location = 7) out vec3 v_boneColor;
#endif

void main() {
#if defined(OBJECT_SKINNED)
    vec4 rows[3];
    flSkinRows(rows);
    uvec4 packedLights = pd.lights;
    v_boneColor = flBoneWeightColor();
#elif defined(FL_INSTANCED)
    vec4 rows[3] = vec4[3](i_row0, i_row1, i_row2);
    uvec4 packedLights = i_lights;
#else
    vec4 rows[3] = pd.modelRows;
    uvec4 packedLights = pd.lights;
#endif
    v_lights = packedLights;
    vec3 world = flTransformPoint(rows, vec3(a_position.xyz));
    v_worldPosition = world;
    // Unit length at the vertices; the fragment shader uses the interpolated, unnormalized frame (MikkTSpace).
    v_normal = normalize(flTransformNormal(rows, a_normal.xyz));
#ifdef NORMAL_MAP
    v_tangent = vec4(normalize(flTransformDirection(rows, a_tangent.xyz)), a_tangent.w * flMirrorSign(rows));
#endif
    float time = frame.time.x;
    v_uv0 = flAnimateUv(a_uv0, time);
#ifdef FL_LIGHTMAPPED
#ifdef OBJECT_LIGHTMAP_INSTANCED
    vec4 st = i_lightmapST;
#else
    vec4 st = pd.lightmapST;
#endif
    v_lightmapUv = vec3(a_uv1 * st.xy + st.zw, float((packedLights.z >> 4u) & 0xFFu));
#endif
#ifdef DETAIL_LAYER
    v_detailUv = a_uv0 * material.detail.z + fract(material.detail.xy * time);
#endif
    gl_Position = frame.viewProj * vec4(world, 1.0);
}
