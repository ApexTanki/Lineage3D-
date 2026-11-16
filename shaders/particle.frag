#version 450
// Particles (M7 plan D5–D9): the atlas cell (blended into the next one when the flipbook interpolates) × the vertex color,
// faded against the scene depth when SOFT, then fogged like sprites: blended particles fade into the fog color,
// additive ones fade out. Premultiplied particles premultiply here, so any straight-alpha atlas works with every mode.
//   SOFT     the soft-particle pipelines: the scene's linear view depth (pass 5c) is set 1, binding 1
//   DISTORT  heat haze (pass 8, M8 plan D11): the atlas is a normal map whose XY bends the half-res grab of the finished
//            scene (set 1, binding 2), masked by a radial falloff, the particle's opacity and the depth fade
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_fog.glsl"
#include "fl_post.glsl"

#define P_BLEND pd.modelRows[0].y    // 0 alpha, 1 additive, 2 premultiplied
#define P_INV_SOFT pd.modelRows[0].z // 1 / fade distance (m)
#define P_STRENGTH pd.lightmapST.y   // distort: m the picture moves at full opacity
#define FL_DEBUG_OVERDRAW 18         // renderer::DebugView::Overdraw
#define FL_OVERDRAW_STEP (1.0 / 32.0) // renderer::DebugDraw::kOverdrawStep

layout(set = 2, binding = 1) uniform sampler2D u_atlas; // binding 0 of set 2 is a uniform-buffer slot on GL
#if defined(SOFT) || defined(DISTORT)
layout(set = 1, binding = 1) uniform sampler2D u_sceneDepth; // binding 0 of set 1 is a uniform-buffer slot on GL
#endif
#ifdef DISTORT
layout(set = 1, binding = 2) uniform sampler2D u_grab;
layout(location = 5) in vec4 v_axes;
#endif

layout(location = 0) in vec4 v_uv;
layout(location = 1) in vec3 v_world;
layout(location = 2) in vec4 v_color;
layout(location = 3) flat in float v_frameBlend;
layout(location = 4) in float v_viewDepth;
layout(location = 0) out vec4 o_color;

void main() {
#ifdef DISTORT
    // Pass 8 draws into the resolved scene without a depth buffer, so the depth fade is also the occlusion test.
    vec4 t = texture(u_atlas, v_uv.xy);
    if (v_frameBlend > 0.0) t = mix(t, texture(u_atlas, v_uv.zw), v_frameBlend);
    vec2 local = v_uv.xy - floor(v_uv.xy); // the whole quad: distort emitters have no flipbook (the parser rejects one);
                                           // the radial mask keeps quad edges soft
    float mask = v_color.a * (1.0 - smoothstep(0.55, 1.0, length(local * 2.0 - 1.0)));
    float scene = texelFetch(u_sceneDepth, ivec2(gl_FragCoord.xy), 0).r;
    mask *= clamp((scene - v_viewDepth) * P_INV_SOFT, 0.0, 1.0) * (1.0 - flFogFactor(v_world)); // fogged haze bends less
    mask *= clamp((v_viewDepth - 0.3) / 0.7, 0.0, 1.0); // gone within 0.3 m of the camera: no screen-filling smear up close
    vec2 bend = (t.rg * 2.0 - 1.0) * (P_STRENGTH * mask);
    vec2 uv = gl_FragCoord.xy * frame.screen.zw + bend.x * v_axes.xy + bend.y * v_axes.zw;
    o_color = vec4(textureLod(u_grab, uv, 0.0).rgb, mask);
#else
    if (int(frame.outputParams.z + 0.5) == FL_DEBUG_OVERDRAW) { // every covered pixel of every quad adds one step
        o_color = vec4(FL_OVERDRAW_STEP, 0.0, 0.0, 0.0);
        return;
    }
    vec4 c = texture(u_atlas, v_uv.xy);
    if (v_frameBlend > 0.0) c = mix(c, texture(u_atlas, v_uv.zw), v_frameBlend);
    c *= v_color;
#ifdef SOFT
    // Both depths are meters along the view direction; the copy matches this pixel exactly (attachment texel rows).
    float scene = texelFetch(u_sceneDepth, ivec2(gl_FragCoord.xy), 0).r;
    c.a *= clamp((scene - v_viewDepth) * P_INV_SOFT, 0.0, 1.0);
#endif
    if (P_BLEND > 1.5) c.rgb *= c.a;
    o_color = vec4(flForwardOutputBlend(flApplyFog(c.rgb, c.a, v_world, P_BLEND), c.a, P_BLEND), c.a); // LDR path: the curve (M8 plan D8)
#endif
}
