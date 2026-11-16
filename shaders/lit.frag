#version 450
// Lit template, fragment stage (ARCHITECTURE.md §4.4, §4.5; M2 plan D1, D5–D7, D12; M3 plan D4–D11; M4 plan D5–D8).
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_material.glsl"
#include "fl_shadows.glsl" // before fl_lighting.glsl: the light loop samples spot shadows
#include "fl_lighting.glsl"
#include "fl_environment.glsl"
#include "fl_fog.glsl"
#include "fl_post.glsl"
#ifdef ALPHA_TEST
#include "fl_lodfade.glsl"
#endif

#if defined(OBJECT_INSTANCED) || defined(OBJECT_LIGHTMAP_INSTANCED)
#define FL_INSTANCED 1
#endif
#if defined(OBJECT_LIGHTMAP) || defined(OBJECT_LIGHTMAP_INSTANCED)
#define FL_LIGHTMAPPED 1
#endif
#if defined(OBJECT_DYNAMIC) || defined(OBJECT_SKINNED)
#define FL_PROBE_LIT 1 // indirect light and sun visibility from the per-draw probe sample
#endif

layout(location = 0) in vec3 v_worldPosition;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv0;
layout(location = 3) flat in uvec4 v_lights;
#ifdef NORMAL_MAP
layout(location = 4) in vec4 v_tangent;
#endif
#ifdef FL_LIGHTMAPPED
layout(location = 5) in vec3 v_lightmapUv;
#endif
#ifdef DETAIL_LAYER
layout(location = 6) in vec2 v_detailUv;
#endif
#ifdef OBJECT_SKINNED
layout(location = 7) in vec3 v_boneColor; // the bone-weights view (M6 plan D12)
#endif
layout(location = 0) out vec4 o_color;

#ifdef FL_LIGHTMAPPED
layout(set = 0, binding = 5) uniform sampler2DArray u_lightmaps;  // E/π without the sun's direct term (BC6H or RGBM)
layout(set = 0, binding = 6) uniform sampler2DArray u_shadowmask; // sun visibility (BC4)
#endif
layout(set = 3, binding = 1) uniform samplerCube u_reflection;     // the sky or the object's local reflection probe
layout(set = 1, binding = 2) uniform sampler2D u_ssao;             // half-res AO of the depth prepass (M8 plan D9)
layout(set = 1, binding = 1) uniform sampler2D u_sceneDepth;       // linear view depth: of the prepass in pass A, of pass A in pass B
layout(set = 1, binding = 3) uniform sampler2D u_grab;             // pass B: half-res resolved scene of pass A

// Material debug views (FrameUniforms.outputParams.z; renderer::DebugView).
const int kViewAlbedo = 1;
const int kViewLighting = 2;
const int kViewNormal = 3;
const int kViewSpecular = 4;
const int kViewGloss = 5;
const int kViewEmission = 6;
const int kViewLightCount = 7;
const int kViewLightmap = 8;
const int kViewProbes = 9;
const int kViewSunVisibility = 10;
const int kViewReflection = 11;
const int kViewCascades = 12;
const int kViewShadow = 13;
const int kViewUv = 14;
const int kViewTangent = 15;
const int kViewLod = 16;
const int kViewBones = 19; // 17 (wireframe) and 18 (overdraw) are separate passes
const int kViewSsao = 20;

// Where a world point lands in the scene targets (both backends: the UV that samples the texel it was rendered to) and
// its depth along the view direction (the linear depth target's unit).
vec2 flScreenUv(vec3 world) {
    vec4 clip = frame.viewProj * vec4(world, 1.0);
    return clip.xy / clip.w * 0.5 + 0.5;
}
float flViewDepth(vec3 world) {
    return dot(world - frame.cameraPosition.xyz, frame.cameraForward.xyz);
}

// SSR (P2, M8 plan D12): march the reflected ray in world space with growing steps; the first step that passes behind
// the opaque scene's depth (within a thickness) is refined by bisection. Returns the hit's UV in the grab, or a negative
// UV when the ray leaves the screen or finds nothing.
vec2 flTraceReflection(vec3 origin, vec3 dir) {
    vec3 p = origin;
    float stepLength = 0.2;
    for (int i = 0; i < 24; ++i) {
        vec3 q = p + dir * stepLength;
        vec2 uv = flScreenUv(q);
        float rayDepth = flViewDepth(q);
        if (rayDepth <= 0.0 || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return vec2(-1.0);
        float behind = rayDepth - textureLod(u_sceneDepth, uv, 0.0).r;
        if (behind > 0.0) {
            if (behind > max(0.3, stepLength * 2.0)) return vec2(-1.0); // passed behind an object without touching it
            vec3 a = p, b = q;
            for (int k = 0; k < 4; ++k) {
                vec3 m = (a + b) * 0.5;
                if (flViewDepth(m) > textureLod(u_sceneDepth, flScreenUv(m), 0.0).r) b = m;
                else a = m;
            }
            return flScreenUv(b);
        }
        p = q;
        stepLength *= 1.3;
    }
    return vec2(-1.0);
}

void main() {
    // Every texture is read before the alpha test can discard, so no implicit derivative follows a discard.
    vec4 albedo = texture(u_albedo, v_uv0) * material.baseColor;
#ifdef DETAIL_LAYER
    vec4 detail = texture(u_detail, v_detailUv);
    albedo.rgb = mix(albedo.rgb, detail.rgb, detail.a * material.detail.w);
#endif
#ifdef NORMAL_MAP
    vec2 normalXY = (texture(u_normal, v_uv0).rg * 2.0 - 1.0) * material.misc.z;
#endif
    vec3 specularColor = material.specularGloss.rgb;
    float gloss = material.specularGloss.a;
#ifdef SPEC_GLOSS_MAP
    vec4 specGloss = texture(u_specGloss, v_uv0);
    specularColor *= specGloss.rgb;
    gloss *= specGloss.a;
#endif
    vec3 emission = material.emission.rgb * material.emission.a;
#ifdef EMISSION_MAP
    emission *= texture(u_emission, v_uv0).rgb;
#endif
#ifdef FL_LIGHTMAPPED
    vec3 indirect = flDecodeHdr(texture(u_lightmaps, v_lightmapUv));
    float sunVisibility = texture(u_shadowmask, v_lightmapUv).r;
#endif

    float alpha = albedo.a;
#ifdef ALPHA_TEST
    float cutoff = material.uvAnim.w;
    if (cutoff < 0.0) {
        alpha = 1.0; // the cross-fade twin of an opaque material (M5 plan D7): no alpha test
    } else if (frame.outputParams.y > 0.5) {
        // Alpha to coverage (MSAA): sharpen alpha around the cutoff so the coverage ramp is about one pixel wide.
        alpha = clamp((alpha - cutoff) / max(fwidth(alpha), 1e-4) + 0.5, 0.0, 1.0);
    } else if (alpha < cutoff) {
        discard;
    }
    // LOD cross-fade (M5 plan D7; fl_lodfade.glsl): every pixel shows exactly one of the two levels.
    if (flLodFadeDiscards(v_lights.w, gl_FragCoord.xy)) discard;
#endif

#ifdef NORMAL_MAP
    // MikkTSpace: the interpolated, unnormalized frame; bitangent = sign × cross(N, T).
    vec3 bitangent = v_tangent.w * cross(v_normal, v_tangent.xyz);
    float normalZ = sqrt(clamp(1.0 - dot(normalXY, normalXY), 0.0, 1.0));
    vec3 n = normalize(normalXY.x * v_tangent.xyz + normalXY.y * bitangent + normalZ * v_normal);
#else
    vec3 n = normalize(v_normal);
#endif
    float facing = gl_FrontFacing ? 1.0 : -1.0;
    n *= facing; // back faces of double-sided materials are lit on their own side
    vec3 geometricNormal = normalize(v_normal) * facing;

    // Indirect light (§4.5): the lightmap on lightmapped statics, the probe SH on dynamic objects, the hemispherical
    // ambient on static instance groups without a lightmap. The sun's direct term is always live, masked by the
    // shadowmask or the probes' sun visibility.
#if defined(FL_PROBE_LIT)
    vec3 indirect = flShIrradiance(pd.sh, n);
    float sunVisibility = pd.lightmapST.x;
#elif !defined(FL_LIGHTMAPPED)
    vec3 indirect = mix(frame.ambientGround.rgb, frame.ambientSky.rgb, n.y * 0.5 + 0.5);
    float sunVisibility = 1.0;
#endif

    // Dynamic shadows (M4 plan D6, D-SHADOWMASK): the sun's direct term is never baked, so min(baked, cascades) can't
    // darken twice. When static geometry casts into the cascades (High), dynamic objects take the cascades alone inside
    // their range, instead of the probes' coarse sun visibility, and blend back to the probes where the cascades end.
    int cascade;
    vec2 csm = flCascadeShadow(v_worldPosition, geometricNormal, cascade);
#if defined(FL_PROBE_LIT)
    float sunShadow =frame.cascadeFilter.w > 0.5 ? mix(sunVisibility, csm.x, csm.y) : min(sunVisibility, mix(1.0, csm.x, csm.y));
#else
    float sunShadow = min(sunVisibility, mix(1.0, csm.x, csm.y));
#endif

    FlSurface s;
    s.position = v_worldPosition;
    s.normal = n;
    s.geometricNormal = geometricNormal;
    s.view = normalize(frame.cameraPosition.xyz - v_worldPosition);
    s.exponent = flSpecularExponent(gloss);
    vec3 directLight = vec3(0.0);
    vec3 specularLight = vec3(0.0);
    flAccumulate(s, frame.sunDirection.xyz, frame.sunColor.rgb * sunShadow, directLight, specularLight);
    flLocalLights(s, v_lights, directLight, specularLight);
    // SSAO (High; DEV-1): it darkens the indirect light and the environment reflection only, never direct light. The
    // half-res result is upsampled bilinearly (gl_FragCoord × 1/size matches where it was rendered on both backends).
    // It comes from the prepass, so only a fragment at the prepass depth takes it: one in front of it (alpha-tested,
    // blended, refractive) isn't in that depth and would otherwise take the AO of whatever lies behind it. The tolerance
    // covers the depth buffer's quantization (about 1 cm at 200 m with D24) and MSAA edges (the copy holds sample 0).
    float ao = 1.0;
    if (frame.post.w > 0.5) {
        float prepassDepth = texelFetch(u_sceneDepth, ivec2(gl_FragCoord.xy), 0).r;
        if (flViewDepth(v_worldPosition) > prepassDepth - max(0.05, 0.01 * prepassDepth))
            ao = texture(u_ssao, gl_FragCoord.xy * frame.screen.zw).r;
    }
    vec3 diffuse = albedo.rgb * (indirect * ao + directLight);
    vec3 specular = specularColor * specularLight;

    // Environment reflection (§4.4): the cube's mip matches the gloss; the sky cube knows no occluders, so it is scaled
    // by how much of the sky's light reaches the surface (its indirect light against the sky's own).
    int slot = flReflectionSlot(v_lights);
    vec3 r = reflect(-s.view, n);
    float lod = (1.0 - gloss) * frame.reflectionProbes[slot * 3 + 1].w;
    vec3 environment = flDecodeHdr(textureLod(u_reflection, flReflectionDirection(slot, v_worldPosition, r), lod));
    float occlusion = slot == 0 ? clamp(dot(indirect, vec3(0.2126, 0.7152, 0.0722)) / max(frame.ambientSky.w, 1e-4), 0.0, 1.0) : 1.0;
    vec3 reflection = environment * flFresnel(specularColor, gloss, dot(n, s.view)) * (material.misc.x * occlusion * ao);

    int view = int(frame.outputParams.z + 0.5);
    if (view != 0) {
        // Data views go through flSrgbToLinear so the screen shows the raw value after the output encoding.
        vec3 c = vec3(1.0, 0.0, 1.0);
        if (view == kViewAlbedo) c = albedo.rgb;
        else if (view == kViewLighting) c = indirect * ao + directLight + specular + reflection;
        else if (view == kViewNormal) c = flSrgbToLinear(n * 0.5 + 0.5);
        else if (view == kViewSpecular) c = specularColor;
        else if (view == kViewGloss) c = flSrgbToLinear(vec3(gloss));
        else if (view == kViewEmission) c = emission;
        else if (view == kViewLightCount) {
            float k = float(min(v_lights.z & 0xFu, 8u)) / 8.0;
            c = flSrgbToLinear(vec3(k, 4.0 * k * (1.0 - k), 1.0 - k));
        }
#ifdef FL_LIGHTMAPPED
        else if (view == kViewLightmap) c = indirect;
#else
        else if (view == kViewLightmap) c = vec3(0.0);
#endif
#ifdef FL_PROBE_LIT
        else if (view == kViewProbes) c = indirect;
#else
        else if (view == kViewProbes) c = vec3(0.0);
#endif
        else if (view == kViewSunVisibility) c = flSrgbToLinear(vec3(sunVisibility));
        else if (view == kViewReflection) c = reflection;
        else if (view == kViewCascades) {
            const vec3 kTint[3] = vec3[3](vec3(1.0, 0.35, 0.35), vec3(0.35, 1.0, 0.35), vec3(0.4, 0.55, 1.0));
            vec3 lit = albedo.rgb * (indirect + directLight) + specular;
            c = cascade >= 0 ? lit * kTint[cascade] : lit;
        }
        else if (view == kViewShadow) c = flSrgbToLinear(vec3(sunShadow));
        else if (view == kViewUv) c = flSrgbToLinear(vec3(fract(v_uv0), 0.0));
#ifdef NORMAL_MAP
        else if (view == kViewTangent) c = flSrgbToLinear(normalize(v_tangent.xyz) * 0.5 + 0.5);
#else
        else if (view == kViewTangent) c = flSrgbToLinear(vec3(0.5)); // no tangent stream without a normal map
#endif
        else if (view == kViewLod) {
            // Level 0 green, 1 yellow, 2 orange, 3 red over the lit frame; fading draws show both levels dithered.
            const vec3 kTint[4] = vec3[4](vec3(0.35, 1.0, 0.35), vec3(1.0, 1.0, 0.3), vec3(1.0, 0.6, 0.25), vec3(1.0, 0.3, 0.3));
            vec3 lit = albedo.rgb * (indirect + directLight) + specular;
            c = lit * kTint[(v_lights.w >> 12u) & 3u];
        }
        else if (view == kViewBones) {
            // Each joint a hashed color, blended by the vertex weights: rigid parts flat, smooth skin as gradients.
            float shade = 0.4 + 0.6 * max(dot(n, frame.sunDirection.xyz), 0.0);
#ifdef OBJECT_SKINNED
            c = flSrgbToLinear(v_boneColor) * shade;
#else
            c = flSrgbToLinear(vec3(0.3)) * shade;
#endif
        }
        else if (view == kViewSsao) c = flSrgbToLinear(vec3(ao));
        o_color = vec4(c, alpha);
        return;
    }
    // Refraction (water, glass; M8 plan D10), in pass B after the grab: the scene behind, sampled where the normal map's
    // bend moves it (a sample in front of the surface would leak a foreground object, so it falls back to straight
    // through), tinted, then reflection by Fresnel. SSR (P2, D12) replaces the cube where its march hits the scene.
    if (material.refraction.a > 0.0 && frame.postParams.x > 0.5) {
        float surfaceDepth = flViewDepth(v_worldPosition);
        vec2 straight = gl_FragCoord.xy * frame.screen.zw;
        vec2 bent = flScreenUv(v_worldPosition + (n - geometricNormal) * material.refraction.a);
        if (textureLod(u_sceneDepth, bent, 0.0).r < surfaceDepth) bent = straight;
        vec3 fresnel = flFresnel(specularColor, gloss, dot(n, s.view));
        // The grab already holds the fog between the camera and what lies behind: the part in front of the surface comes
        // out before the tint (the glass doesn't filter fog that's in front of it) and goes back once, with the surface's
        // own fog below, so water and glass aren't foggier than their surroundings.
        float fogInFront = flFogFactor(v_worldPosition);
        vec3 behind = max(textureLod(u_grab, bent, 0.0).rgb - frame.fogColor.rgb * fogInFront, vec3(0.0));
        vec3 through = behind * material.refraction.rgb * (1.0 - fresnel);
        vec3 mirror = reflection;
        if (frame.postParams.y > 0.5) {
            vec2 hit = flTraceReflection(v_worldPosition + geometricNormal * 0.02, r);
            if (hit.x >= 0.0) {
                vec2 edge = min(hit, 1.0 - hit);
                float fade = smoothstep(0.0, 0.08, min(edge.x, edge.y)); // no hard cut where the screen ends
                mirror = mix(mirror, textureLod(u_grab, hit, 0.0).rgb * fresnel * material.misc.x, fade);
            }
        }
        o_color = vec4(flForwardOutput(through + flApplyFog(mirror + specular + emission, 1.0, v_worldPosition, 0.0)), 1.0);
        return;
    }
    bool premultiplied = material.misc.w > 0.5;
    vec3 color = (premultiplied ? diffuse * alpha : diffuse) + specular + reflection + emission;
    // LDR path (M8 plan D8): exposure and the curve here; on the HDR path the tonemap pass applies them once. A lit
    // premultiplied color adds specular that its coverage doesn't scale, so it takes the curve as it is.
    o_color = vec4(flForwardOutput(flApplyFog(color, alpha, v_worldPosition, premultiplied ? 2.0 : material.misc.y)), alpha);
}
