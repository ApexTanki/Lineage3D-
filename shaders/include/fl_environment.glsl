#ifndef FL_ENVIRONMENT_GLSL
#define FL_ENVIRONMENT_GLSL

// Baked lighting helpers (M3 plan D5, D6, D8): HDR texture decoding, L1 SH irradiance and cubemap reflections with box
// projection. Needs fl_bindings.glsl.

// Lightmaps and environment cubes are BC6H (linear float) or, on devices without BPTC, RGBM with M = 8.
vec3 flDecodeHdr(vec4 texel) {
    return frame.outputParams.w > 0.5 ? texel.rgb * (texel.a * 8.0) : texel.rgb;
}

// L1 SH irradiance in evaluated form: E/π = w + dot(xyz, n) per channel; L1 rings for strong lights, so clamp.
vec3 flShIrradiance(vec4 sh[3], vec3 n) {
    return max(vec3(dot(sh[0].xyz, n) + sh[0].w, dot(sh[1].xyz, n) + sh[1].w, dot(sh[2].xyz, n) + sh[2].w), vec3(0.0));
}

// Reflection slot of a draw (PerDraw.lights.z / instance attribute 12, bits 12–15).
int flReflectionSlot(uvec4 packedLights) {
    return int((packedLights.z >> 12u) & 0xFu);
}

// The lookup direction for reflection vector r at `position`: box-projected for local probes (the ray from the surface
// is intersected with the probe's box, and the hit is looked up from the probe's position), r itself for the sky.
vec3 flReflectionDirection(int slot, vec3 position, vec3 r) {
    vec4 boxMin = frame.reflectionProbes[slot * 3];
    if (boxMin.w < 0.5) return r;
    vec3 boxMax = frame.reflectionProbes[slot * 3 + 1].xyz;
    vec3 probe = frame.reflectionProbes[slot * 3 + 2].xyz;
    vec3 safe = vec3(abs(r.x) > 1e-5 ? r.x : 1e-5, abs(r.y) > 1e-5 ? r.y : 1e-5, abs(r.z) > 1e-5 ? r.z : 1e-5);
    vec3 far = max((boxMax - position) / safe, (boxMin.xyz - position) / safe);
    float t = max(min(min(far.x, far.y), far.z), 0.0);
    return position + r * t - probe;
}

// Lagarde's gloss-aware Schlick Fresnel (ARCHITECTURE.md §4.4): F = s + (max(g, s) − s)·(1 − NdotV)⁵.
vec3 flFresnel(vec3 specularColor, float gloss, float nDotV) {
    float f = pow(1.0 - clamp(nDotV, 0.0, 1.0), 5.0);
    return specularColor + (max(vec3(gloss), specularColor) - specularColor) * f;
}

#endif
