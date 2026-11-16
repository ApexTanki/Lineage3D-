#ifndef FL_LIGHTING_GLSL
#define FL_LIGHTING_GLSL

// Lighting model (ARCHITECTURE.md §4.4): energy-normalized Blinn-Phong in linear space. Light intensities fold in π,
// so a white Lambertian surface facing a light reflects exactly color × intensity.

#define FL_MAX_LIGHTS 256
#define FL_MAX_LIGHTS_PER_OBJECT 8

// One local light, 64 bytes. Mirrored by renderer::GpuLight (engine/renderer/core/ShaderTypes.h).
struct FlLight {
    vec4 positionInvRange; // xyz world position, w = 1 / range
    vec4 color;            // rgb = color × intensity (linear)
    vec4 spotDirection;    // xyz = unit direction the spot points along, w = cos(outer angle); points: w = -2
    vec4 spotParams;       // x = cos(inner angle), always > cos(outer angle); points: -1. y = spot shadow slot, -1 = none
};

// Set 0, binding 1: this frame's local lights, at a dynamic offset into the uniform ring (16 KB, the GL 3.3 minimum
// block size). Objects index it through their packed per-object list (PerDraw.lights / instance attribute 12).
layout(std140, set = 0, binding = 1) uniform LightArray {
    FlLight items[FL_MAX_LIGHTS];
} lights;

// Gloss 0–1 → Blinn-Phong exponent 2–2048.
float flSpecularExponent(float gloss) {
    return exp2(1.0 + 10.0 * gloss);
}

// Range window: 1 at the light, 0 at the range, with zero slope at both ends. Same formula as renderer::lightAttenuation.
float flAttenuation(float distanceSquared, float invRange) {
    float x = clamp(1.0 - distanceSquared * invRange * invRange, 0.0, 1.0);
    return x * x;
}

struct FlSurface {
    vec3 position;        // world
    vec3 normal;          // unit, world (normal-mapped)
    vec3 geometricNormal; // unit, world, on the viewer's side: shadow lookups offset along it
    vec3 view;            // unit, toward the eye
    float exponent;
};

// Adds one light: `diffuse` collects irradiance (multiplied by albedo later), `specular` the normalized Blinn-Phong
// lobe (multiplied by the specular color later).
void flAccumulate(FlSurface s, vec3 l, vec3 radiance, inout vec3 diffuse, inout vec3 specular) {
    float nl = max(dot(s.normal, l), 0.0);
    vec3 h = l + s.view; // zero when the light sits exactly opposite the eye: guarded, normalize() would give NaN
    h *= inversesqrt(max(dot(h, h), 1e-12));
    float nh = max(dot(s.normal, h), 0.0);
    diffuse += radiance * nl;
    specular += radiance * (nl * (s.exponent + 8.0) * 0.125 * pow(nh, s.exponent));
}

// The object's local lights: x, y = 8 × u8 indices into LightArray, z bits 0–3 = count (ARCHITECTURE.md §4.7).
void flLocalLights(FlSurface s, uvec4 packedLights, inout vec3 diffuse, inout vec3 specular) {
    uint count = min(packedLights.z & 0xFu, uint(FL_MAX_LIGHTS_PER_OBJECT));
    for (uint i = 0u; i < count; ++i) {
        uint word = i < 4u ? packedLights.x : packedLights.y;
        FlLight light = lights.items[(word >> ((i & 3u) * 8u)) & 0xFFu];
        vec3 toLight = light.positionInvRange.xyz - s.position;
        float d2 = dot(toLight, toLight);
        vec3 l = toLight * inversesqrt(max(d2, 1e-8));
        float spot = smoothstep(light.spotDirection.w, light.spotParams.x, dot(-l, light.spotDirection.xyz));
#ifdef FL_SHADOWS_GLSL
        // A spot with a shadow slot (M4 plan D8); skipped where the cone already ends the light.
        if (light.spotParams.y >= 0.0 && spot > 0.0) spot *= flSpotShadow(int(light.spotParams.y + 0.5), s.position, s.geometricNormal, sqrt(d2));
#endif
        flAccumulate(s, l, light.color.rgb * (flAttenuation(d2, light.positionInvRange.w) * spot), diffuse, specular);
    }
}

#endif
