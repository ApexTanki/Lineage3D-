#ifndef FL_SHADOWS_GLSL
#define FL_SHADOWS_GLSL

// Shadow receivers (ARCHITECTURE.md §4.14; M4 plan D5, D8). Fragment stage only; needs fl_bindings.glsl.
//   Sun: cascaded maps in one D16 array. Each pixel picks a cascade by view depth; in the last 10 % of a cascade, an
//        interleaved-gradient-noise dither moves pixels to the next one (the last cascade fades out instead), so
//        exactly one cascade is sampled per pixel. A receiver normal offset of about a texel removes acne.
//   Spots: up to four lights render into tiles of one D16 atlas; their light loop multiplies by flSpotShadow().
// Filters (frame.cascadeBands.w): 1 = one hardware comparison tap (2×2 bilinear PCF with linear depth compare),
// 2 = 16 Poisson taps rotated per pixel, 3 = four nearest taps blended by hand (devices without linear compare, P16).

layout(set = 0, binding = 2) uniform sampler2DArrayShadow u_cascades;
layout(set = 0, binding = 3) uniform sampler2DShadow u_spotAtlas;

const vec2 kPoisson16[16] = vec2[16](
    vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725), vec2(-0.09418410, -0.92938870), vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464), vec2(-0.38277543, 0.27676845), vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420), vec2(-0.26496911, -0.41893023), vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590), vec2(0.19984126, 0.78641367), vec2(0.14383161, -0.14100790));
// 1 / the longest offset above (1.2339): scaled by it, the kernel reaches exactly the radius in frame.cascadeFilter.
const float kPoisson16Scale = 0.81042;

// Interleaved gradient noise (Jimenez 2014): a per-pixel value in [0, 1) that is well spread in screen space.
float flInterleavedGradientNoise(vec2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

mat2 flRotation(float turns) {
    float a = turns * 6.2831853;
    float c = cos(a), s = sin(a);
    return mat2(c, s, -s, c);
}

// Implicit LOD on purpose. The cascades have one level, and their sampler one filter for both magnification and
// minification, without anisotropy, so the LOD can't change a tap, even where implicit derivatives are undefined
// (after the alpha test's discard, in per-pixel branches). sampler2DArrayShadow has no textureLod in GLSL 330, and
// explicit zero gradients (textureGrad) run on a slow path on NVIDIA: 16 Poisson taps cost 0.47 ms of a 0.78 ms scene
// pass at 1080p (M9 performance pass; ShadowMaps creates the maps with one level).
float flCascadeTap(vec2 uv, float layer, float depth) {
    return texture(u_cascades, vec4(uv, layer, depth));
}

float flFilterCascade(vec3 coord, int cascade, float noise) {
    float layer = float(cascade);
    int mode = int(frame.cascadeBands.w + 0.5);
    if (mode == 2) {
        mat2 r = flRotation(noise) * (frame.cascadeFilter[cascade] * kPoisson16Scale);
        float sum = 0.0;
        for (int i = 0; i < 16; ++i) sum += flCascadeTap(coord.xy + r * kPoisson16[i], layer, coord.z);
        return sum * (1.0 / 16.0);
    }
    if (mode == 3) {
        // 2×2 bilinear PCF by hand: four nearest comparisons at the texel centers around the point.
        float size = frame.cascadeTexel.w;
        vec2 st = coord.xy * size - 0.5;
        vec2 base = floor(st);
        vec2 f = st - base;
        vec2 uv = (base + 0.5) / size;
        float t = 1.0 / size;
        float s00 = flCascadeTap(uv, layer, coord.z), s10 = flCascadeTap(uv + vec2(t, 0.0), layer, coord.z);
        float s01 = flCascadeTap(uv + vec2(0.0, t), layer, coord.z), s11 = flCascadeTap(uv + vec2(t, t), layer, coord.z);
        return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
    }
    return flCascadeTap(coord.xy, layer, coord.z);
}

// The sun's visibility through the cascades at a surface point: x = visibility (1 = lit), y = coverage (1 inside the
// cascades, fading to 0 over the last band of the last one, 0 beyond or without cascades). `normal` is the unit
// geometric normal on the viewer's side; `cascade` returns the cascade sampled, or −1.
vec2 flCascadeShadow(vec3 position, vec3 normal, out int cascade) {
    cascade = -1;
    int count = int(frame.cascadeSplits.w + 0.5);
    if (count == 0) return vec2(1.0, 0.0);
    float depth = dot(position - frame.cameraPosition.xyz, frame.cameraForward.xyz);
    int c = 0;
    while (c < count && depth > frame.cascadeSplits[c]) ++c;
    if (c >= count) return vec2(1.0, 0.0);
    float noise = flInterleavedGradientNoise(gl_FragCoord.xy);
    float band = frame.cascadeBands[c];
    float t = clamp((depth - (frame.cascadeSplits[c] - band)) / max(band, 1e-4), 0.0, 1.0);
    float coverage = 1.0;
    if (c + 1 < count) {
        if (noise < t) ++c;
    } else {
        coverage = 1.0 - t;
    }
    cascade = c;
    // Normal offset: about a texel, more at grazing angles, where a texel spans more depth.
    float nl = clamp(dot(normal, frame.sunDirection.xyz), 0.0, 1.0);
    vec3 p = position + normal * (frame.cascadeTexel[c] * (0.6 + 1.6 * sqrt(1.0 - nl * nl)));
    vec3 coord = (frame.cascadeMatrix[c] * vec4(p, 1.0)).xyz;
    float rotation = flInterleavedGradientNoise(gl_FragCoord.xy + vec2(5.588, 5.588));
    return vec2(flFilterCascade(coord, c, rotation), coverage);
}

// A shadowed spot light's visibility (slot 0–3). Four rotated taps of the hardware 2×2 comparison, kept inside the
// light's tile. `distance` is the distance from the light (world texels grow with it).
float flSpotShadow(int slot, vec3 position, vec3 normal, float distance) {
    float texel = frame.spotTexel[slot] * distance;
    vec4 h = frame.spotMatrix[slot] * vec4(position + normal * (1.5 * texel), 1.0);
    if (h.w <= 1e-5) return 1.0;
    vec3 coord = h.xyz / h.w;
    vec4 tile = frame.spotTile[slot];
    float radius = 1.5 / float(textureSize(u_spotAtlas, 0).x);
    mat2 r = flRotation(flInterleavedGradientNoise(gl_FragCoord.xy + vec2(11.27, 3.93))) * radius;
    float sum = 0.0;
    for (int i = 0; i < 4; ++i) sum += textureLod(u_spotAtlas, vec3(clamp(coord.xy + r * kPoisson16[i], tile.xy, tile.zw), coord.z), 0.0);
    return sum * 0.25;
}

#endif
