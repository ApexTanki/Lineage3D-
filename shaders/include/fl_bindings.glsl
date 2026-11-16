#ifndef FL_BINDINGS_GLSL
#define FL_BINDINGS_GLSL

// Binding convention (ARCHITECTURE.md §4.6): set 0 per frame, 1 per pass, 2 per material, 3 per-draw
// resources. Bindings exist only in the SPIR-V. SPIRV-Cross drops them for GLSL 330, and the
// GL backend re-binds blocks and samplers by name, so block and sampler names here are part of the ABI.

#define FL_REFLECTION_SLOTS 8 // slot 0 = the sky, 1–7 = a map's local reflection probes (M3 plan D8)

// Set 0, binding 0: per-frame constants (dynamic offset into the frame's uniform ring).
// Mirrored by renderer::FrameUniforms (engine/renderer/core/ShaderTypes.h).
layout(std140, set = 0, binding = 0) uniform FrameUniforms {
    mat4 viewProj;       // backend-ready: built by renderer::ClipSpace
    mat4 skyFromClip;    // sky pass: backend clip position on the far plane → world view direction (rotation only)
    vec4 time;           // x = seconds since start, y = frame delta
    vec4 outputParams;   // x = 1: encode sRGB in the shader (backbuffer not sRGB); y = 1: alpha to coverage is on;
                         // z = material debug view (0 = lit); w = 1: HDR textures (lightmaps, cubes) are RGBM
    vec4 cameraPosition; // xyz, world space
    vec4 sunDirection;   // xyz = unit vector toward the sun; w = cos(sun disc radius)
    vec4 sunColor;       // rgb = color × intensity; w = sun disc radiance multiplier (sky pass)
    vec4 ambientSky;     // hemispherical ambient: rgb seen by normals pointing up; w = luminance of the sky's E/π
    vec4 ambientGround;  // ... and down; w = sky cube intensity
    vec4 fogColor;       // rgb; w = mode: 0 off, 1 linear, 2 exp, 3 exp2
    vec4 fogParams;      // x = start, y = end (linear), z = density (exp/exp2), w = max opacity
    vec4 fogHeight;      // x = base height, y = density there, z = falloff (1/m), w = 1: height fog on
    vec4 fogSky;         // x = horizon band of the sky (view direction y) that takes distance fog
    vec4 environment;    // x = max mip of the sky cube, y = 1: a sky is loaded (sky pass + horizon fog)
    vec4 reflectionProbes[3 * FL_REFLECTION_SLOTS]; // per slot: boxMin (w = 1: box projection), boxMax (w = max mip),
                                                    // probe position
    // Sun shadow cascades (M4 plan D4, D5) and spot shadows (D8); see fl_shadows.glsl.
    mat4 cascadeMatrix[3]; // world → (u, v, depth) of cascade layer i
    vec4 cascadeSplits;    // xyz = view depth where cascade i ends; w = cascade count (0: no shadow map)
    vec4 cascadeBands;     // xyz = width (m) of the band before each split that dithers into the next cascade (the
                           // last one fades out); w = filter: 1 hardware tap, 2 rotated Poisson-16, 3 four nearest taps
    vec4 cascadeTexel;     // xyz = world size of one texel of cascade i; w = map size in texels
    vec4 cascadeFilter;    // xyz = Poisson radius of cascade i in UV; w = 1: static geometry casts into the cascades
    vec4 cameraForward;    // xyz = unit view direction (cascades are chosen by view depth)
    mat4 spotMatrix[4];    // per spot shadow slot: world → atlas (u, v, depth) × w
    vec4 spotTile[4];      // per slot: the tile's UV rectangle (min.xy, max.zw), inset so filter taps stay inside
    vec4 spotTexel;        // per slot: world size of one texel per meter of distance from the light
    vec4 cameraUp;         // xyz = the view's up axis in world space: billboards build their basis from it (M7), so it
                           // never degenerates for anything on screen
    // Post-processing (M8 plan D13).
    vec4 screen;           // xy = scene target size in pixels, zw = 1 / size
    vec4 post;             // x = exposure; y = 1: the forward shaders apply the curve (LDR path); z = tonemapper
                           // (0 ACES, 1 Hable, 2 clamp); w = 1: set 0 binding 7 holds this frame's SSAO
    vec4 postParams;       // x = 1: bindings 8 and 9 hold this frame's refraction grab and its linear depth; y = 1: SSR
                           // on refractive surfaces; zw unused
} frame;

// Per-draw constants, 128 bytes (ARCHITECTURE.md §4.7). Push constants on Vulkan; a UBO range on GL.
layout(push_constant, std430) uniform PerDraw {
    vec4 modelRows[3]; // affine object→world, including position dequantization
    vec4 lightmapST;   // lightmapped: UV1 scale.xy, offset.zw; dynamic: x = sun visibility from the probes
    uvec4 lights;      // x, y: 8 light indices; z: count (bits 0–3) | lightmap page (4–11) | reflection slot (12–15)
    vec4 sh[3];        // dynamic: L1 SH irradiance rows R, G, B (E/π = sh.w + dot(sh.xyz, n))
} pd;

// Final color for passes that write the backbuffer directly.
vec4 flEncodeOutput(vec4 linearColor) {
    if (frame.outputParams.x > 0.5) linearColor.rgb = flLinearToSrgb(linearColor.rgb);
    return linearColor;
}

#endif
