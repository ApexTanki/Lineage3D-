#ifndef FL_MATERIAL_GLSL
#define FL_MATERIAL_GLSL

// Set 2: per-material resources (ARCHITECTURE.md §4.6). Mirrored by renderer::MaterialUniforms. Colors are linear.
layout(std140, set = 2, binding = 0) uniform MaterialUniforms {
    vec4 baseColor;     // rgb tint, a opacity (multiply the albedo map)
    vec4 specularGloss; // rgb specular color, a gloss 0–1 (multiply the spec/gloss map when there is one)
    vec4 emission;      // rgb color (multiplies the emission map when there is one), a intensity
    vec4 uvAnim;        // xy = UV scroll (units/s), z = UV rotation (rad/s, about (0.5, 0.5)); w = alpha cutoff
    vec4 flipbook;      // x = columns, y = rows, z = frames per second (0 = static)
    vec4 detail;        // xy = detail scroll (units/s), z = detail tiling, w = detail strength
    vec4 misc;          // x = reflection strength, y = 1: additive (fog fades it out), z = normal-map strength,
                        // w = 1: premultiplied output (diffuse × alpha)
    vec4 refraction;    // refractive (water, glass; M8 plan D10): rgb what passes through, a bend strength in m (0: none)
} material;

#ifdef FL_STAGE_FRAGMENT
// Every material set binds all five textures (absent ones point at 1×1 defaults); a variant samples only the maps its
// features name.
layout(set = 2, binding = 1) uniform sampler2D u_albedo;    // sRGB
#ifdef NORMAL_MAP
layout(set = 2, binding = 2) uniform sampler2D u_normal;    // linear RG (BC5), Z reconstructed
#endif
#ifdef SPEC_GLOSS_MAP
layout(set = 2, binding = 3) uniform sampler2D u_specGloss; // sRGB rgb specular color, linear alpha gloss
#endif
#ifdef EMISSION_MAP
layout(set = 2, binding = 4) uniform sampler2D u_emission;  // sRGB
#endif
#ifdef DETAIL_LAYER
layout(set = 2, binding = 5) uniform sampler2D u_detail;    // sRGB RGBA: blended over the albedo by alpha × strength
#endif
#endif

// Texture animation of UV0 (M3 plan D11), per vertex (every step is affine, so it interpolates exactly). A flipbook maps
// the mesh's UVs (one cell, 0–1) into the current cell; it can't also scroll or rotate, because the UVs would leave the
// cell (the material parser rejects the combination). Otherwise: rotate about (0.5, 0.5), then scroll; scroll offsets
// wrap with fract() (the textures repeat), so precision doesn't decay as time grows.
vec2 flAnimateUv(vec2 uv, float time) {
    vec2 grid = material.flipbook.xy;
    if (grid.x * grid.y > 1.0) {
        float frames = grid.x * grid.y;
        float frame = mod(floor(time * material.flipbook.z), frames);
        return (uv + vec2(mod(frame, grid.x), floor(frame / grid.x))) / grid;
    }
    float angle = material.uvAnim.z * time;
    if (angle != 0.0) {
        float c = cos(angle), s = sin(angle);
        uv = mat2(c, s, -s, c) * (uv - 0.5) + 0.5;
    }
    return uv + fract(material.uvAnim.xy * time);
}

#endif
