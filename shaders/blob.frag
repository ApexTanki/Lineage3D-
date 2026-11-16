#version 450
// Blob shadows, fragment stage (M4 plan D9): a soft disc that multiplies the scene color (blend dst × src), so it darkens
// whatever lighting is already there. Fog fades it, so fogged distance keeps the fog color.
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_fog.glsl"

layout(location = 0) in vec2 v_local;
layout(location = 1) in vec3 v_world;
layout(location = 2) flat in float v_strength;
layout(location = 0) out vec4 o_color;

void main() {
    float shade = v_strength * (1.0 - smoothstep(0.62, 1.0, length(v_local))); // full up to the footprint, gone at the rim
    shade *= 1.0 - flFogFactor(v_world);
    o_color = vec4(vec3(1.0 - shade), 1.0);
}
