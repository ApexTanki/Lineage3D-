#version 450
// Shadow casters, fragment stage (M4 plan D7): nothing to write but depth; alpha-tested casters discard like the
// forward pass (same albedo alpha, cutoff and UVs), so cut-out leaves and fences cast cut-out shadows.
#include "fl_common.glsl"
#include "fl_material.glsl"

#ifdef ALPHA_TEST
layout(location = 0) in vec2 v_uv0;
#endif

void main() {
#ifdef ALPHA_TEST
    if (texture(u_albedo, v_uv0).a * material.baseColor.a < material.uvAnim.w) discard;
#endif
}
