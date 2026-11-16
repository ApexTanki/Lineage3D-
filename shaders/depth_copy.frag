#version 450
// Pass 5c (ARCHITECTURE.md §6, M7 plan D8): the scene depth as linear view depth in meters, for soft particles (GL can't
// sample a depth buffer that is attached while it draws, so pass B reads this copy). Both backends' depth buffers hold
// the canonical value (0 at the near plane, 1 at the far plane; ARCHITECTURE.md §4.2), so one formula serves both.
//   MSAA  the depth is multisampled: sample 0 (the particles fade per pixel, not per sample)
#include "fl_common.glsl"
#include "fl_bindings.glsl"

#ifdef MSAA
layout(set = 1, binding = 1) uniform sampler2DMS u_depth; // binding 0 of set 1 is a uniform-buffer slot on GL
#else
layout(set = 1, binding = 1) uniform sampler2D u_depth;
#endif

layout(location = 0) out float o_depth;

void main() {
    float d = texelFetch(u_depth, ivec2(gl_FragCoord.xy), 0).r;
    float n = pd.modelRows[0].x, f = pd.modelRows[0].y; // near and far planes (m)
    o_depth = n * f / (f - d * (f - n));
}
