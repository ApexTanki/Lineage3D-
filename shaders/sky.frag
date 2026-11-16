#version 450
// Skybox, fragment stage (M3 plan D8–D10): the sky cube's top mip, the sun disc (drawn here, never baked, so neither
// lighting nor reflections count the sun twice) and fog over the horizon.
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_environment.glsl"
#include "fl_fog.glsl"
#include "fl_post.glsl"

layout(set = 0, binding = 4) uniform samplerCube u_sky;

layout(location = 0) in vec4 v_direction;
layout(location = 0) out vec4 o_color;

void main() {
    vec3 d = normalize(v_direction.xyz / v_direction.w);
    vec3 color = flDecodeHdr(textureLod(u_sky, d, 0.0)) * frame.ambientGround.w;
    float cosDisc = frame.sunDirection.w;
    // max(): a zero disc radius makes both edges 1, and smoothstep with equal edges is undefined (NaN on some GPUs).
    float disc = smoothstep(cosDisc, cosDisc + max((1.0 - cosDisc) * 0.4, 1e-6), dot(d, frame.sunDirection.xyz));
    color += frame.sunColor.rgb * (frame.sunColor.w * disc);
    color = mix(color, frame.fogColor.rgb, flSkyFogFactor(d));
    // Data views show the sky black, so only surfaces carry data.
    if (int(frame.outputParams.z + 0.5) != 0) color = vec3(0.0);
    o_color = vec4(flForwardOutput(color), 1.0); // LDR path: the curve (M8 plan D8)
}
