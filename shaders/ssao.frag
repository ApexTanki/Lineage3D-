#version 450
// Pass 2b, SSAO (High; ARCHITECTURE.md §6, DEV-1, M8 plan D9), at half resolution from the linear depth of the prepass.
//   MODE  AO    12 samples in a normal-oriented hemisphere (normals rebuilt from depth), rotated per pixel by
//               interleaved gradient noise, each compared with the depth where it projects, with a range check
//         BLUR  a separable 9-tap blur whose weights fall off with depth difference, so AO doesn't bleed across edges
// World positions come from the linear depth along each pixel's view ray (frame.skyFromClip), so both backends' clip
// conventions are handled where the frame matrices are built.
#include "fl_common.glsl"
#include "fl_bindings.glsl"

layout(set = 1, binding = 1) uniform sampler2D u_depth; // linear view depth in meters, full resolution
layout(set = 1, binding = 2) uniform sampler2D u_ao;    // BLUR: the AO to blur (half resolution)

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_ao;

float depthAt(ivec2 p) {
    ivec2 size = textureSize(u_depth, 0);
    return texelFetch(u_depth, clamp(p, ivec2(0), size - 1), 0).r;
}

#ifdef MODE_AO
// The world position of full-resolution pixel p at linear depth d.
vec3 worldAt(ivec2 p, float d) {
    vec2 uv = (vec2(p) + 0.5) / vec2(textureSize(u_depth, 0));
    vec4 h = frame.skyFromClip * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 dir = h.xyz / h.w;
    return frame.cameraPosition.xyz + dir * (d / dot(dir, frame.cameraForward.xyz));
}

// Per-draw: modelRows[0] = (radius m, intensity, far plane m, unused).
void main() {
    vec4 params = pd.modelRows[0];
    // The top-left full-resolution pixel of this texel's 2×2 block (v_uv × size = gl_FragCoord.xy on both backends).
    ivec2 p = ivec2(v_uv * vec2(textureSize(u_depth, 0) / 2)) * 2;
    float d = depthAt(p);
    if (d >= params.z * 0.999) { // the sky
        o_ao = vec4(1.0);
        return;
    }
    vec3 position = worldAt(p, d);
    // The normal from the flatter neighbour on each axis, so silhouettes don't tilt it.
    float dl = depthAt(p - ivec2(1, 0)), dr = depthAt(p + ivec2(1, 0)), du = depthAt(p - ivec2(0, 1)), dd = depthAt(p + ivec2(0, 1));
    vec3 ddx = abs(dr - d) < abs(d - dl) ? worldAt(p + ivec2(1, 0), dr) - position : position - worldAt(p - ivec2(1, 0), dl);
    vec3 ddy = abs(dd - d) < abs(d - du) ? worldAt(p + ivec2(0, 1), dd) - position : position - worldAt(p - ivec2(0, 1), du);
    vec3 n = normalize(cross(ddx, ddy));
    if (dot(n, frame.cameraPosition.xyz - position) < 0.0) n = -n;
    // A tangent frame around n, turned by interleaved gradient noise (Jimenez) so the 12 directions differ per pixel.
    float noise = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float angle = noise * 6.2831853;
    vec3 helper = abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(helper, n));
    vec3 b = cross(n, t);
    vec3 tr = t * cos(angle) + b * sin(angle);
    vec3 br = cross(n, tr);
    const vec3 kKernel[12] = vec3[12](vec3(0.53, 0.18, 0.35), vec3(-0.31, 0.62, 0.22), vec3(-0.58, -0.27, 0.41), vec3(0.24, -0.66, 0.28),
                                      vec3(0.71, 0.33, 0.52), vec3(-0.12, 0.81, 0.47), vec3(-0.79, 0.05, 0.36), vec3(0.08, -0.84, 0.44),
                                      vec3(0.41, 0.46, 0.78), vec3(-0.47, 0.39, 0.74), vec3(-0.36, -0.52, 0.72), vec3(0.49, -0.37, 0.75));
    vec2 size = vec2(textureSize(u_depth, 0));
    float radius = params.x;
    float occlusion = 0.0;
    for (int i = 0; i < 12; ++i) {
        float scale = mix(0.15, 1.0, float(i * i) / 121.0); // more samples close to the point
        vec3 k = normalize(kKernel[i]) * scale;
        vec3 s = position + (tr * k.x + br * k.y + n * k.z) * radius;
        vec4 clip = frame.viewProj * vec4(s, 1.0);
        if (clip.w <= 0.0) continue;
        vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) continue;
        float sampleDepth = depthAt(ivec2(uv * size));
        float expected = dot(s - frame.cameraPosition.xyz, frame.cameraForward.xyz);
        float range = smoothstep(0.0, 1.0, radius / max(abs(d - sampleDepth), 1e-4)); // far occluders don't count
        occlusion += (sampleDepth < expected - 0.02 * radius ? 1.0 : 0.0) * range;
    }
    o_ao = vec4(clamp(1.0 - params.y * occlusion / 12.0, 0.0, 1.0));
}
#else
// Per-draw: modelRows[0] = (direction x, direction y in half-res texels, unused, unused).
void main() {
    ivec2 size = textureSize(u_ao, 0);
    ivec2 p = ivec2(v_uv * vec2(size)); // v_uv × size = gl_FragCoord.xy on both backends
    ivec2 step = ivec2(pd.modelRows[0].xy);
    float center = depthAt(p * 2);
    const float kWeights[5] = float[5](0.227, 0.194, 0.122, 0.054, 0.016); // Gaussian, sigma ≈ 2 texels
    float sum = texelFetch(u_ao, p, 0).r * kWeights[0];
    float total = kWeights[0];
    for (int i = 1; i < 5; ++i)
        for (int s = -1; s <= 1; s += 2) {
            ivec2 q = clamp(p + step * (i * s), ivec2(0), size - 1);
            float w = kWeights[i] * exp(-abs(depthAt(q * 2) - center) * 8.0 / max(center, 0.1)); // relative depth difference
            sum += texelFetch(u_ao, q, 0).r * w;
            total += w;
        }
    o_ao = vec4(sum / total);
}
#endif
