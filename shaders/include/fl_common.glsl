#ifndef FL_COMMON_GLSL
#define FL_COMMON_GLSL

// Shared helpers. Sources are GLSL 450 compiled to SPIR-V and cross-compiled to GLSL 330,
// so only features from the GL 3.3 core floor may be used (ARCHITECTURE.md §4.10).

#ifdef FL_STAGE_VERTEX
// The depth prepass and forward passes are different programs; invariance keeps their depths equal.
invariant gl_Position;
#endif

vec3 flSrgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

vec3 flLinearToSrgb(vec3 c) {
    c = max(c, vec3(0.0));
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

// Affine object→world transform stored as three rows (ARCHITECTURE.md §4.1).
vec3 flTransformPoint(vec4 rows[3], vec3 p) {
    vec4 h = vec4(p, 1.0);
    return vec3(dot(rows[0], h), dot(rows[1], h), dot(rows[2], h));
}

// Directions that move with the surface (tangents): the linear part only. Not normalized.
vec3 flTransformDirection(vec4 rows[3], vec3 v) {
    return vec3(dot(rows[0].xyz, v), dot(rows[1].xyz, v), dot(rows[2].xyz, v));
}

// -1 when the transform mirrors (negative determinant), else 1.
float flMirrorSign(vec4 rows[3]) {
    return dot(rows[0].xyz, cross(rows[1].xyz, rows[2].xyz)) < 0.0 ? -1.0 : 1.0;
}

// Normals go through the cofactor of the linear part, times sign(det): the inverse-transpose direction without a
// division, correct under non-uniform scale and mirroring (same formula as core::Affine3x4::transformNormal).
// Not normalized.
vec3 flTransformNormal(vec4 rows[3], vec3 n) {
    vec3 a = rows[0].xyz, b = rows[1].xyz, c = rows[2].xyz;
    vec3 bc = cross(b, c);
    float s = dot(a, bc) < 0.0 ? -1.0 : 1.0;
    return s * vec3(dot(bc, n), dot(cross(c, a), n), dot(cross(a, b), n));
}

#endif
