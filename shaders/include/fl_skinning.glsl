#ifndef FL_SKINNING_GLSL
#define FL_SKINNING_GLSL

// GPU linear blend skinning (ARCHITECTURE.md R3; M6 plan D9), for OBJECT_SKINNED variants. The palette holds up to 128
// affine rows per joint, grid coordinates → world space with the dequantization folded in, bound at set 3, binding 0
// with a dynamic offset into the frame's uniform ring. The block name is part of the GL binding ABI (fl_bindings.glsl).
layout(std140, set = 3, binding = 0) uniform BonePalette {
    vec4 rows[3 * 128]; // mirrored by renderer::kPaletteBytes
} palette;

layout(location = 5) in uvec4 a_joints; // RGBA8_UINT: indices into the mesh's palette
layout(location = 6) in vec4 a_weights; // RGBA8_UNORM: they sum to exactly 1

// The weighted sum of the vertex's four palette entries. Blending the matrices equals blending the transformed points,
// and normals and tangents then use the same cofactor formula as rigid draws.
void flSkinRows(out vec4 rows[3]) {
    rows[0] = vec4(0.0);
    rows[1] = vec4(0.0);
    rows[2] = vec4(0.0);
    for (int k = 0; k < 4; ++k) {
        float w = a_weights[k];
        uint j = a_joints[k] * 3u;
        rows[0] += w * palette.rows[j];
        rows[1] += w * palette.rows[j + 1u];
        rows[2] += w * palette.rows[j + 2u];
    }
}

// The bone-weights view (M6 plan D12): a hashed color per joint, so neighbouring joints differ, blended by the weights.
vec3 flJointColor(uint joint) {
    return 0.25 + 0.75 * fract(vec3(float(joint)) * vec3(0.6180339, 0.4142135, 0.7320508) + vec3(0.12, 0.47, 0.81));
}
vec3 flBoneWeightColor() {
    return flJointColor(a_joints.x) * a_weights.x + flJointColor(a_joints.y) * a_weights.y + flJointColor(a_joints.z) * a_weights.z +
           flJointColor(a_joints.w) * a_weights.w;
}

#endif
