#version 450
// Copies the linear scene target to the backbuffer with the output encoding. M8 replaces the clamp with exposure
// and tonemapping. The overdraw view (M5 plan D10) turns the per-pixel draw count into a heat ramp instead.
#include "fl_common.glsl"
#include "fl_bindings.glsl"

layout(set = 1, binding = 1) uniform sampler2D u_scene;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

const int kViewOverdraw = 18;       // renderer::DebugView::Overdraw
const float kOverdrawScale = 32.0;  // 1 / DebugDraw::kOverdrawStep

// Draws covering a pixel → display color: 0 black, 1 dark blue, 2 blue, 3 cyan, 4 green, 6 yellow, 8 orange, 12 red,
// 16 and more white.
vec3 flHeat(float layers) {
    const float kStops[9] = float[9](0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0);
    const vec3 kColors[9] = vec3[9](vec3(0.0), vec3(0.0, 0.0, 0.45), vec3(0.0, 0.25, 1.0), vec3(0.0, 0.85, 0.95), vec3(0.1, 0.9, 0.2),
                                    vec3(0.95, 0.95, 0.1), vec3(1.0, 0.55, 0.0), vec3(0.95, 0.1, 0.05), vec3(1.0));
    for (int i = 1; i < 9; ++i)
        if (layers < kStops[i]) return mix(kColors[i - 1], kColors[i], (layers - kStops[i - 1]) / (kStops[i] - kStops[i - 1]));
    return kColors[8];
}

void main() {
    vec3 color = clamp(texture(u_scene, v_uv).rgb, 0.0, 1.0);
    if (int(frame.outputParams.z + 0.5) == kViewOverdraw) color = flSrgbToLinear(flHeat(texture(u_scene, v_uv).r * kOverdrawScale));
    o_color = flEncodeOutput(vec4(color, 1.0));
}
