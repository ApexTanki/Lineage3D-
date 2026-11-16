#version 450
// Debug geometry passes, fragment stage (M5 plan D10): the pass color, as is.

layout(std140, set = 1, binding = 0) uniform DebugPass {
    mat4 viewProj;
    vec4 color;
} pass;

layout(location = 0) out vec4 o_color;

void main() {
    o_color = pass.color;
}
