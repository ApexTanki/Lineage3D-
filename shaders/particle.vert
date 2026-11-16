#version 450
// Particles (M7 plan D5–D7, D9): one instanced draw per emitter, six vertices per particle from gl_VertexIndex (no vertex
// buffer) and 32 bytes of instance data each (ARCHITECTURE.md §7.8). The per-draw block holds the emitter's settings
// (renderer::ParticleDrawConstants): the PerDraw layout of fl_bindings.glsl, read with its own meaning (macros below).
#include "fl_common.glsl"
#include "fl_bindings.glsl"

#define P_BILLBOARD pd.modelRows[0].x // 0 camera, 1 axis (Y), 2 velocity, 3 world
#define P_INTENSITY pd.modelRows[0].w
#define P_FLIPBOOK pd.modelRows[1]    // columns, rows, frame count, interpolate
#define P_NORMAL pd.modelRows[2].xyz  // world billboards
#define P_LIT pd.modelRows[2].w
#define P_SUN_VISIBILITY pd.lightmapST.x

layout(location = 8) in vec3 i_position;     // world center
layout(location = 9) in vec4 i_sizeRotation; // width, height (m), rotation (rad), frame blend
layout(location = 10) in vec4 i_color;       // linear RGB, opacity
layout(location = 11) in vec4 i_axisFrame;   // xyz = velocity (velocity billboards), w = frame

layout(location = 0) out vec4 v_uv;        // xy: the frame's cell, zw: the next frame's (interpolation)
layout(location = 1) out vec3 v_world;
layout(location = 2) out vec4 v_color;     // lit or emissive color × intensity, opacity
layout(location = 3) flat out float v_frameBlend;
layout(location = 4) out float v_viewDepth; // soft particles: meters along the view direction
#ifdef DISTORT
layout(location = 5) out vec4 v_axes; // heat haze (M8 plan D11): the quad's right and up axes as screen-UV offsets per meter
#endif

// The UV rectangle (offset xy, scale zw) of a flipbook cell, read row by row from the top left (renderer/vfx/Flipbook.h).
// Integer arithmetic: a float division may land a hair below a whole number (6 / 3 → 1.9999999) and pick the wrong row.
vec4 flipbookCell(int frame, int columns, int rows) {
    int cell = frame % (columns * rows);
    int row = cell / columns;
    return vec4(float(cell - row * columns) / float(columns), float(row) / float(rows), 1.0 / float(columns), 1.0 / float(rows));
}

void main() {
    // Two triangles: (−1,−1) (1,−1) (1,1) and (−1,−1) (1,1) (−1,1).
    int i = gl_VertexIndex;
    vec2 corner = vec2(i == 1 || i == 2 || i == 4 ? 1.0 : -1.0, i == 2 || i == 4 || i == 5 ? 1.0 : -1.0);
    vec3 toCamera = frame.cameraPosition.xyz - i_position;
    vec3 f = dot(toCamera, toCamera) > 1e-8 ? normalize(toCamera) : vec3(0.0, 0.0, 1.0);
    int mode = int(P_BILLBOARD + 0.5);
    float rotation = i_sizeRotation.z;
    vec3 right, up;
    vec3 v = i_axisFrame.xyz;
    vec3 along = v - dot(v, f) * f; // the velocity as the camera sees it
    if (mode == 2 && dot(along, along) > 1e-8) {
        // Velocity-stretched: the long side (height) along the screen-projected velocity (sparks, tracers).
        up = normalize(along);
        right = cross(up, f);
    } else if (mode == 1) {
        // Axis-locked: upright, turned about world Y toward the camera.
        vec3 level = vec3(toCamera.x, 0.0, toCamera.z);
        vec3 g = dot(level, level) > 1e-8 ? normalize(level) : vec3(0.0, 0.0, 1.0);
        right = vec3(g.z, 0.0, -g.x);
        up = vec3(0.0, 1.0, 0.0);
    } else {
        // Camera-facing, or world-aligned to a fixed normal; both turned by the particle's rotation. A velocity
        // billboard moving straight at the camera faces it too. Facing the camera, the basis comes from the view's up
        // axis, which is never parallel to the direction of anything on screen (a world-up helper would flip the quads
        // as that direction crosses the vertical).
        vec3 n = mode == 3 ? P_NORMAL : f;
        vec3 helper = mode == 3 ? (abs(n.y) > 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0)) : frame.cameraUp.xyz;
        vec3 r = normalize(cross(helper, n));
        vec3 u = cross(n, r);
        float c = cos(rotation), s = sin(rotation);
        right = r * c + u * s;
        up = u * c - r * s;
    }
    v_world = i_position + right * (corner.x * 0.5 * i_sizeRotation.x) + up * (corner.y * 0.5 * i_sizeRotation.y);

    // Corner (−1, −1) is the cell's bottom left; v = 0 is the top image row (the flipbook convention).
    vec2 uv = vec2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    int columns = max(int(P_FLIPBOOK.x + 0.5), 1), rows = max(int(P_FLIPBOOK.y + 0.5), 1), frames = max(int(P_FLIPBOOK.z + 0.5), 1);
    int frameIndex = int(i_axisFrame.w + 0.5); // a whole number, exact in half precision up to 2048
    vec4 a = flipbookCell(frameIndex, columns, rows);
    vec4 b = flipbookCell((frameIndex + 1) % frames, columns, rows);
    v_uv = vec4(a.xy + uv * a.zw, b.xy + uv * b.zw);
    v_frameBlend = P_FLIPBOOK.w > 0.5 ? i_sizeRotation.w : 0.0;

    // Cheap lighting (M7 plan D9): the probe irradiance at the emitter plus the sun, wrapped so backlit smoke keeps some
    // of it, with a normal bent from the camera toward the quad's corner so a puff shades like a ball. Per vertex.
    vec3 light = vec3(1.0);
    if (P_LIT > 0.5) {
        vec3 n = normalize(f + (right * corner.x + up * corner.y) * 0.6);
        vec3 ambient = max(vec3(pd.sh[0].w + dot(pd.sh[0].xyz, n), pd.sh[1].w + dot(pd.sh[1].xyz, n), pd.sh[2].w + dot(pd.sh[2].xyz, n)),
                           vec3(0.0));
        light = ambient + frame.sunColor.rgb * (P_SUN_VISIBILITY * (0.35 + 0.65 * max(dot(n, frame.sunDirection.xyz), 0.0)));
    }
    v_color = vec4(i_color.rgb * light * P_INTENSITY, i_color.a);
    v_viewDepth = dot(v_world - frame.cameraPosition.xyz, frame.cameraForward.xyz);
#ifdef DISTORT
    // The quad's axes as screen-UV offsets per meter: the projection's derivative at its center, so the bend follows each
    // backend's screen orientation the way the scene itself was rendered. The center's w is clamped, so a quad centred at
    // or behind the camera plane (flying through the heat) keeps a finite, unflipped bend; the fragment stage fades haze
    // out near the camera anyway.
    vec4 c0 = frame.viewProj * vec4(i_position, 1.0), cr = frame.viewProj * vec4(i_position + right, 1.0);
    vec4 cu = frame.viewProj * vec4(i_position + up, 1.0);
    float w = max(c0.w, 0.25);
    vec2 ndc = c0.xy / w;
    v_axes = vec4((cr.xy - c0.xy) - ndc * (cr.w - c0.w), (cu.xy - c0.xy) - ndc * (cu.w - c0.w)) * (0.5 / w);
#endif
    gl_Position = frame.viewProj * vec4(v_world, 1.0);
}
