#pragma once

#include "core/Math.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

// Particle presets (.fpart, JSON; ARCHITECTURE.md §7.8, M7 plan D2). An effect is a list of emitters; each emitter
// spawns billboards from a shape, moves them under gravity and drag, and shades them with curves over their life:
//
//   { "name": "explosion",
//     "emitters": [
//       { "name": "smoke", "max": 24, "rate": 0, "burst": [[0.05, 16]], "delay": 0, "duration": 0, "loop": false,
//         "life": [1.5, 2.5], "offset": [0, 0.5, 0], "shape": "sphere", "radius": 0.6, "direction": [0, 1, 0], "cone": 60, "speed": [0.5, 2],
//         "inherit": 0, "gravity": -0.4, "drag": 1.2,
//         "size": [[0, 1.0], [1, 3.5]], "sizeRandom": [0.8, 1.2],
//         "color": [[0, [0.4, 0.38, 0.36, 1]], [1, [0.6, 0.58, 0.56, 1]]], "alpha": [[0, 0], [0.1, 0.8], [1, 0]],
//         "intensity": 1, "rotation": [0, 360], "spin": [-40, 40],
//         "billboard": "camera", "blend": "alpha", "texture": "textures/fx_smoke.ktx2",
//         "flipbook": { "cols": 4, "rows": 4, "fps": 12, "interpolate": true, "randomStart": true },
//         "soft": 0.6, "lit": true, "lod": [40, 160] },
//       { "name": "sparks", "billboard": "velocity", "stretch": 0.04, "collide": { "restitution": 0.4, "friction": 0.3 }, ... },
//       { "name": "ring", "billboard": "world", "normal": [0, 1, 0], ... },
//       { "name": "flash", "light": { "color": [1, 0.7, 0.4], "intensity": 8, "range": 12, "life": 0.2 }, ... } ] }
//
// Parsing is strict, like materials: unknown keys, keys that don't apply (a radius on a point shape, a stretch on a
// camera billboard), out-of-range values and emitters that can never spawn fail the load, and fl_assetc runs the same
// parser when it cooks, so a bad preset fails the build. Curves are piecewise-linear keys [[t, value], ...] over the
// particle's normalized age (t from 0 to 1, strictly increasing), or a constant; they are baked into kCurveSamples
// evenly spaced samples, so a lookup is two reads and a lerp. Colors are linear and at most 1 per channel (`intensity`
// brightens an emitter); the `alpha` curve multiplies the color curve's alpha and is folded into it when baking.
// Times are seconds; angles in the file are degrees and radians here.

namespace assets {

inline constexpr uint32_t kCurveSamples = 32;
inline constexpr uint32_t kMaxEffectEmitters = 8;
inline constexpr uint32_t kMaxEmitterBursts = 8;
inline constexpr uint32_t kMaxEmitterParticles = 4096;
inline constexpr uint32_t kMaxCurveKeys = 8;
inline constexpr uint32_t kMaxFlipbookCells = 32; // columns and rows each: frame numbers stay exact in half floats

// Distort (heat haze, M8 plan D11): the texture is a normal map whose XY bends the picture behind; drawn after the
// transparents (pass 8), never lit, and not drawn at all when heat haze is off.
enum class ParticleBlend : uint8_t { Alpha, Additive, Premultiplied, Distort };
enum class ParticleBillboard : uint8_t {
    Camera,   // faces the camera, turned by its rotation
    AxisY,    // upright, turned about world Y toward the camera (flames, dust columns)
    Velocity, // along its screen-projected velocity, stretched by speed (sparks, tracers)
    World     // a fixed world-space normal (a shockwave ring on the ground), turned about it by its rotation
};
enum class ParticleShape : uint8_t { Point, Sphere, Disc, Box };

// A curve over normalized age, as evenly spaced samples: samples[i] is the value at t = i / (kCurveSamples − 1).
template <typename T>
struct ParticleCurve {
    std::array<T, kCurveSamples> samples{};
    T operator()(float t) const {
        const float c = t > 0.0f ? (t < 1.0f ? t : 1.0f) : 0.0f; // also maps NaN to 0
        const float x = c * float(kCurveSamples - 1);
        const uint32_t i = x < float(kCurveSamples - 2) ? uint32_t(x) : kCurveSamples - 2;
        return samples[i] + (samples[i + 1] - samples[i]) * (x - float(i));
    }
};
using ScalarCurve = ParticleCurve<float>;
using ColorCurve = ParticleCurve<glm::vec4>;

struct ParticleBurst {
    float time = 0.0f;  // seconds after the emitter starts (within each period when it loops)
    uint32_t count = 0; // 1 – max
};

struct ParticleLight {
    glm::vec3 color{1.0f}; // linear
    float intensity = 1.0f;
    float range = 1.0f;    // meters
    float life = 0.1f;     // seconds; the intensity fades to 0 over it
};

struct ParticleEmitterDesc {
    std::string name;
    uint32_t max = 0;                    // particles alive at once (its budget and its block in the arena)
    float rate = 0.0f;                   // particles per second while it runs
    ParticleBurst bursts[kMaxEmitterBursts]{};
    uint32_t burstCount = 0;
    float delay = 0.0f;                  // seconds after the effect starts
    float duration = 0.0f;               // seconds of rate spawning; the period when it loops
    bool loop = false;
    glm::vec2 life{1.0f};                // seconds, min and max
    glm::vec3 offset{0.0f};              // effect space: where the shape sits relative to the effect's origin
    ParticleShape shape = ParticleShape::Point;
    float radius = 0.0f;                 // sphere (volume) and disc (in the effect's XZ plane)
    glm::vec3 extents{0.0f};             // box half sizes
    glm::vec3 direction{0.0f, 1.0f, 0.0f}; // effect space, unit
    float cone = 0.0f;                   // half angle around the direction, radians (π = every direction)
    glm::vec2 speed{0.0f};               // m/s, min and max
    float inherit = 0.0f;                // share of the effect's own velocity given to new particles
    float gravity = 0.0f;                // m/s² downward (negative rises)
    float drag = 0.0f;                   // 1/s: velocity loses this share per second
    ScalarCurve size;                    // meters over life
    glm::vec2 sizeRandom{1.0f};          // per-particle size multiplier, min and max
    ColorCurve color;                    // linear RGBA over life, alpha curve folded in; each channel ≤ 1
    float intensity = 1.0f;              // multiplies the color (HDR brightness)
    glm::vec2 rotation{0.0f};            // initial rotation range, radians
    glm::vec2 spin{0.0f};                // radians per second, min and max
    ParticleBillboard billboard = ParticleBillboard::Camera;
    glm::vec3 normal{0.0f, 1.0f, 0.0f};  // world billboards: world space, unit
    float stretch = 0.0f;                // velocity billboards: extra length (m) per m/s
    ParticleBlend blend = ParticleBlend::Alpha;
    std::string texture;                 // data-relative .ktx2 atlas; empty = a white square
    uint32_t columns = 1, rows = 1;      // flipbook grid, cells read row by row from the top left (Flipbook.h)
    uint32_t frameCount = 1;             // cells used, from cell 0
    float fps = 0.0f;                    // frames per second, or…
    bool overLife = false;               // … the frames spread evenly over each particle's life
    bool flipbookLoop = true;            // fps mode: wrap (else hold the last frame)
    bool interpolate = false;            // blend each frame into the next (P1)
    bool randomStart = false;            // each particle starts on a random frame (without fps or overLife: keeps it)
    float soft = 0.0f;                   // depth fade distance against the scene (m); 0 = hard edges
    bool lit = false;                    // ambient + sun + probe tint; otherwise emissive
    float strength = 0.1f;               // distort: m the texture's XY moves the picture behind at full opacity
    bool collide = false;                // bounce off the effect's ground plane
    float restitution = 0.0f;            // share of the normal speed kept by a bounce
    float friction = 0.0f;               // share of the tangential speed lost by a bounce
    glm::vec2 lod{0.0f};                 // spawn distance LOD: full rate within lod.x, ¼ at lod.y, none beyond; y 0 = off
    bool hasLight = false;
    ParticleLight light;                 // a dynamic point light started with the emitter
};

struct ParticleEffectFile {
    std::string name;
    std::vector<ParticleEmitterDesc> emitters; // 1 – kMaxEffectEmitters
    bool loops = false;                        // some emitter loops: the effect runs until stopped
    uint32_t particleCapacity = 0;             // the sum of the emitters' max
};

bool parseParticleEffect(std::span<const std::byte> json, ParticleEffectFile& out, std::string* error);
// The invariants the runtime relies on (counts within the fixed arrays and pools, finite values, unit directions, a
// looping period of at least 0.01 s, a flipbook within its grid): what parseParticleEffect guarantees, for presets built
// in code (ResourceManager::addPreset refuses one that fails).
bool validateParticleEffect(const ParticleEffectFile& effect, std::string* error);

const char* particleBlendName(ParticleBlend blend);
const char* particleBillboardName(ParticleBillboard billboard);

} // namespace assets
