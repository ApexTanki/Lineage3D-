#pragma once

// Baked map package (ARCHITECTURE.md §7.7, M3 plan D2), written by `fl_assetc map` and read by world::RenderWorld:
//   map.json        runtime manifest: sun, ambient, sky, fog, lightmap files, probe file, reflection probes, instances
//   lightmaps.ktx2  BC6H 2D array (one layer per page); lightmaps_rgbm.ktx2: the same as RGBM (M = 8) for devices
//                   without BC6H; shadowmask.ktx2: sun visibility, BC4 array
//   probes.fprobe   the L1 SH probe grid (binary, below)
//   sky*.ktx2, reflection<i>*.ktx2   HDR cubes with gloss-prefiltered mips (BC6H + RGBM variants)
// Lightmap texels hold E/π (the response of a white Lambertian surface, the engine's light unit) of everything except
// the sun's direct term, which shaders compute live and mask with the shadowmask (§4.5, D-SHADOWMASK).

#include "core/Math.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace assets {

// Distance and height fog (M3 plan D10). Distance fog: linear between start and end, or 1 − e^(−density·d) (exp),
// or 1 − e^(−(density·d)²) (exp2). Height fog adds an exponential layer with density heightDensity at heightBase that
// falls off by e^(−heightFalloff·Δy), integrated along the view ray. The sky takes distance fog over a band of view
// directions above the horizon (skyHorizon, in direction y) and height fog integrated to infinity.
struct FogSettings {
    enum class Mode : uint8_t { Off, Linear, Exp, Exp2 };
    Mode mode = Mode::Off;
    glm::vec3 color{0.62f, 0.70f, 0.80f}; // linear
    float start = 30.0f;
    float end = 400.0f;
    float density = 0.004f;
    float maxOpacity = 1.0f;
    bool height = false;
    float heightBase = 0.0f;
    float heightDensity = 0.04f;
    float heightFalloff = 0.35f;
    float skyHorizon = 0.15f;
};

const char* fogModeName(FogSettings::Mode mode);
bool parseFogMode(const char* name, FogSettings::Mode& out);

// Lightmap placement of one mesh node of a static instance: UV1 · scale + offset inside layer `page`.
struct MapLightmapRect {
    int32_t page = -1; // −1: not lightmapped
    glm::vec4 scaleOffset{0.0f};
};

struct MapInstance {
    std::string model;                  // data-relative .fmodel
    std::vector<std::string> materials; // per material slot; "" = the model's own material
    glm::mat4 transform{1.0f};
    bool isStatic = true;               // static: lightmapped and instanced; dynamic: probe-lit, may move
    bool occluder = false;              // static only: its mesh boxes occlude for the camera (M5 plan D4)
    uint8_t layer = 1;
    std::vector<MapLightmapRect> lightmaps; // one per mesh node, in node order (static only)
    uint8_t reflectionProbe = 0;            // 0 = the sky; i = reflectionProbes[i − 1]
};

struct MapReflectionProbe {
    std::string cube;     // BC6H
    std::string cubeRgbm; // RGBM fallback
    glm::vec3 position{0.0f};
    glm::vec3 boxMin{0.0f};
    glm::vec3 boxMax{0.0f};
};

inline constexpr uint32_t kMaxReflectionProbes = 7; // + the sky = 8 slots (4-bit index, FrameUniforms array)

struct MapManifest {
    glm::vec3 sunDirection{0.0f, 1.0f, 0.0f}; // toward the sun, normalized
    glm::vec3 sunColor{1.0f};                 // color × intensity
    glm::vec3 ambientSky{0.3f};               // hemispherical fallback for objects with neither lightmap nor probe
    glm::vec3 ambientGround{0.1f};
    std::string skyCube;                      // BC6H cube, gloss-prefiltered mips
    std::string skyCubeRgbm;
    float skyIntensity = 1.0f;
    float sunDiscIntensity = 30.0f;           // sun disc radiance in the sky pass, × sunColor
    float sunDiscRadiusDeg = 0.6f;
    float exposure = 1.0f;                    // scene color × this before the tone curve (M8 plan D2; R8: a per-map value)
    FogSettings fog;
    std::string lightmaps;                    // BC6H array
    std::string lightmapsRgbm;
    std::string shadowmask;
    uint32_t lightmapPages = 0;
    uint32_t lightmapSize = 0;                // page width = height, texels
    std::string probes;                       // .fprobe; empty = no probe grid
    std::vector<MapReflectionProbe> reflectionProbes;
    std::vector<MapInstance> instances;
};

bool parseMapManifest(std::span<const std::byte> json, MapManifest& out, std::string* error);
std::string writeMapManifest(const MapManifest& manifest);

// probes.fprobe: header, then dims.x·dims.y·dims.z probes, x fastest. Each probe stores L1 SH irradiance in its
// evaluated form (M3 plan D6): E/π(n) = sh[c].w + dot(sh[c].xyz, n) for c = R, G, B; plus the fraction of the sun disc
// the probe sees and whether it lies outside geometry.
inline constexpr char kProbeMagic[4] = {'F', 'P', 'R', 'B'};
inline constexpr uint16_t kProbeVersion = 1;

struct ProbeGridData {
    struct Probe {
        glm::vec4 sh[3]{};
        float sunOcclusion = 1.0f; // 1 = the sun is fully visible
        float valid = 1.0f;        // 0: inside geometry, skipped by the interpolation
        float pad[2] = {};
    };
    uint32_t dims[3] = {0, 0, 0};
    glm::vec3 origin{0.0f};  // world position of probe (0, 0, 0)
    glm::vec3 spacing{1.0f}; // meters between neighbors
    std::vector<Probe> probes;
};
static_assert(sizeof(ProbeGridData::Probe) == 64, "probe records are 64 bytes");

bool readProbeGrid(std::span<const std::byte> file, ProbeGridData& out, std::string* error);
std::vector<std::byte> writeProbeGrid(const ProbeGridData& grid);

} // namespace assets
