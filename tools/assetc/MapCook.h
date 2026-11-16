#pragma once

#include <string>

namespace tools {

// `fl_assetc map` (M3 plan D1–D6): bakes a map source into a map package (assets/MapFormat.h).
//
// Source (assets/source/maps/<name>.map.json; unknown keys are errors):
//   { "lightmap": { "texelsPerMeter": 4, "pageSize": 1024, "maxPages": 4 },
//     "bake": { "samples": 256, "bounces": 2, "sunSamples": 64, "lightSamples": 64, "probeRays": 2048 },
//     "sun": { "direction": [x, y, z] (toward the sun), "color": [r, g, b], "intensity": 2.5, "angularRadiusDeg": 0.7 },
//     "sky": { "zenith", "horizon", "ground", "glow", "cloudColor": [r, g, b], "intensity", "glowExponent", "clouds",
//              "cloudScale", "seed", "faceSize": 512, "sunDiscIntensity": 30, "sunDiscRadiusDeg": 0.6 },
//     "fog": { "mode": "off|linear|exp|exp2", "color", "start", "end", "density", "maxOpacity", "height": true,
//              "heightBase", "heightDensity", "heightFalloff", "skyHorizon" },
//     "probes": { "min": [x, y, z], "max": [x, y, z], "spacing": 4 },
//     "reflectionProbes": [ { "position", "boxMin", "boxMax", "faceSize": 128 } ],
//     "lights": [ { "type": "point|spot", "position", "direction", "color", "intensity", "range", "innerConeDeg",
//                   "outerConeDeg", "radius" } ],                       (baked: they reach dynamic objects via probes)
//     "instances": [ { "model": "models/x.fmodel", "materials": ["materials/a.fmat", ""], "position": [x, y, z],
//                      "rotation": [pitch, yaw, roll] (degrees, applied yaw · pitch · roll), "scale": s | [x, y, z],
//                      "lightmapScale": 1, "static": true, "layer": 1 } ] }
// Models and materials are the cooked ones in `dataDir`, so maps cook after models, materials and textures. Static
// instances are lightmapped (one rect per mesh node, sized from the mesh's reference lightmap size, the map density,
// the instance scale and lightmapScale) and get a reflection probe when their center lies in a probe box; dynamic ones
// are probe-lit. Blended submeshes don't take part in the bake.
// Output: <dataDir>/<outDir>/ map.json, lightmaps.ktx2, lightmaps_rgbm.ktx2, shadowmask.ktx2, probes.fprobe,
// sky.ktx2, sky_rgbm.ktx2, reflection<i>.ktx2, reflection<i>_rgbm.ktx2.
bool cookMap(const std::string& sourcePath, const std::string& outDir, const std::string& dataDir, std::string& error);

} // namespace tools
