#pragma once

#include "assets/Model.h"

#include <string>
#include <vector>

namespace tools {

struct CookOptions {
    std::string manifest;    // assets/source/cook.json
    std::string sourceDir;   // assets/source
    std::string externalDir; // assets/external (downloaded sample assets; may be missing)
    std::string outDir;      // <build>/bin/data
    std::string toolPath;    // this executable: a newer tool re-cooks everything
    bool force = false;
};

// Runs the cook manifest incrementally (M1 plan D2):
//   { "copy":     [ { "from": "materials", "to": "materials" } ],
//     "textures": [ { "input": "builtin:checker:...", "role": "albedo", "out": "textures/x.ktx2" },
//                   { "input": "builtin:fence", "role": "albedo_cutout", "cutoff": 0.5, "out": "textures/f.ktx2" } ],
//     "models":   [ { "input": "builtin:cube", "material": "materials/x.fmat", "out": "models/cube.fmodel" },
//                   { "input": "builtin:lamp", "materials": ["materials/pole.fmat", "materials/bulb.fmat"], ... },
//                   { "input": "external:Models/Avocado/glTF/Avocado.gltf", "out": "models/avocado.fmodel",
//                     "optional": true },
//                   { "input": "rig:tank_rig", "materials": [...], "out": "models/tank_rig.fmodel",
//                     "anims": "anims/tank_rig.fanim" } ] }
// Inputs are builtin specs, procedural rigs, paths relative to sourceDir, or "external:" paths relative to externalDir.
// "anims" also writes the source's clip library (M6 plan D3). An output is rebuilt when it (or its clip library) is
// missing or older than its inputs, the manifest or the tool. Missing optional inputs are skipped with a note. Returns
// the process exit code.
int runCook(const CookOptions& options);

// One model from a builtin spec ("builtin:..."), a procedural rig ("rig:<name>", Rigs.h) or a glTF file, written to
// outPath; with animsPath, its clips go to that .fanim (M6 plan D3, D4). A builtin mesh or rig needs one material path
// per material slot; glTF files bring their own materials. `generateLods` = false keeps one level (the manifest's
// "lods": false).
bool cookModel(const std::string& input, const std::vector<std::string>& materials, const std::string& outPath,
               const std::string& dataDir, std::string& error, bool generateLods = true, const std::string& animsPath = {});

} // namespace tools
