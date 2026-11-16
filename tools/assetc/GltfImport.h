#pragma once

#include "assets/AnimFile.h"
#include "assets/Model.h"

#include <string>

namespace tools {

// glTF 2.0 (.gltf + buffers, or .glb) → engine model (ARCHITECTURE.md §7.2). The default scene's node hierarchy
// becomes the node table; each glTF mesh becomes one engine mesh whose triangle primitives are its submeshes.
// Each glTF material is written to <dataDir>/materials/<name>/<material>.fmat and its maps are cooked to
// <dataDir>/textures/<name>/; the model's material slots reference those data-relative paths. Materials (M2 plan D9):
// base color, normal (+ scale), emissive (+ factor, KHR_materials_emissive_strength) and alpha modes;
// KHR_materials_pbrSpecularGlossiness is taken as is, and metallic-roughness is converted to specular/gloss (baked into
// new albedo and spec/gloss maps when metalness or roughness varies per texel). Occlusion maps, extra UV sets and
// KHR_texture_transform are ignored (noted in the tool output). Skins (M6 plan D4): JOINTS_0/WEIGHTS_0 and inverse
// binds, split into parts of at most assets::kMaxSkinJoints joints; every node gets a unique name so clips can bind to
// it. With `anims`, every animation's node channels (translation, rotation, scale; STEP, LINEAR, CUBICSPLINE) are
// resampled at assets::kAnimSampleRate into a clip library. Morph targets, cameras and lights are ignored.
bool importGltf(const std::string& gltfPath, const std::string& name, const std::string& dataDir, assets::ModelData& out,
                std::string& error, bool generateLods = true, assets::AnimData* anims = nullptr);

// Writes a file atomically (temp file + rename), creating directories as needed.
bool writeFileAtomic(const std::string& path, const void* data, size_t size, std::string& error);

} // namespace tools
