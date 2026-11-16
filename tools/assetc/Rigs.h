#pragma once

#include "MeshBuilder.h"
#include "assets/AnimFile.h"

#include <string>
#include <vector>

namespace tools {

// A model with a node hierarchy, rigid and skinned meshes and clips, before cooking (M6 plan D1, D5).
struct RigNode {
    std::string name;
    int32_t parent = -1; // parents first
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};
    int32_t mesh = -1; // into RigModel::meshes
};

struct RigModel {
    std::vector<RigNode> nodes;
    std::vector<RawMesh> meshes; // a skinned mesh names its joints by index into `nodes` and is in model (bind) space
    uint32_t materialSlots = 1;  // the cook manifest gives one material path per slot
    assets::AnimData clips;
};

// Procedural rigs (CC0 by construction), addressed as "rig:<name>" in the cook manifest:
//   tank_rig          a skinned tank facing +Z: hull, turret, gun, a smooth-skinned four-joint antenna and eight road
//                     wheels; slot 0 = body, slot 1 = running gear. Clips: idle, drive (loops), turret_scan (turret
//                     sweep with a hull tilt that a turret mask removes), fire (gun recoil and antenna whip, one-shot),
//                     hit (a jolt meant for an additive layer, one-shot).
//   tentacle          a smooth-skinned 12-joint tapered tube 3.8 m tall; clips wave, curl (loops), reach (one-shot).
//   radar_mast        rigid node animation: a mast, a head that turns and a nodding dish; clip spin.
//   gate              rigid node animation: two doors on hinges; clips open, close (one-shots), cycle (loop).
//   skin_check        four boxes on a joint chain, every vertex fully on one joint: one skinned mesh;
//   skin_check_rigid  the same boxes as rigid meshes on the same named nodes. Both play skin_check's clip bend, so the
//                     two must render alike (--skin-check, M6 plan D13).
bool makeBuiltinRig(const std::string& name, RigModel& out, std::string& error);

// RigModel → ModelData: skins split to at most assets::kMaxSkinJoints joints (extra parts become sibling nodes), every
// mesh built by buildMesh(), one material slot per given path.
bool assembleRig(RigModel rig, const std::vector<std::string>& materials, bool generateLods, assets::ModelData& out,
                 std::string& error);

} // namespace tools
