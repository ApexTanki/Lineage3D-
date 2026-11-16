#include "Rigs.h"

#include "Builtins.h"
#include "assets/NameHash.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>

namespace tools {
namespace {

constexpr float kPi = 3.14159265358979f;

float rad(float degrees) {
    return degrees * kPi / 180.0f;
}

float smooth(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

glm::quat axisAngle(glm::vec3 axis, float radians) {
    return glm::angleAxis(radians, axis);
}

glm::mat4 trs(glm::vec3 t, glm::quat r = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3 s = glm::vec3(1.0f)) {
    return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(r) * glm::scale(glm::mat4(1.0f), s);
}

// Parts gathered per material slot, then laid out as one submesh per slot.
struct Parts {
    std::vector<RawVertex> vertices;
    std::vector<std::vector<uint32_t>> slots;

    // Appends `part` transformed by `m`; joint ≥ 0 binds every vertex fully to that skin joint.
    void add(const RawMesh& part, const glm::mat4& m, int joint, uint32_t slot) {
        const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(m)));
        const auto base = uint32_t(vertices.size());
        for (RawVertex v : part.vertices) {
            v.position = glm::vec3(m * glm::vec4(v.position, 1.0f));
            v.normal = glm::normalize(normalMatrix * v.normal);
            if (joint >= 0) {
                v.joints = glm::u16vec4(uint16_t(joint), 0, 0, 0);
                v.weights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
            }
            vertices.push_back(v);
        }
        if (slots.size() <= slot) slots.resize(slot + 1);
        for (uint32_t i : part.indices) slots[slot].push_back(base + i);
    }

    RawMesh finish() const {
        RawMesh m;
        m.vertices = vertices;
        for (size_t s = 0; s < slots.size(); ++s) {
            if (slots[s].empty()) continue;
            m.submeshes.push_back({uint32_t(m.indices.size()), uint32_t(slots[s].size()), uint32_t(s)});
            m.indices.insert(m.indices.end(), slots[s].begin(), slots[s].end());
        }
        m.hasNormals = true;
        m.hasTangents = false;
        m.hasUv0 = true;
        m.hasUv1 = false;
        return m;
    }
};

// A tapered tube along +Y from `base`, smooth-skinned to a joint chain whose joints sit `spacing` apart from the base:
// a vertex between joints i and i + 1 blends them linearly (classic linear blend skinning), the tip follows the last.
void addChainTube(Parts& parts, glm::vec3 base, float height, float r0, float r1, uint32_t rings, uint32_t sides,
                  const std::vector<uint16_t>& chain, float spacing, glm::vec2 uvRepeat, uint32_t slot) {
    RawMesh tube;
    const float slope = (r0 - r1) / height;
    auto skin = [&](float y, RawVertex& v) {
        const float s = y / spacing;
        const auto last = uint32_t(chain.size() - 1);
        const uint32_t i = std::min(uint32_t(std::max(0.0f, std::floor(s))), last);
        const float f = std::clamp(s - float(i), 0.0f, 1.0f);
        if (i >= last || f <= 0.0f) {
            v.joints = glm::u16vec4(chain[i], 0, 0, 0);
            v.weights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
        } else {
            v.joints = glm::u16vec4(chain[i], chain[i + 1], 0, 0);
            v.weights = glm::vec4(1.0f - f, f, 0.0f, 0.0f);
        }
    };
    for (uint32_t k = 0; k <= rings; ++k) {
        const float t = float(k) / float(rings);
        const float y = height * t, r = r0 + (r1 - r0) * t;
        for (uint32_t j = 0; j <= sides; ++j) {
            const float phi = 2.0f * kPi * float(j) / float(sides);
            RawVertex v;
            v.position = base + glm::vec3(r * std::cos(phi), y, r * std::sin(phi));
            v.normal = glm::normalize(glm::vec3(std::cos(phi), slope, std::sin(phi)));
            v.uv0 = glm::vec2(float(j) / float(sides), 1.0f - t) * uvRepeat;
            skin(y, v);
            tube.vertices.push_back(v);
        }
    }
    for (uint32_t k = 0; k < rings; ++k)
        for (uint32_t j = 0; j < sides; ++j) {
            const uint32_t a = k * (sides + 1) + j, b = a + 1, c = a + sides + 1, d = c + 1;
            for (uint32_t i : {a, c, b, b, c, d}) tube.indices.push_back(i); // counter-clockwise seen from outside
        }
    const auto center = uint32_t(tube.vertices.size()); // tip cap
    RawVertex tip;
    tip.position = base + glm::vec3(0.0f, height, 0.0f);
    tip.normal = glm::vec3(0.0f, 1.0f, 0.0f);
    tip.uv0 = glm::vec2(0.5f, 0.0f);
    skin(height, tip);
    tube.vertices.push_back(tip);
    for (uint32_t j = 0; j < sides; ++j) {
        const uint32_t p0 = rings * (sides + 1) + j, p1 = p0 + 1;
        for (uint32_t i : {center, p1, p0}) tube.indices.push_back(i);
    }
    // The vertices already carry their skin; add() with joint −1 keeps it.
    parts.add(tube, glm::mat4(1.0f), -1, slot);
}

// Clip authoring: tracks sampled from functions of time at the cooker's rate. Loops use periodic functions whose period
// divides the duration, so the last frame equals the first.
struct Clip {
    assets::AnimData::Clip data;
    uint32_t frames = 1;

    Clip(const char* name, float duration) {
        data.name = name;
        data.duration = duration;
        data.sampleRate = assets::animEffectiveRate(duration); // keys span exactly [0, duration]
        frames = assets::animFrameCount(duration, data.sampleRate);
    }
    float time(uint32_t f) const { return std::min(float(f) / data.sampleRate, data.duration); }

    void translation(const char* node, const std::function<glm::vec3(float)>& fn) {
        assets::AnimData::Track t{assets::nameHash(node), assets::AnimChannel::Translation, {}};
        for (uint32_t f = 0; f < frames; ++f) {
            const glm::vec3 v = fn(time(f));
            t.keys.insert(t.keys.end(), {v.x, v.y, v.z});
        }
        data.tracks.push_back(std::move(t));
    }
    void rotation(const char* node, const std::function<glm::quat(float)>& fn) {
        assets::AnimData::Track t{assets::nameHash(node), assets::AnimChannel::Rotation, {}};
        glm::quat previous(1.0f, 0.0f, 0.0f, 0.0f);
        for (uint32_t f = 0; f < frames; ++f) {
            glm::quat q = glm::normalize(fn(time(f)));
            if (f > 0 && glm::dot(previous, q) < 0.0f) q = -q; // hemisphere-continuous keys
            previous = q;
            t.keys.insert(t.keys.end(), {q.x, q.y, q.z, q.w});
        }
        data.tracks.push_back(std::move(t));
    }
    void scale(const char* node, const std::function<glm::vec3(float)>& fn) {
        assets::AnimData::Track t{assets::nameHash(node), assets::AnimChannel::Scale, {}};
        for (uint32_t f = 0; f < frames; ++f) {
            const glm::vec3 v = fn(time(f));
            t.keys.insert(t.keys.end(), {v.x, v.y, v.z});
        }
        data.tracks.push_back(std::move(t));
    }
};

int addNode(RigModel& rig, const char* name, int parent, glm::vec3 translation, glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f)) {
    RigNode n;
    n.name = name;
    n.parent = parent;
    n.translation = translation;
    n.rotation = rotation;
    rig.nodes.push_back(n);
    return int(rig.nodes.size() - 1);
}

// Model-space bind matrices of every node (parents first).
std::vector<glm::mat4> bindPose(const RigModel& rig) {
    std::vector<glm::mat4> model(rig.nodes.size());
    for (size_t i = 0; i < rig.nodes.size(); ++i) {
        const RigNode& n = rig.nodes[i];
        const glm::mat4 local = trs(n.translation, n.rotation, n.scale);
        model[i] = n.parent >= 0 ? model[size_t(n.parent)] * local : local;
    }
    return model;
}

// The skin of `mesh`: joints are the given nodes, inverse binds from the bind pose.
void setSkin(RawMesh& mesh, const RigModel& rig, const std::vector<int>& jointNodes) {
    const std::vector<glm::mat4> bind = bindPose(rig);
    mesh.skinJoints.clear();
    mesh.inverseBinds.clear();
    for (int node : jointNodes) {
        mesh.skinJoints.push_back(uint32_t(node));
        mesh.inverseBinds.push_back(glm::inverse(bind[size_t(node)]));
    }
}

bool part(const std::string& spec, RawMesh& out, std::string& error) {
    out = RawMesh{};
    return makeBuiltinMesh(spec, out, error);
}

glm::quat alongX() { // the unit cylinder's axis (Y) turned onto X
    return axisAngle({0, 0, 1}, -kPi / 2);
}
glm::quat alongZ() { // ... onto Z
    return axisAngle({1, 0, 0}, kPi / 2);
}

bool makeTank(RigModel& rig, std::string& error) {
    rig.materialSlots = 2;
    const int root = addNode(rig, "root", -1, {0, 0, 0});
    const int hull = addNode(rig, "hull", root, {0, 1.0f, 0});
    const int turret = addNode(rig, "turret", hull, {0, 0.4f, -0.1f});
    const int gun = addNode(rig, "gun", turret, {0, 0.35f, 1.05f});
    int antenna[4];
    antenna[0] = addNode(rig, "antenna_0", turret, {-0.6f, 0.6f, -1.0f});
    for (int k = 1; k < 4; ++k) antenna[k] = addNode(rig, ("antenna_" + std::to_string(k)).c_str(), antenna[k - 1], {0, 0.4f, 0});
    const float wheelZ[4] = {-2.1f, -0.7f, 0.7f, 2.1f};
    int wheels[8];
    for (int i = 0; i < 4; ++i) wheels[i] = addNode(rig, ("wheel_l" + std::to_string(i)).c_str(), root, {-1.85f, 0.35f, wheelZ[i]});
    for (int i = 0; i < 4; ++i) wheels[4 + i] = addNode(rig, ("wheel_r" + std::to_string(i)).c_str(), root, {1.85f, 0.35f, wheelZ[i]});
    const int meshNode = addNode(rig, "tank_skin", root, {0, 0, 0});

    // Skin joints: root (tracks), hull, turret, gun, the antenna chain, the wheels.
    std::vector<int> joints = {root, hull, turret, gun, antenna[0], antenna[1], antenna[2], antenna[3]};
    joints.insert(joints.end(), wheels, wheels + 8);
    auto jointOf = [&](int node) { return int(std::find(joints.begin(), joints.end(), node) - joints.begin()); };

    Parts p;
    RawMesh m;
    if (!part("box:0.7:0.75:6:1", m, error)) return false; // tracks stay on the ground with the root
    for (float x : {-1.45f, 1.45f}) p.add(m, trs({x, 0.375f, 0}), jointOf(root), 1);
    if (!part("cylinder", m, error)) return false;
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 c = rig.nodes[size_t(wheels[i])].translation;
        p.add(m, trs(c, alongX(), {0.68f, 0.22f, 0.68f}), jointOf(wheels[i]), 1);
    }
    if (!part("box:2.6:0.8:5.4:1", m, error)) return false;
    p.add(m, trs({0, 1.0f, 0}), jointOf(hull), 0);
    if (!part("box:2.4:0.4:0.7:1", m, error)) return false; // front plate
    p.add(m, trs({0, 0.85f, 2.95f}, axisAngle({1, 0, 0}, rad(-25))), jointOf(hull), 0);
    if (!part("box:1.9:0.6:2.4:1", m, error)) return false;
    p.add(m, trs({0, 1.7f, -0.2f}), jointOf(turret), 0);
    if (!part("cylinder", m, error)) return false;
    p.add(m, trs({0.45f, 2.11f, -0.6f}, glm::quat(1, 0, 0, 0), {0.64f, 0.22f, 0.64f}), jointOf(turret), 0); // cupola
    if (!part("box:0.75:0.45:0.35:1", m, error)) return false;
    p.add(m, trs({0, 1.75f, 1.12f}), jointOf(gun), 0); // mantlet
    if (!part("cylinder", m, error)) return false;
    p.add(m, trs({0, 1.75f, 2.8f}, alongZ(), {0.17f, 3.0f, 0.17f}), jointOf(gun), 0); // barrel
    p.add(m, trs({0, 1.75f, 4.3f}, alongZ(), {0.26f, 0.3f, 0.26f}), jointOf(gun), 0); // muzzle brake
    addChainTube(p, {-0.6f, 2.0f, -1.1f}, 1.6f, 0.03f, 0.01f, 16, 6,
                 {uint16_t(jointOf(antenna[0])), uint16_t(jointOf(antenna[1])), uint16_t(jointOf(antenna[2])), uint16_t(jointOf(antenna[3]))},
                 0.4f, {1, 4}, 1);
    RawMesh mesh = p.finish();
    setSkin(mesh, rig, joints);
    rig.meshes.push_back(std::move(mesh));
    rig.nodes[size_t(meshNode)].mesh = 0;

    const glm::vec3 X(1, 0, 0), Y(0, 1, 0), Z(0, 0, 1);
    auto antennaName = [](int k) { return "antenna_" + std::to_string(k); };
    {
        Clip c("idle", 2.0f);
        c.translation("hull", [](float t) { return glm::vec3(0, 1.0f + 0.008f * std::sin(kPi * t), 0); });
        c.rotation("hull", [&](float t) { return axisAngle(X, rad(0.3f) * std::sin(kPi * t + 0.5f)); });
        for (int k = 0; k < 4; ++k)
            c.rotation(antennaName(k).c_str(), [&, k](float t) { return axisAngle(X, rad(1.5f + float(k)) * std::sin(kPi * t - 0.5f * float(k))); });
        rig.clips.clips.push_back(c.data);
    }
    {
        Clip c("drive", 1.0f);
        c.translation("hull", [](float t) { return glm::vec3(0, 1.0f + 0.02f * std::sin(4 * kPi * t), 0); });
        c.rotation("hull", [&](float t) {
            return axisAngle(X, rad(0.8f) * std::sin(2 * kPi * t)) * axisAngle(Z, rad(0.4f) * std::sin(2 * kPi * t + 1.0f));
        });
        for (int i = 0; i < 8; ++i) {
            const std::string name = rig.nodes[size_t(wheels[i])].name;
            c.rotation(name.c_str(), [&](float t) { return axisAngle(X, 2 * kPi * t); }); // one turn per second: 2.1 m/s
        }
        for (int k = 0; k < 4; ++k)
            c.rotation(antennaName(k).c_str(), [&, k](float t) {
                return axisAngle(X, -rad(4.0f + 2.0f * float(k)) + rad(1.5f) * std::sin(4 * kPi * t - 0.6f * float(k)));
            });
        rig.clips.clips.push_back(c.data);
    }
    {
        // A turret sweep with a hull tilt in the same clip: played through a turret mask, only the sweep remains.
        Clip c("turret_scan", 6.0f);
        c.rotation("turret", [&](float t) { return axisAngle(Y, rad(40) * std::sin(2 * kPi * t / 6)); });
        c.rotation("gun", [&](float t) { return axisAngle(X, rad(-4) * std::sin(2 * kPi * t / 3)); });
        c.rotation("hull", [&](float t) { return axisAngle(X, rad(3) * std::sin(2 * kPi * t / 6)); });
        rig.clips.clips.push_back(c.data);
    }
    {
        Clip c("fire", 0.5f);
        c.translation("gun", [](float t) {
            const float recoil = t < 0.05f ? t / 0.05f : std::pow(std::max(0.0f, 1.0f - (t - 0.05f) / 0.45f), 2.0f);
            return glm::vec3(0, 0.35f, 1.05f - 0.35f * recoil);
        });
        for (int k = 0; k < 4; ++k)
            c.rotation(antennaName(k).c_str(), [&, k](float t) {
                return axisAngle(X, rad(10.0f + 4.0f * float(k)) * std::exp(-6.0f * t) * std::sin(2 * kPi * 3 * t));
            });
        rig.clips.clips.push_back(c.data);
    }
    {
        // Starts and ends at the rest pose (sin(0) = 0): meant for an additive layer.
        Clip c("hit", 0.5f);
        c.translation("hull", [](float t) { return glm::vec3(0, 1.0f + 0.03f * std::exp(-6 * t) * std::sin(2 * kPi * 5 * t), 0); });
        c.rotation("hull", [&](float t) {
            const float decay = std::exp(-6 * t);
            return axisAngle(X, rad(4) * decay * std::sin(2 * kPi * 5 * t)) * axisAngle(Z, rad(2) * decay * std::sin(2 * kPi * 4 * t));
        });
        c.rotation("turret", [&](float t) { return axisAngle(Y, rad(2) * std::exp(-8 * t) * std::sin(2 * kPi * 4 * t)); });
        rig.clips.clips.push_back(c.data);
    }
    return true;
}

bool makeTentacle(RigModel& rig, std::string&) {
    rig.materialSlots = 1;
    const int root = addNode(rig, "tentacle_root", -1, {0, 0, 0});
    std::vector<int> chain;
    for (int i = 0; i < 12; ++i)
        chain.push_back(addNode(rig, ("tentacle_" + std::to_string(i)).c_str(), i == 0 ? root : chain.back(), {0, i == 0 ? 0.0f : 0.32f, 0}));
    const int meshNode = addNode(rig, "tentacle_skin", root, {0, 0, 0});
    Parts p;
    std::vector<uint16_t> local;
    for (int i = 0; i < 12; ++i) local.push_back(uint16_t(i));
    addChainTube(p, {0, 0, 0}, 3.8f, 0.32f, 0.04f, 48, 16, local, 0.32f, {2, 6}, 0);
    RawMesh mesh = p.finish();
    setSkin(mesh, rig, chain);
    rig.meshes.push_back(std::move(mesh));
    rig.nodes[size_t(meshNode)].mesh = 0;

    const glm::vec3 X(1, 0, 0), Z(0, 0, 1);
    auto name = [](int i) { return "tentacle_" + std::to_string(i); };
    {
        Clip c("wave", 3.0f);
        for (int i = 0; i < 12; ++i)
            c.rotation(name(i).c_str(), [&, i](float t) {
                return axisAngle(Z, rad(8) * std::sin(2 * kPi * t / 3 - 0.55f * float(i))) *
                       axisAngle(X, rad(4) * std::sin(4 * kPi * t / 3 - 0.4f * float(i)));
            });
        rig.clips.clips.push_back(c.data);
    }
    {
        Clip c("curl", 4.0f);
        for (int i = 1; i < 12; ++i)
            c.rotation(name(i).c_str(), [&, i](float t) { return axisAngle(X, rad(5.0f + 1.2f * float(i)) * (0.5f - 0.5f * std::cos(2 * kPi * t / 4))); });
        rig.clips.clips.push_back(c.data);
    }
    {
        Clip c("reach", 2.0f);
        for (int i = 0; i < 12; ++i)
            c.rotation(name(i).c_str(), [&, i](float t) { return axisAngle(X, -rad(i == 0 ? 10.0f : 4.0f) * smooth(t / 2)); });
        rig.clips.clips.push_back(c.data);
    }
    return true;
}

bool makeRadarMast(RigModel& rig, std::string& error) {
    rig.materialSlots = 2;
    const int mast = addNode(rig, "mast", -1, {0, 0, 0});
    const int head = addNode(rig, "head", mast, {0, 4.0f, 0});
    const int dish = addNode(rig, "dish", head, {0, 0.25f, 0.25f});
    RawMesh m;
    Parts mastParts, headParts, dishParts;
    if (!part("cylinder", m, error)) return false;
    mastParts.add(m, trs({0, 2.0f, 0}, glm::quat(1, 0, 0, 0), {0.24f, 4.0f, 0.24f}), -1, 0);
    dishParts.add(m, trs({0, 0.55f, 0.1f}, alongZ(), {1.8f, 0.08f, 1.8f}), -1, 1); // the dish disc faces +Z
    dishParts.add(m, trs({0, 0.55f, 0.45f}, alongZ(), {0.06f, 0.6f, 0.06f}), -1, 1); // feed
    if (!part("box:0.8:0.3:0.8:1", m, error)) return false;
    mastParts.add(m, trs({0, 0.15f, 0}), -1, 0);
    if (!part("box:0.6:0.4:0.6:1", m, error)) return false;
    headParts.add(m, trs({0, 0.2f, 0}), -1, 0);
    rig.meshes = {mastParts.finish(), headParts.finish(), dishParts.finish()};
    rig.nodes[size_t(mast)].mesh = 0;
    rig.nodes[size_t(head)].mesh = 1;
    rig.nodes[size_t(dish)].mesh = 2;
    Clip c("spin", 4.0f);
    c.rotation("head", [](float t) { return axisAngle({0, 1, 0}, 2 * kPi * t / 4); });
    c.rotation("dish", [](float t) { return axisAngle({1, 0, 0}, rad(-15) + rad(10) * std::sin(2 * kPi * t / 2)); });
    rig.clips.clips.push_back(c.data);
    return true;
}

bool makeGate(RigModel& rig, std::string& error) {
    rig.materialSlots = 2;
    const int gate = addNode(rig, "gate", -1, {0, 0, 0});
    const int left = addNode(rig, "door_l", gate, {-2.0f, 0, 0});
    const int right = addNode(rig, "door_r", gate, {2.0f, 0, 0});
    RawMesh m;
    Parts frame, doorL, doorR;
    if (!part("box:0.35:3.2:0.35:1", m, error)) return false;
    for (float x : {-2.2f, 2.2f}) frame.add(m, trs({x, 1.6f, 0}), -1, 0);
    if (!part("box:4.75:0.35:0.35:1", m, error)) return false;
    frame.add(m, trs({0, 3.375f, 0}), -1, 0);
    if (!part("box:1.95:2.6:0.1:1", m, error)) return false;
    doorL.add(m, trs({1.0f, 1.4f, 0}), -1, 1);
    doorR.add(m, trs({-1.0f, 1.4f, 0}), -1, 1);
    rig.meshes = {frame.finish(), doorL.finish(), doorR.finish()};
    rig.nodes[size_t(gate)].mesh = 0;
    rig.nodes[size_t(left)].mesh = 1;
    rig.nodes[size_t(right)].mesh = 2;
    const glm::vec3 Y(0, 1, 0);
    auto doors = [&](Clip& c, const std::function<float(float)>& open) { // open(t): 0 closed … 1 open
        c.rotation("door_l", [&, open](float t) { return axisAngle(Y, rad(100) * open(t)); });
        c.rotation("door_r", [&, open](float t) { return axisAngle(Y, -rad(100) * open(t)); });
    };
    {
        Clip c("open", 1.2f);
        doors(c, [](float t) { return smooth(t / 1.2f); });
        rig.clips.clips.push_back(c.data);
    }
    {
        Clip c("close", 1.2f);
        doors(c, [](float t) { return 1.0f - smooth(t / 1.2f); });
        rig.clips.clips.push_back(c.data);
    }
    {
        Clip c("cycle", 6.0f);
        doors(c, [](float t) { return t < 3.0f ? smooth(t / 1.2f) : 1.0f - smooth((t - 3.0f) / 1.2f); });
        rig.clips.clips.push_back(c.data);
    }
    return true;
}

bool makeSkinCheck(RigModel& rig, bool skinned, std::string& error) {
    rig.materialSlots = 1;
    std::vector<int> chain;
    for (int i = 0; i < 4; ++i)
        chain.push_back(addNode(rig, ("check_" + std::to_string(i)).c_str(), i == 0 ? -1 : chain.back(), {0, i == 0 ? 0.0f : 1.1f, 0}));
    const std::vector<glm::mat4> bind = bindPose(rig);
    const float width[4] = {0.8f, 0.7f, 0.6f, 0.5f};
    RawMesh box;
    if (!part("box:1:1:1:0.5", box, error)) return false;
    if (skinned) {
        Parts p;
        for (int i = 0; i < 4; ++i)
            p.add(box, bind[size_t(chain[size_t(i)])] * trs({0, 0.5f, 0}, glm::quat(1, 0, 0, 0), {width[i], 1.0f, width[i]}), i, 0);
        RawMesh mesh = p.finish();
        setSkin(mesh, rig, chain);
        rig.meshes.push_back(std::move(mesh));
        const int meshNode = addNode(rig, "check_skin", chain[0], {0, 0, 0});
        rig.nodes[size_t(meshNode)].mesh = 0;
    } else {
        for (int i = 0; i < 4; ++i) {
            Parts p;
            p.add(box, trs({0, 0.5f, 0}, glm::quat(1, 0, 0, 0), {width[i], 1.0f, width[i]}), -1, 0);
            rig.meshes.push_back(p.finish());
            rig.nodes[size_t(chain[size_t(i)])].mesh = i;
        }
    }
    // Rotation, translation and a non-uniform scale under rotated children (shear): everything skinning must reproduce.
    Clip c("bend", 2.0f);
    c.translation("check_0", [](float t) { return glm::vec3(0, 0.1f * std::sin(kPi * t), 0); });
    c.rotation("check_0", [](float t) { return axisAngle({0, 1, 0}, rad(20) * std::sin(kPi * t)); });
    for (int i = 1; i < 4; ++i)
        c.rotation(("check_" + std::to_string(i)).c_str(), [i](float t) {
            return axisAngle({0, 0, 1}, rad(28) * std::sin(kPi * t + 0.7f * float(i))) * axisAngle({0, 1, 0}, rad(15) * std::sin(kPi * t - 0.4f * float(i)));
        });
    c.scale("check_2", [](float t) { return glm::vec3(1.0f, 1.0f + 0.2f * std::sin(kPi * t), 1.0f); });
    rig.clips.clips.push_back(c.data);
    return true;
}

template <size_t N>
void copyName(char (&dst)[N], const std::string& src) {
    std::memset(dst, 0, N);
    std::memcpy(dst, src.data(), std::min(src.size(), N - 1));
}

} // namespace

bool makeBuiltinRig(const std::string& name, RigModel& out, std::string& error) {
    out = RigModel{};
    if (name == "tank_rig") return makeTank(out, error);
    if (name == "tentacle") return makeTentacle(out, error);
    if (name == "radar_mast") return makeRadarMast(out, error);
    if (name == "gate") return makeGate(out, error);
    if (name == "skin_check") return makeSkinCheck(out, true, error);
    if (name == "skin_check_rigid") return makeSkinCheck(out, false, error);
    error = "unknown rig '" + name + "'";
    return false;
}

bool assembleRig(RigModel rig, const std::vector<std::string>& materials, bool generateLods, assets::ModelData& out,
                 std::string& error) {
    if (materials.size() != rig.materialSlots) {
        error = "the rig has " + std::to_string(rig.materialSlots) + " material slot(s) but " + std::to_string(materials.size()) +
                " material path(s) were given";
        return false;
    }
    out = assets::ModelData{};
    for (const RigNode& n : rig.nodes) {
        assets::ModelNode node{};
        copyName(node.name, n.name);
        node.parent = n.parent;
        node.mesh = -1;
        node.translation[0] = n.translation.x, node.translation[1] = n.translation.y, node.translation[2] = n.translation.z;
        node.rotation[0] = n.rotation.x, node.rotation[1] = n.rotation.y, node.rotation[2] = n.rotation.z, node.rotation[3] = n.rotation.w;
        node.scale[0] = n.scale.x, node.scale[1] = n.scale.y, node.scale[2] = n.scale.z;
        out.nodes.push_back(node);
    }
    for (size_t ni = 0; ni < rig.nodes.size(); ++ni) {
        const int32_t meshIndex = rig.nodes[ni].mesh;
        if (meshIndex < 0) continue;
        RawMesh raw = rig.meshes[size_t(meshIndex)];
        raw.generateLods = generateLods;
        std::vector<RawMesh> pieces;
        if (!splitSkin(raw, pieces, error)) return false;
        for (size_t p = 0; p < pieces.size(); ++p) {
            assets::ModelData::Mesh mesh;
            if (!buildMesh(std::move(pieces[p]), mesh, out.submeshes, error)) {
                error = "node " + rig.nodes[ni].name + ": " + error;
                return false;
            }
            const auto index = int32_t(out.meshes.size());
            out.meshes.push_back(std::move(mesh));
            if (p == 0) {
                out.nodes[ni].mesh = index;
            } else {
                assets::ModelNode extra = out.nodes[ni];
                copyName(extra.name, rig.nodes[ni].name + "_part" + std::to_string(p));
                extra.mesh = index;
                out.nodes.push_back(extra);
            }
        }
    }
    for (uint32_t s = 0; s < rig.materialSlots; ++s) {
        assets::ModelMaterialSlot slot{};
        copyName(slot.name, "slot" + std::to_string(s));
        copyName(slot.path, materials[s]);
        out.materials.push_back(slot);
    }
    return true;
}

} // namespace tools
