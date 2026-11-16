#include "MapCook.h"

#include "Baker.h"
#include "GltfImport.h"
#include "HdrCodec.h"
#include "ImageCook.h"
#include "LightmapUv.h"
#include "Sky.h"
#include "assets/MapFormat.h"
#include "assets/Material.h"
#include "assets/Model.h"
#include "assets/Texture.h"

#include <yyjson.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <numeric>

namespace tools {
namespace fs = std::filesystem;
namespace {

constexpr float kDegrees = 0.01745329252f;

bool readBytes(const std::string& path, std::vector<std::byte>& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    file.seekg(0, std::ios::end);
    out.resize(size_t(file.tellg()));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(out.data()), std::streamsize(out.size()));
    return bool(file);
}

// Strict JSON access with error context: unknown keys and wrong types fail the cook.
struct Json {
    std::string& error;

    bool fail(const std::string& message) {
        error = message;
        return false;
    }
    bool keys(yyjson_val* obj, std::initializer_list<const char*> allowed, const char* where) {
        if (!obj) return true;
        if (!yyjson_is_obj(obj)) return fail(std::string(where) + " must be an object");
        yyjson_obj_iter it = yyjson_obj_iter_with(obj);
        while (yyjson_val* key = yyjson_obj_iter_next(&it)) {
            const char* k = yyjson_get_str(key);
            bool known = false;
            for (const char* a : allowed) known = known || std::strcmp(k, a) == 0;
            if (!known) return fail(std::string("unknown key \"") + k + "\" in " + where);
        }
        return true;
    }
    bool number(yyjson_val* obj, const char* key, float& out, double lo, double hi) {
        yyjson_val* v = obj ? yyjson_obj_get(obj, key) : nullptr;
        if (!v) return true;
        if (!yyjson_is_num(v) || !std::isfinite(yyjson_get_num(v)) || yyjson_get_num(v) < lo || yyjson_get_num(v) > hi)
            return fail(std::string("\"") + key + "\" must be a number from " + std::to_string(lo) + " to " + std::to_string(hi));
        out = float(yyjson_get_num(v));
        return true;
    }
    bool count(yyjson_val* obj, const char* key, uint32_t& out, uint32_t lo, uint32_t hi) {
        float f = float(out);
        if (!number(obj, key, f, lo, hi)) return false;
        out = uint32_t(f);
        return true;
    }
    bool vec3(yyjson_val* obj, const char* key, glm::vec3& out, bool required = false) {
        yyjson_val* v = obj ? yyjson_obj_get(obj, key) : nullptr;
        if (!v) return !required || fail(std::string("missing \"") + key + "\"");
        if (yyjson_is_num(v)) { // a single number means all three components
            out = glm::vec3(float(yyjson_get_num(v)));
            return std::isfinite(out.x) || fail(std::string("\"") + key + "\" must be finite");
        }
        if (!yyjson_is_arr(v) || yyjson_arr_size(v) != 3) return fail(std::string("\"") + key + "\" must be an array of 3 numbers");
        for (size_t i = 0; i < 3; ++i) {
            yyjson_val* c = yyjson_arr_get(v, i);
            if (!yyjson_is_num(c) || !std::isfinite(yyjson_get_num(c))) return fail(std::string("\"") + key + "\" needs numbers");
            out[int(i)] = float(yyjson_get_num(c));
        }
        return true;
    }
    bool color(yyjson_val* obj, const char* key, glm::vec3& out) {
        if (!vec3(obj, key, out)) return false;
        return (out.x >= 0.0f && out.y >= 0.0f && out.z >= 0.0f) || fail(std::string("\"") + key + "\" must not be negative");
    }
};

struct SourceInstance {
    std::string model;
    std::vector<std::string> materials;
    glm::mat4 transform{1.0f};
    float lightmapScale = 1.0f;
    bool isStatic = true;
    bool occluder = false; // M5 plan D4: its boxes hide what is behind them (static, solid box-like meshes only)
    uint8_t layer = 1;
};

struct SourceProbe {
    glm::vec3 position{0.0f}, boxMin{0.0f}, boxMax{0.0f};
    uint32_t faceSize = 128;
};

struct MapSource {
    float texelsPerMeter = 4.0f;
    uint32_t pageSize = 1024, maxPages = 4;
    BakeSettings bake;
    glm::vec3 sunDirection{0.4f, 0.8f, 0.4f}, sunColor{1.0f};
    float sunIntensity = 2.0f;
    SkySettings sky;
    uint32_t skyFaceSize = 512;
    float sunDiscIntensity = 30.0f, sunDiscRadiusDeg = 0.6f;
    float exposure = 1.0f; // optional (M8 plan D2; R8)
    assets::FogSettings fog;
    bool hasProbes = false;
    glm::vec3 probeMin{0.0f}, probeMax{0.0f}, probeSpacing{4.0f};
    std::vector<SourceProbe> reflectionProbes;
    std::vector<BakeLight> lights;
    std::vector<SourceInstance> instances;
};

bool parseSource(const std::vector<std::byte>& bytes, MapSource& src, std::string& error) {
    yyjson_doc* doc = yyjson_read(reinterpret_cast<const char*>(bytes.data()), bytes.size(), 0);
    if (!doc) {
        error = "not valid JSON";
        return false;
    }
    struct Guard {
        yyjson_doc* d;
        ~Guard() { yyjson_doc_free(d); }
    } guard{doc};
    yyjson_val* root = yyjson_doc_get_root(doc);
    Json j{error};
    if (!j.keys(root, {"lightmap", "bake", "sun", "sky", "fog", "probes", "reflectionProbes", "lights", "instances", "exposure"}, "the map"))
        return false;
    if (!j.number(root, "exposure", src.exposure, 0.01, 100.0)) return false; // optional: the tone curve's input scale (R8)

    yyjson_val* lm = yyjson_obj_get(root, "lightmap");
    if (!j.keys(lm, {"texelsPerMeter", "pageSize", "maxPages"}, "lightmap") || !j.number(lm, "texelsPerMeter", src.texelsPerMeter, 0.25, 64.0) ||
        !j.count(lm, "pageSize", src.pageSize, 64, 4096) || !j.count(lm, "maxPages", src.maxPages, 1, 64))
        return false;
    if (src.pageSize % 4) return j.fail("lightmap.pageSize must be a multiple of 4 (BC6H blocks)");

    yyjson_val* bake = yyjson_obj_get(root, "bake");
    if (!j.keys(bake, {"samples", "bounces", "sunSamples", "lightSamples", "probeRays"}, "bake") ||
        !j.count(bake, "samples", src.bake.samples, 1, 65536) || !j.count(bake, "bounces", src.bake.bounces, 0, 8) ||
        !j.count(bake, "sunSamples", src.bake.sunSamples, 1, 4096) || !j.count(bake, "lightSamples", src.bake.lightSamples, 1, 4096) ||
        !j.count(bake, "probeRays", src.bake.probeRays, 16, 1u << 20))
        return false;

    yyjson_val* sun = yyjson_obj_get(root, "sun");
    float sunRadius = 0.7f;
    if (!j.keys(sun, {"direction", "color", "intensity", "angularRadiusDeg"}, "sun") || !j.vec3(sun, "direction", src.sunDirection, true) ||
        !j.color(sun, "color", src.sunColor) || !j.number(sun, "intensity", src.sunIntensity, 0.0, 1000.0) ||
        !j.number(sun, "angularRadiusDeg", sunRadius, 0.0, 10.0))
        return false;
    if (glm::length(src.sunDirection) < 1e-6f) return j.fail("sun.direction is zero");
    src.sunDirection = glm::normalize(src.sunDirection);
    src.bake.sunAngularRadius = sunRadius * kDegrees;

    yyjson_val* sky = yyjson_obj_get(root, "sky");
    float seed = float(src.sky.seed);
    if (!j.keys(sky, {"zenith", "horizon", "ground", "glow", "cloudColor", "intensity", "glowExponent", "clouds", "cloudScale", "seed", "faceSize",
                      "sunDiscIntensity", "sunDiscRadiusDeg"},
                "sky") ||
        !j.color(sky, "zenith", src.sky.zenith) || !j.color(sky, "horizon", src.sky.horizon) || !j.color(sky, "ground", src.sky.ground) ||
        !j.color(sky, "glow", src.sky.glow) || !j.color(sky, "cloudColor", src.sky.cloudColor) ||
        !j.number(sky, "intensity", src.sky.intensity, 0.0, 1e4) || !j.number(sky, "glowExponent", src.sky.glowExponent, 1.0, 4096.0) ||
        !j.number(sky, "clouds", src.sky.clouds, 0.0, 1.0) || !j.number(sky, "cloudScale", src.sky.cloudScale, 0.01, 100.0) ||
        !j.number(sky, "seed", seed, 0.0, 1e9) || !j.count(sky, "faceSize", src.skyFaceSize, 16, 2048) ||
        !j.number(sky, "sunDiscIntensity", src.sunDiscIntensity, 0.0, 1e6) || !j.number(sky, "sunDiscRadiusDeg", src.sunDiscRadiusDeg, 0.0, 10.0))
        return false;
    src.sky.seed = uint32_t(seed);
    src.sky.sunDirection = src.sunDirection;

    yyjson_val* fog = yyjson_obj_get(root, "fog");
    assets::FogSettings& f = src.fog;
    if (!j.keys(fog, {"mode", "color", "start", "end", "density", "maxOpacity", "height", "heightBase", "heightDensity", "heightFalloff", "skyHorizon"},
                "fog"))
        return false;
    if (fog) {
        if (yyjson_val* mode = yyjson_obj_get(fog, "mode"); mode && !assets::parseFogMode(yyjson_get_str(mode), f.mode))
            return j.fail("fog.mode must be off, linear, exp or exp2");
        if (!j.color(fog, "color", f.color) || !j.number(fog, "start", f.start, 0.0, 1e6) || !j.number(fog, "end", f.end, 0.0, 1e6) ||
            !j.number(fog, "density", f.density, 0.0, 10.0) || !j.number(fog, "maxOpacity", f.maxOpacity, 0.0, 1.0) ||
            !j.number(fog, "heightBase", f.heightBase, -1e5, 1e5) || !j.number(fog, "heightDensity", f.heightDensity, 0.0, 10.0) ||
            !j.number(fog, "heightFalloff", f.heightFalloff, 0.0, 100.0) || !j.number(fog, "skyHorizon", f.skyHorizon, 0.0, 1.0))
            return false;
        if (yyjson_val* h = yyjson_obj_get(fog, "height")) {
            if (!yyjson_is_bool(h)) return j.fail("fog.height must be true or false");
            f.height = yyjson_get_bool(h);
        }
        if (f.mode == assets::FogSettings::Mode::Linear && f.end <= f.start) return j.fail("fog.end must be beyond fog.start");
    }

    if (yyjson_val* probes = yyjson_obj_get(root, "probes")) {
        if (!j.keys(probes, {"min", "max", "spacing"}, "probes") || !j.vec3(probes, "min", src.probeMin, true) ||
            !j.vec3(probes, "max", src.probeMax, true) || !j.vec3(probes, "spacing", src.probeSpacing, true))
            return false;
        for (int a = 0; a < 3; ++a)
            if (!(src.probeSpacing[a] > 0.05f) || src.probeMax[a] < src.probeMin[a]) return j.fail("probes: bad grid");
        src.hasProbes = true;
    }

    size_t i = 0, n = 0;
    yyjson_val* item = nullptr;
    yyjson_arr_foreach(yyjson_obj_get(root, "reflectionProbes"), i, n, item) {
        SourceProbe p;
        if (!j.keys(item, {"position", "boxMin", "boxMax", "faceSize"}, "reflectionProbes[]") || !j.vec3(item, "position", p.position, true) ||
            !j.vec3(item, "boxMin", p.boxMin, true) || !j.vec3(item, "boxMax", p.boxMax, true) || !j.count(item, "faceSize", p.faceSize, 8, 1024))
            return false;
        if (!(p.boxMin.x < p.boxMax.x && p.boxMin.y < p.boxMax.y && p.boxMin.z < p.boxMax.z)) return j.fail("reflectionProbes[]: empty box");
        src.reflectionProbes.push_back(p);
    }
    if (src.reflectionProbes.size() > assets::kMaxReflectionProbes) return j.fail("at most 7 reflection probes per map");

    yyjson_arr_foreach(yyjson_obj_get(root, "lights"), i, n, item) {
        if (!j.keys(item, {"type", "position", "direction", "color", "intensity", "range", "innerConeDeg", "outerConeDeg", "radius"}, "lights[]"))
            return false;
        BakeLight l;
        const char* type = yyjson_get_str(yyjson_obj_get(item, "type"));
        if (!type || (std::strcmp(type, "point") != 0 && std::strcmp(type, "spot") != 0)) return j.fail("lights[].type must be point or spot");
        l.spot = std::strcmp(type, "spot") == 0;
        float intensity = 1.0f, inner = 20.0f, outer = 30.0f;
        if (!j.vec3(item, "position", l.position, true) || !j.vec3(item, "direction", l.direction) || !j.color(item, "color", l.color) ||
            !j.number(item, "intensity", intensity, 0.0, 1e5) || !j.number(item, "range", l.range, 0.01, 1e4) ||
            !j.number(item, "innerConeDeg", inner, 0.0, 89.0) || !j.number(item, "outerConeDeg", outer, 1.0, 89.0) ||
            !j.number(item, "radius", l.radius, 0.0, 10.0))
            return false;
        l.color *= intensity;
        if (l.spot) {
            if (glm::length(l.direction) < 1e-6f) return j.fail("lights[]: a spot needs a direction");
            l.direction = glm::normalize(l.direction);
            l.cosOuter = std::cos(outer * kDegrees);
            l.cosInner = std::max(std::cos(std::min(inner, outer) * kDegrees), l.cosOuter + 1e-4f); // same as the runtime
        }
        src.lights.push_back(l);
    }

    yyjson_arr_foreach(yyjson_obj_get(root, "instances"), i, n, item) {
        if (!j.keys(item, {"model", "materials", "position", "rotation", "scale", "lightmapScale", "static", "layer", "occluder"}, "instances[]"))
            return false;
        SourceInstance inst;
        const char* model = yyjson_get_str(yyjson_obj_get(item, "model"));
        if (!model || !*model) return j.fail("instances[].model must be a path");
        inst.model = model;
        size_t mi = 0, mn = 0;
        yyjson_val* m = nullptr;
        yyjson_arr_foreach(yyjson_obj_get(item, "materials"), mi, mn, m) {
            if (!yyjson_is_str(m)) return j.fail("instances[].materials must be paths");
            inst.materials.emplace_back(yyjson_get_str(m));
        }
        glm::vec3 position(0.0f), rotation(0.0f), scale(1.0f);
        float layer = 1.0f;
        if (!j.vec3(item, "position", position) || !j.vec3(item, "rotation", rotation) || !j.vec3(item, "scale", scale) ||
            !j.number(item, "lightmapScale", inst.lightmapScale, 0.05, 16.0) || !j.number(item, "layer", layer, 0.0, 15.0))
            return false;
        if (std::fabs(scale.x * scale.y * scale.z) < 1e-9f) return j.fail("instances[].scale must not be zero");
        if (yyjson_val* s = yyjson_obj_get(item, "static")) {
            if (!yyjson_is_bool(s)) return j.fail("instances[].static must be true or false");
            inst.isStatic = yyjson_get_bool(s);
        }
        if (yyjson_val* o = yyjson_obj_get(item, "occluder")) {
            if (!yyjson_is_bool(o)) return j.fail("instances[].occluder must be true or false");
            inst.occluder = yyjson_get_bool(o);
            if (inst.occluder && !inst.isStatic) return j.fail("instances[].occluder needs a static instance");
        }
        inst.layer = uint8_t(layer);
        inst.transform = glm::translate(glm::mat4(1.0f), position) * glm::rotate(glm::mat4(1.0f), rotation.y * kDegrees, glm::vec3(0, 1, 0)) *
                         glm::rotate(glm::mat4(1.0f), rotation.x * kDegrees, glm::vec3(1, 0, 0)) *
                         glm::rotate(glm::mat4(1.0f), rotation.z * kDegrees, glm::vec3(0, 0, 1)) * glm::scale(glm::mat4(1.0f), scale);
        src.instances.push_back(std::move(inst));
    }
    if (src.instances.empty()) return j.fail("the map has no instances");
    return true;
}

// A cooked model decoded back to floats.
struct LoadedMesh {
    std::vector<glm::vec3> positions, normals;
    std::vector<glm::vec2> uv0, uv1;
    std::vector<uint32_t> indices;
    std::vector<assets::ModelSubmesh> submeshes;
    uint16_t lightmapWidth = 0, lightmapHeight = 0;
    glm::vec3 boundsMin{0.0f}, boundsMax{0.0f};
};
struct LoadedModel {
    std::vector<std::pair<glm::mat4, uint32_t>> nodes; // node → model space, mesh index (mesh nodes only, in node order)
    std::vector<LoadedMesh> meshes;
    std::vector<std::string> materialPaths;
};

glm::mat4 nodeLocal(const assets::ModelNode& n) {
    const glm::quat q(n.rotation[3], n.rotation[0], n.rotation[1], n.rotation[2]);
    return glm::translate(glm::mat4(1.0f), glm::vec3(n.translation[0], n.translation[1], n.translation[2])) * glm::mat4_cast(q) *
           glm::scale(glm::mat4(1.0f), glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
}

bool loadModel(const std::string& dataDir, const std::string& path, LoadedModel& out, std::string& error) {
    std::vector<std::byte> bytes;
    if (!readBytes((fs::path(dataDir) / path).string(), bytes)) {
        error = "cannot read " + path + " (is it cooked by an earlier manifest entry?)";
        return false;
    }
    assets::ModelView view;
    if (!assets::openModel(bytes, view, &error)) {
        error = path + ": " + error;
        return false;
    }
    std::vector<glm::mat4> world(view.nodes.size());
    for (size_t i = 0; i < view.nodes.size(); ++i) {
        const assets::ModelNode& n = view.nodes[i];
        world[i] = n.parent >= 0 ? world[size_t(n.parent)] * nodeLocal(n) : nodeLocal(n);
        if (n.mesh >= 0) out.nodes.push_back({world[i], uint32_t(n.mesh)});
    }
    for (uint32_t m = 0; m < view.meshes.size(); ++m) {
        assets::DecodedMesh decoded;
        if (!assets::decodeMesh(view, m, decoded, &error)) {
            error = path + ": " + error;
            return false;
        }
        const assets::ModelMesh& info = view.meshes[m];
        LoadedMesh mesh;
        const size_t vertices = info.vertexCount;
        const auto* pos = reinterpret_cast<const assets::PositionVertex*>(decoded.positions.data());
        const auto* surf = reinterpret_cast<const assets::SurfaceVertex*>(decoded.surface.data());
        for (size_t v = 0; v < vertices; ++v) {
            mesh.positions.push_back(glm::vec3(pos[v].x, pos[v].y, pos[v].z) * info.dequantStep +
                                     glm::vec3(info.dequantOffset[0], info.dequantOffset[1], info.dequantOffset[2]));
            mesh.normals.push_back(glm::vec3(surf[v].normal[0], surf[v].normal[1], surf[v].normal[2]) / 127.0f);
            mesh.uv0.push_back(glm::vec2(surf[v].uv0[0], surf[v].uv0[1]));
            mesh.uv1.push_back(glm::vec2(surf[v].uv1[0], surf[v].uv1[1]) / 65535.0f);
        }
        mesh.indices.resize(info.indexCount);
        for (size_t k = 0; k < info.indexCount; ++k) {
            uint32_t index = 0;
            if (info.indexBytes == 2) {
                uint16_t i16;
                std::memcpy(&i16, decoded.indices.data() + k * 2, 2);
                index = i16;
            } else {
                std::memcpy(&index, decoded.indices.data() + k * 4, 4);
            }
            mesh.indices[k] = index;
        }
        for (uint32_t s = 0; s < info.submeshCount; ++s) mesh.submeshes.push_back(view.submeshes[info.firstSubmesh + s]);
        mesh.lightmapWidth = info.lightmapWidth;
        mesh.lightmapHeight = info.lightmapHeight;
        mesh.boundsMin = glm::vec3(info.boundsMin[0], info.boundsMin[1], info.boundsMin[2]);
        mesh.boundsMax = glm::vec3(info.boundsMax[0], info.boundsMax[1], info.boundsMax[2]);
        out.meshes.push_back(std::move(mesh));
    }
    for (const assets::ModelMaterialSlot& slot : view.materials) out.materialPaths.emplace_back(slot.path);
    return true;
}

// The first mip at most 256 px across, decoded to linear RGBA floats.
bool loadBakeTexture(const std::string& dataDir, const std::string& path, BakeTexture& out, std::string& error) {
    std::vector<std::byte> bytes;
    if (!readBytes((fs::path(dataDir) / path).string(), bytes)) {
        error = "cannot read " + path;
        return false;
    }
    assets::DecodedTexture tex;
    if (!assets::decodeKtx2(bytes, assets::TranscodeTargets{false, false}, tex, &error)) {
        error = path + ": " + error;
        return false;
    }
    if (tex.format != assets::PixelFormat::RGBA8Srgb && tex.format != assets::PixelFormat::RGBA8Unorm) {
        error = path + ": unexpected decoded format";
        return false;
    }
    size_t level = 0;
    while (level + 1 < tex.levels.size() && std::max(tex.levels[level].width, tex.levels[level].height) > 256) ++level;
    const assets::DecodedTexture::Level& l = tex.levels[level];
    const bool srgb = tex.format == assets::PixelFormat::RGBA8Srgb;
    out.width = l.width;
    out.height = l.height;
    out.texels.resize(size_t(l.width) * l.height);
    const auto* p = reinterpret_cast<const uint8_t*>(tex.data.data() + l.offset);
    for (size_t i = 0; i < out.texels.size(); ++i) {
        glm::vec4 c(p[i * 4], p[i * 4 + 1], p[i * 4 + 2], p[i * 4 + 3]);
        c /= 255.0f;
        if (srgb)
            for (int ch = 0; ch < 3; ++ch) c[ch] = srgbToLinear(p[i * 4 + size_t(ch)]);
        out.texels[i] = c;
    }
    return true;
}

// Shelf packing of instance rects into square pages, tallest first.
bool packRects(std::vector<LightmapRect>& rects, uint32_t pageSize, uint32_t maxPages, uint32_t& pages, std::string& error) {
    std::vector<size_t> order(rects.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return rects[a].height > rects[b].height; });
    uint32_t page = 0, x = 0, y = 0, shelf = 0;
    for (size_t i : order) {
        LightmapRect& r = rects[i];
        if (x + r.width > pageSize) {
            y += shelf;
            x = 0;
            shelf = 0;
        }
        if (y + r.height > pageSize) {
            ++page;
            x = y = shelf = 0;
            if (page >= maxPages) {
                error = "the lightmap rects don't fit into " + std::to_string(maxPages) + " pages of " + std::to_string(pageSize) +
                        "² (raise lightmap.maxPages or lower texelsPerMeter)";
                return false;
            }
        }
        r.page = page;
        r.x = x;
        r.y = y;
        x += r.width;
        shelf = std::max(shelf, r.height);
    }
    pages = rects.empty() ? 1 : page + 1;
    return true;
}

bool writeCube(const CubeMips& mips, const std::string& dataDir, const std::string& bc6hPath, const std::string& rgbmPath, std::string& error) {
    std::vector<KtxImage> bc6h, rgbm;
    for (uint32_t m = 0; m < mips.size(); ++m)
        for (uint32_t f = 0; f < 6; ++f) {
            const HdrImage& image = mips[m][f];
            bc6h.push_back({m, f, image.width, image.height, encodeBc6hImage(image)});
            KtxImage r{m, f, image.width, image.height, std::vector<uint8_t>(image.pixels.size() * 4)};
            for (size_t p = 0; p < image.pixels.size(); ++p) encodeRgbm(image.pixels[p], &r.bytes[p * 4]);
            rgbm.push_back(std::move(r));
        }
    const uint32_t size = mips[0][0].width;
    return writeKtx2Images(kVkBc6hUfloat, KtxShape::Cube, size, size, uint32_t(mips.size()), 6, bc6h, KtxEncoding::Raw, false, true,
                           (fs::path(dataDir) / bc6hPath).string(), error) &&
           writeKtx2Images(kVkR8G8B8A8Unorm, KtxShape::Cube, size, size, uint32_t(mips.size()), 6, rgbm, KtxEncoding::Uastc, false, true,
                           (fs::path(dataDir) / rgbmPath).string(), error);
}

uint32_t mipCount(uint32_t size) {
    uint32_t mips = 1;
    while ((size >> mips) > 0) ++mips;
    return mips;
}

double elapsed(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

} // namespace

bool cookMap(const std::string& sourcePath, const std::string& outDir, const std::string& dataDir, std::string& error) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::byte> sourceBytes;
    if (!readBytes(sourcePath, sourceBytes)) {
        error = "cannot read " + sourcePath;
        return false;
    }
    MapSource src;
    if (!parseSource(sourceBytes, src, error)) return false;

    // Models, materials and their color maps.
    std::map<std::string, LoadedModel> models;
    for (const SourceInstance& inst : src.instances) {
        if (models.count(inst.model)) continue;
        LoadedModel model;
        if (!loadModel(dataDir, inst.model, model, error)) return false;
        models.emplace(inst.model, std::move(model));
    }
    const glm::vec3 sunColor = src.sunColor * src.sunIntensity;
    Baker baker(src.bake, src.sky, src.sunDirection, sunColor);
    std::map<std::string, int32_t> textureIds;
    std::map<std::string, std::pair<uint32_t, bool>> materialIds; // bake material id, blended
    auto texture = [&](const std::string& path, int32_t& id) {
        if (path.empty()) return true;
        if (auto it = textureIds.find(path); it != textureIds.end()) {
            id = it->second;
            return true;
        }
        BakeTexture t;
        if (!loadBakeTexture(dataDir, path, t, error)) return false;
        id = int32_t(baker.addTexture(std::move(t)));
        textureIds.emplace(path, id);
        return true;
    };
    auto material = [&](const std::string& path, uint32_t& id, bool& blended) {
        if (auto it = materialIds.find(path); it != materialIds.end()) {
            id = it->second.first;
            blended = it->second.second;
            return true;
        }
        std::vector<std::byte> bytes;
        assets::MaterialFile file;
        std::string parseError;
        if (!readBytes((fs::path(dataDir) / path).string(), bytes) || !assets::parseMaterial(bytes, file, &parseError)) {
            error = "material " + path + ": " + (bytes.empty() ? "cannot read it" : parseError);
            return false;
        }
        BakeMaterial m;
        m.baseColor = file.baseColor;
        m.emission = file.emissionColor * file.emissionIntensity;
        m.alphaTest = file.blend == assets::BlendMode::AlphaTest;
        m.alphaCutoff = file.alphaCutoff;
        m.doubleSided = file.doubleSided;
        m.castShadows = file.castShadows;
        if (!texture(file.albedo, m.albedo) || !texture(file.emission, m.emissionMap)) return false;
        blended = file.blend == assets::BlendMode::AlphaBlend || file.blend == assets::BlendMode::Additive ||
                  file.blend == assets::BlendMode::Premultiplied;
        id = baker.addMaterial(m);
        materialIds.emplace(path, std::make_pair(id, blended));
        return true;
    };

    // Static instances: one lightmap rect per mesh node with opaque or alpha-tested submeshes.
    struct Placement {
        size_t instance;
        uint32_t node;
        int32_t rect = -1;
    };
    std::vector<LightmapRect> rects;
    std::vector<Placement> placements;
    std::vector<BakeMesh> meshes;
    assets::MapManifest manifest;
    for (size_t ii = 0; ii < src.instances.size(); ++ii) {
        const SourceInstance& inst = src.instances[ii];
        const LoadedModel& model = models.at(inst.model);
        assets::MapInstance out;
        out.model = inst.model;
        out.materials = inst.materials;
        out.transform = inst.transform;
        out.isStatic = inst.isStatic;
        out.occluder = inst.occluder;
        out.layer = inst.layer;
        if (inst.materials.size() > model.materialPaths.size()) {
            error = inst.model + ": more material overrides than material slots";
            return false;
        }
        glm::vec3 lo(INFINITY), hi(-INFINITY);
        for (uint32_t ni = 0; ni < model.nodes.size(); ++ni) {
            const auto& [nodeMatrix, meshIndex] = model.nodes[ni];
            const LoadedMesh& mesh = model.meshes[meshIndex];
            const glm::mat4 world = inst.transform * nodeMatrix;
            for (int c = 0; c < 8; ++c) {
                const glm::vec3 corner((c & 1) ? mesh.boundsMax.x : mesh.boundsMin.x, (c & 2) ? mesh.boundsMax.y : mesh.boundsMin.y,
                                       (c & 4) ? mesh.boundsMax.z : mesh.boundsMin.z);
                const glm::vec3 w = glm::vec3(world * glm::vec4(corner, 1.0f));
                lo = glm::min(lo, w);
                hi = glm::max(hi, w);
            }
            if (!inst.isStatic) continue;
            BakeMesh bm;
            const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(world)));
            for (size_t v = 0; v < mesh.positions.size(); ++v) {
                bm.positions.push_back(glm::vec3(world * glm::vec4(mesh.positions[v], 1.0f)));
                const glm::vec3 n = normalMatrix * mesh.normals[v];
                bm.normals.push_back(glm::length(n) > 1e-8f ? glm::normalize(n) : glm::vec3(0.0f, 1.0f, 0.0f));
            }
            bm.uv0 = mesh.uv0;
            bm.uv1 = mesh.uv1;
            const bool mirrored = glm::determinant(glm::mat3(world)) < 0.0f;
            bool anyBaked = false;
            for (const assets::ModelSubmesh& sub : mesh.submeshes) {
                const std::string& override = sub.materialSlot < inst.materials.size() ? inst.materials[sub.materialSlot] : std::string();
                const std::string& path = !override.empty() ? override : model.materialPaths[sub.materialSlot];
                uint32_t id = 0;
                bool blended = false;
                if (!material(path, id, blended)) return false;
                if (blended) continue; // glass and glows neither block nor bounce baked light
                anyBaked = true;
                const uint32_t first = sub.lods[0].firstIndex, count = sub.lods[0].indexCount;
                for (uint32_t k = first; k + 2 < first + count; k += 3) {
                    // A mirrored transform flips the winding; keep counter-clockwise fronts for the back-face test.
                    bm.indices.push_back(mesh.indices[k]);
                    bm.indices.push_back(mesh.indices[mirrored ? k + 2 : k + 1]);
                    bm.indices.push_back(mesh.indices[mirrored ? k + 1 : k + 2]);
                    bm.materials.push_back(id);
                }
            }
            Placement placement{ii, ni, -1};
            if (anyBaked && mesh.lightmapWidth > 0 && mesh.lightmapHeight > 0) {
                const float scale = std::cbrt(std::fabs(glm::determinant(glm::mat3(world))));
                const float factor = src.texelsPerMeter / kLightmapReferenceDensity * scale * inst.lightmapScale;
                LightmapRect r;
                r.width = std::min(src.pageSize, std::max<uint32_t>(mesh.lightmapWidth, uint32_t(std::lround(float(mesh.lightmapWidth) * factor))));
                r.height = std::min(src.pageSize, std::max<uint32_t>(mesh.lightmapHeight, uint32_t(std::lround(float(mesh.lightmapHeight) * factor))));
                placement.rect = int32_t(rects.size());
                rects.push_back(r);
            }
            bm.lightmapRect = placement.rect;
            if (!bm.indices.empty()) meshes.push_back(std::move(bm));
            placements.push_back(placement);
        }
        // Reflection probe: the smallest box holding the instance's center.
        if (inst.isStatic) {
            const glm::vec3 center = (lo + hi) * 0.5f;
            float best = INFINITY;
            for (size_t p = 0; p < src.reflectionProbes.size(); ++p) {
                const SourceProbe& probe = src.reflectionProbes[p];
                if (glm::all(glm::greaterThanEqual(center, probe.boxMin)) && glm::all(glm::lessThanEqual(center, probe.boxMax))) {
                    const glm::vec3 d = probe.boxMax - probe.boxMin;
                    if (d.x * d.y * d.z < best) {
                        best = d.x * d.y * d.z;
                        out.reflectionProbe = uint8_t(p + 1);
                    }
                }
            }
        }
        manifest.instances.push_back(std::move(out));
    }
    uint32_t pages = 1;
    if (!packRects(rects, src.pageSize, src.maxPages, pages, error)) return false;
    for (const Placement& p : placements) {
        assets::MapInstance& inst = manifest.instances[p.instance];
        if (inst.lightmaps.size() <= p.node) inst.lightmaps.resize(p.node + 1);
        if (p.rect < 0) continue;
        const LightmapRect& r = rects[size_t(p.rect)];
        const float s = float(src.pageSize);
        inst.lightmaps[p.node] = {int32_t(r.page), glm::vec4(float(r.width) / s, float(r.height) / s, float(r.x) / s, float(r.y) / s)};
    }
    for (BakeMesh& m : meshes) baker.addMesh(std::move(m));
    for (const BakeLight& l : src.lights) baker.addLight(l);
    baker.build();
    std::printf("  map: %zu instances, %u triangles, %zu lightmap rects in %u page(s) of %u², %zu baked lights\n", src.instances.size(),
                baker.triangleCount(), rects.size(), pages, src.pageSize, src.lights.size());

    // Lightmaps + shadowmask.
    auto t0 = std::chrono::steady_clock::now();
    baker.bakeLightmaps(rects, src.pageSize, pages);
    std::printf("  lightmaps: %u texels (%u inside geometry, rebuilt from neighbours) in %.1f s\n", baker.bakedTexels(), baker.invalidTexels(),
                elapsed(t0));
    const std::string base = outDir + "/";
    {
        std::vector<KtxImage> bc6h, rgbm, mask;
        for (uint32_t p = 0; p < pages; ++p) {
            const HdrImage& light = baker.lightmapPages()[p];
            bc6h.push_back({0, p, src.pageSize, src.pageSize, encodeBc6hImage(light)});
            KtxImage r{0, p, src.pageSize, src.pageSize, std::vector<uint8_t>(light.pixels.size() * 4)};
            for (size_t i = 0; i < light.pixels.size(); ++i) encodeRgbm(light.pixels[i], &r.bytes[i * 4]);
            rgbm.push_back(std::move(r));
            KtxImage m{0, p, src.pageSize, src.pageSize, std::vector<uint8_t>(light.pixels.size())};
            const std::vector<float>& shadow = baker.shadowmaskPages()[p];
            for (size_t i = 0; i < shadow.size(); ++i) m.bytes[i] = uint8_t(std::lround(std::clamp(shadow[i], 0.0f, 1.0f) * 255.0f));
            mask.push_back(std::move(m));
        }
        manifest.lightmaps = base + "lightmaps.ktx2";
        manifest.lightmapsRgbm = base + "lightmaps_rgbm.ktx2";
        manifest.shadowmask = base + "shadowmask.ktx2";
        manifest.lightmapPages = pages;
        manifest.lightmapSize = src.pageSize;
        const uint32_t s = src.pageSize;
        if (!writeKtx2Images(kVkBc6hUfloat, KtxShape::Array, s, s, 1, pages, bc6h, KtxEncoding::Raw, false, true,
                             (fs::path(dataDir) / manifest.lightmaps).string(), error) ||
            !writeKtx2Images(kVkR8G8B8A8Unorm, KtxShape::Array, s, s, 1, pages, rgbm, KtxEncoding::Raw, false, true,
                             (fs::path(dataDir) / manifest.lightmapsRgbm).string(), error) ||
            !writeKtx2Images(kVkR8Unorm, KtxShape::Array, s, s, 1, pages, mask, KtxEncoding::Uastc, false, true,
                             (fs::path(dataDir) / manifest.shadowmask).string(), error))
            return false;
    }

    // Sky.
    t0 = std::chrono::steady_clock::now();
    const CubeMips skyMips = prefilterCube(renderSkyCube(src.sky, src.skyFaceSize), mipCount(src.skyFaceSize));
    manifest.skyCube = base + "sky.ktx2";
    manifest.skyCubeRgbm = base + "sky_rgbm.ktx2";
    if (!writeCube(skyMips, dataDir, manifest.skyCube, manifest.skyCubeRgbm, error)) return false;
    std::printf("  sky: %u² × 6, %zu mips in %.1f s\n", src.skyFaceSize, skyMips.size(), elapsed(t0));

    // Probes.
    if (src.hasProbes) {
        t0 = std::chrono::steady_clock::now();
        assets::ProbeGridData grid;
        for (int a = 0; a < 3; ++a) grid.dims[a] = uint32_t(std::floor((src.probeMax[a] - src.probeMin[a]) / src.probeSpacing[a] + 1e-3f)) + 1;
        grid.origin = src.probeMin;
        grid.spacing = src.probeSpacing;
        if (uint64_t(grid.dims[0]) * grid.dims[1] * grid.dims[2] > (1u << 20)) {
            error = "the probe grid is too dense";
            return false;
        }
        baker.bakeProbes(grid);
        uint32_t invalid = 0;
        for (const auto& p : grid.probes) invalid += p.valid > 0.5f ? 0u : 1u;
        manifest.probes = base + "probes.fprobe";
        const std::vector<std::byte> bytes = assets::writeProbeGrid(grid);
        if (!writeFileAtomic((fs::path(dataDir) / manifest.probes).string(), bytes.data(), bytes.size(), error)) return false;
        std::printf("  probes: %u × %u × %u (%u inside geometry) in %.1f s\n", grid.dims[0], grid.dims[1], grid.dims[2], invalid, elapsed(t0));
    }

    // Reflection probes.
    for (size_t p = 0; p < src.reflectionProbes.size(); ++p) {
        t0 = std::chrono::steady_clock::now();
        const SourceProbe& probe = src.reflectionProbes[p];
        const CubeMips mips = prefilterCube(baker.renderCube(probe.position, probe.faceSize), mipCount(probe.faceSize));
        assets::MapReflectionProbe out;
        out.cube = base + "reflection" + std::to_string(p) + ".ktx2";
        out.cubeRgbm = base + "reflection" + std::to_string(p) + "_rgbm.ktx2";
        out.position = probe.position;
        out.boxMin = probe.boxMin;
        out.boxMax = probe.boxMax;
        if (!writeCube(mips, dataDir, out.cube, out.cubeRgbm, error)) return false;
        manifest.reflectionProbes.push_back(out);
        std::printf("  reflection probe %zu: %u² × 6 in %.1f s\n", p, probe.faceSize, elapsed(t0));
    }

    manifest.sunDirection = src.sunDirection;
    manifest.sunColor = sunColor;
    manifest.ambientSky = skyIrradiance(src.sky, glm::vec3(0.0f, 1.0f, 0.0f));
    manifest.ambientGround = skyIrradiance(src.sky, glm::vec3(0.0f, -1.0f, 0.0f));
    manifest.skyIntensity = 1.0f; // the cube holds the sky's radiance already
    manifest.sunDiscIntensity = src.sunDiscIntensity;
    manifest.sunDiscRadiusDeg = src.sunDiscRadiusDeg;
    manifest.exposure = src.exposure;
    manifest.fog = src.fog;
    const std::string text = assets::writeMapManifest(manifest);
    if (!writeFileAtomic((fs::path(dataDir) / base / "map.json").string(), text.data(), text.size(), error)) return false;
    std::printf("  map baked in %.1f s\n", elapsed(start));
    return true;
}

} // namespace tools
