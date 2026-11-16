#include "assets/MapFormat.h"

#include <yyjson.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace assets {
namespace {

constexpr const char* kFogModes[] = {"off", "linear", "exp", "exp2"};

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

struct Reader {
    std::string* error;
    std::string where;

    bool number(yyjson_val* obj, const char* key, float& out, double lo = -1e9, double hi = 1e9, bool required = false) {
        yyjson_val* v = yyjson_obj_get(obj, key);
        if (!v) return !required || fail(error, where + ": missing \"" + key + "\"");
        if (!yyjson_is_num(v)) return fail(error, where + ": \"" + key + "\" must be a number");
        const double d = yyjson_get_num(v);
        if (!std::isfinite(d) || d < lo || d > hi) return fail(error, where + ": \"" + key + "\" is out of range");
        out = float(d);
        return true;
    }
    bool floats(yyjson_val* v, const char* key, float* out, size_t count) {
        if (!yyjson_is_arr(v) || yyjson_arr_size(v) != count)
            return fail(error, where + ": \"" + key + "\" must be an array of " + std::to_string(count) + " numbers");
        for (size_t i = 0; i < count; ++i) {
            yyjson_val* c = yyjson_arr_get(v, i);
            if (!yyjson_is_num(c) || !std::isfinite(yyjson_get_num(c))) return fail(error, where + ": \"" + key + "\" needs numbers");
            out[i] = float(yyjson_get_num(c));
        }
        return true;
    }
    bool vec3(yyjson_val* obj, const char* key, glm::vec3& out, bool required = false) {
        yyjson_val* v = yyjson_obj_get(obj, key);
        if (!v) return !required || fail(error, where + ": missing \"" + key + "\"");
        return floats(v, key, &out.x, 3);
    }
    bool string(yyjson_val* obj, const char* key, std::string& out, bool required = false) {
        yyjson_val* v = yyjson_obj_get(obj, key);
        if (!v) return !required || fail(error, where + ": missing \"" + key + "\"");
        if (!yyjson_is_str(v)) return fail(error, where + ": \"" + key + "\" must be a string");
        out = yyjson_get_str(v);
        return true;
    }
};

std::string jsonString(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

std::string vec3String(const glm::vec3& v) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "[%.9g, %.9g, %.9g]", v.x, v.y, v.z);
    return buf;
}

template <typename T>
void put(std::vector<std::byte>& out, const T& value) {
    const auto* p = reinterpret_cast<const std::byte*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}

struct ProbeHeader {
    char magic[4];
    uint16_t version;
    uint16_t flags;
    uint32_t fileSize;
    uint32_t dims[3];
    float origin[3];
    float spacing[3];
    uint32_t probeCount;
    uint32_t pad[3];
};
static_assert(sizeof(ProbeHeader) == 64, "probe header is 64 bytes, so the records stay 16-byte aligned");

} // namespace

const char* fogModeName(FogSettings::Mode mode) {
    return kFogModes[uint32_t(mode)];
}

bool parseFogMode(const char* name, FogSettings::Mode& out) {
    for (uint32_t i = 0; name && i < 4; ++i)
        if (std::strcmp(name, kFogModes[i]) == 0) {
            out = FogSettings::Mode(i);
            return true;
        }
    return false;
}

bool parseMapManifest(std::span<const std::byte> json, MapManifest& out, std::string* error) {
    yyjson_doc* doc = yyjson_read(reinterpret_cast<const char*>(json.data()), json.size(), 0);
    if (!doc) return fail(error, "map.json is not valid JSON");
    struct Guard {
        yyjson_doc* d;
        ~Guard() { yyjson_doc_free(d); }
    } guard{doc};
    yyjson_val* root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) return fail(error, "map.json: the root must be an object");
    if (yyjson_get_int(yyjson_obj_get(root, "version")) != 1) return fail(error, "map.json: unsupported version");
    out = MapManifest{};
    Reader r{error, "map.json"};

    r.where = "map.json sun";
    yyjson_val* sun = yyjson_obj_get(root, "sun");
    if (!r.vec3(sun, "direction", out.sunDirection, true) || !r.vec3(sun, "color", out.sunColor, true)) return false;
    if (glm::length(out.sunDirection) < 1e-6f) return fail(error, "map.json: the sun direction is zero");
    out.sunDirection = glm::normalize(out.sunDirection);
    if (yyjson_val* exposure = yyjson_obj_get(root, "exposure")) { // optional since M8 (R8)
        if (!yyjson_is_num(exposure) || !(yyjson_get_num(exposure) >= 0.01 && yyjson_get_num(exposure) <= 100.0))
            return fail(error, "map.json: exposure must be a number from 0.01 to 100");
        out.exposure = float(yyjson_get_num(exposure));
    }
    r.where = "map.json ambient";
    yyjson_val* ambient = yyjson_obj_get(root, "ambient");
    if (!r.vec3(ambient, "sky", out.ambientSky, true) || !r.vec3(ambient, "ground", out.ambientGround, true)) return false;

    if (yyjson_val* sky = yyjson_obj_get(root, "sky")) {
        r.where = "map.json sky";
        if (!r.string(sky, "cube", out.skyCube, true) || !r.string(sky, "cubeRgbm", out.skyCubeRgbm, true) ||
            !r.number(sky, "intensity", out.skyIntensity, 0.0, 1e4) || !r.number(sky, "sunDiscIntensity", out.sunDiscIntensity, 0.0, 1e6) ||
            !r.number(sky, "sunDiscRadiusDeg", out.sunDiscRadiusDeg, 0.0, 10.0))
            return false;
    }
    if (yyjson_val* fog = yyjson_obj_get(root, "fog")) {
        r.where = "map.json fog";
        FogSettings& f = out.fog;
        if (yyjson_val* mode = yyjson_obj_get(fog, "mode"); mode && !parseFogMode(yyjson_get_str(mode), f.mode))
            return fail(error, "map.json fog: unknown mode");
        if (!r.vec3(fog, "color", f.color) || !r.number(fog, "start", f.start, 0.0, 1e6) || !r.number(fog, "end", f.end, 0.0, 1e6) ||
            !r.number(fog, "density", f.density, 0.0, 10.0) || !r.number(fog, "maxOpacity", f.maxOpacity, 0.0, 1.0) ||
            !r.number(fog, "heightBase", f.heightBase) || !r.number(fog, "heightDensity", f.heightDensity, 0.0, 10.0) ||
            !r.number(fog, "heightFalloff", f.heightFalloff, 0.0, 100.0) || !r.number(fog, "skyHorizon", f.skyHorizon, 0.0, 1.0))
            return false;
        if (yyjson_val* h = yyjson_obj_get(fog, "height")) f.height = yyjson_get_bool(h);
        if (f.mode == FogSettings::Mode::Linear && f.end <= f.start) return fail(error, "map.json fog: end must be beyond start");
    }
    if (yyjson_val* lm = yyjson_obj_get(root, "lightmaps")) {
        r.where = "map.json lightmaps";
        float pages = 0.0f, size = 0.0f;
        if (!r.string(lm, "bc6h", out.lightmaps, true) || !r.string(lm, "rgbm", out.lightmapsRgbm, true) ||
            !r.string(lm, "shadowmask", out.shadowmask, true) || !r.number(lm, "pages", pages, 1.0, 256.0, true) ||
            !r.number(lm, "size", size, 4.0, 8192.0, true))
            return false;
        out.lightmapPages = uint32_t(pages);
        out.lightmapSize = uint32_t(size);
    }
    r.where = "map.json";
    if (!r.string(root, "probes", out.probes)) return false;

    size_t i = 0, n = 0;
    yyjson_val* item = nullptr;
    yyjson_arr_foreach(yyjson_obj_get(root, "reflectionProbes"), i, n, item) {
        r.where = "map.json reflectionProbes[" + std::to_string(i) + "]";
        MapReflectionProbe p;
        if (!r.string(item, "cube", p.cube, true) || !r.string(item, "cubeRgbm", p.cubeRgbm, true) ||
            !r.vec3(item, "position", p.position, true) || !r.vec3(item, "boxMin", p.boxMin, true) || !r.vec3(item, "boxMax", p.boxMax, true))
            return false;
        if (!(p.boxMin.x < p.boxMax.x && p.boxMin.y < p.boxMax.y && p.boxMin.z < p.boxMax.z)) return fail(error, r.where + ": empty box");
        out.reflectionProbes.push_back(p);
    }
    if (out.reflectionProbes.size() > kMaxReflectionProbes) return fail(error, "map.json: more than 7 reflection probes");

    yyjson_arr_foreach(yyjson_obj_get(root, "instances"), i, n, item) {
        r.where = "map.json instances[" + std::to_string(i) + "]";
        MapInstance inst;
        if (!r.string(item, "model", inst.model, true)) return false;
        size_t mi = 0, mn = 0;
        yyjson_val* m = nullptr;
        yyjson_arr_foreach(yyjson_obj_get(item, "materials"), mi, mn, m) {
            if (!yyjson_is_str(m)) return fail(error, r.where + ": materials must be strings");
            inst.materials.emplace_back(yyjson_get_str(m));
        }
        float rows[12];
        if (!r.floats(yyjson_obj_get(item, "transform"), "transform", rows, 12)) return false;
        inst.transform = glm::mat4(1.0f);
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 4; ++col) inst.transform[col][row] = rows[row * 4 + col];
        if (yyjson_val* s = yyjson_obj_get(item, "static")) inst.isStatic = yyjson_get_bool(s);
        if (yyjson_val* o = yyjson_obj_get(item, "occluder")) inst.occluder = yyjson_get_bool(o) && inst.isStatic;
        float layer = 1.0f, probe = 0.0f;
        if (!r.number(item, "layer", layer, 0.0, 15.0) || !r.number(item, "reflection", probe, 0.0, double(out.reflectionProbes.size())))
            return false;
        inst.layer = uint8_t(layer);
        inst.reflectionProbe = uint8_t(probe);
        size_t li = 0, ln = 0;
        yyjson_val* l = nullptr;
        yyjson_arr_foreach(yyjson_obj_get(item, "lightmaps"), li, ln, l) {
            float v[5];
            if (!r.floats(l, "lightmaps[]", v, 5)) return false;
            // Checked as a float first: converting an out-of-range float to int is undefined.
            if (!(v[0] >= -1.0f && v[0] < float(out.lightmapPages)) || v[0] != std::floor(v[0]))
                return fail(error, r.where + ": lightmap page out of range");
            MapLightmapRect rect;
            rect.page = int32_t(v[0]);
            rect.scaleOffset = glm::vec4(v[1], v[2], v[3], v[4]);
            inst.lightmaps.push_back(rect);
        }
        out.instances.push_back(std::move(inst));
    }

    // Every file is data-relative: no absolute paths, drive letters or ".." (a map file must not reach outside the data).
    auto safe = [](const std::string& p) {
        return p.find("..") == std::string::npos && p.find(':') == std::string::npos && (p.empty() || (p[0] != '/' && p[0] != '\\'));
    };
    std::vector<const std::string*> paths = {&out.skyCube, &out.skyCubeRgbm, &out.lightmaps, &out.lightmapsRgbm, &out.shadowmask, &out.probes};
    for (const MapReflectionProbe& p : out.reflectionProbes) paths.insert(paths.end(), {&p.cube, &p.cubeRgbm});
    for (const MapInstance& inst : out.instances) {
        paths.push_back(&inst.model);
        for (const std::string& m : inst.materials) paths.push_back(&m);
    }
    for (const std::string* p : paths)
        if (!safe(*p)) return fail(error, "map.json: \"" + *p + "\" is not a data-relative path");
    return true;
}

std::string writeMapManifest(const MapManifest& m) {
    char buf[512];
    std::string out = "{\n  \"version\": 1,\n";
    out += "  \"sun\": { \"direction\": " + vec3String(m.sunDirection) + ", \"color\": " + vec3String(m.sunColor) + " },\n";
    out += "  \"ambient\": { \"sky\": " + vec3String(m.ambientSky) + ", \"ground\": " + vec3String(m.ambientGround) + " },\n";
    std::snprintf(buf, sizeof(buf), "  \"exposure\": %.9g,\n", m.exposure);
    out += buf;
    if (!m.skyCube.empty()) {
        std::snprintf(buf, sizeof(buf), ", \"intensity\": %.9g, \"sunDiscIntensity\": %.9g, \"sunDiscRadiusDeg\": %.9g },\n",
                      m.skyIntensity, m.sunDiscIntensity, m.sunDiscRadiusDeg);
        out += "  \"sky\": { \"cube\": " + jsonString(m.skyCube) + ", \"cubeRgbm\": " + jsonString(m.skyCubeRgbm) + buf;
    }
    const FogSettings& f = m.fog;
    std::snprintf(buf, sizeof(buf),
                  ", \"start\": %.9g, \"end\": %.9g, \"density\": %.9g, \"maxOpacity\": %.9g,\n"
                  "           \"height\": %s, \"heightBase\": %.9g, \"heightDensity\": %.9g, \"heightFalloff\": %.9g, \"skyHorizon\": %.9g },\n",
                  f.start, f.end, f.density, f.maxOpacity, f.height ? "true" : "false", f.heightBase, f.heightDensity, f.heightFalloff,
                  f.skyHorizon);
    out += "  \"fog\": { \"mode\": " + jsonString(fogModeName(f.mode)) + ", \"color\": " + vec3String(f.color) + buf;
    if (!m.lightmaps.empty()) {
        std::snprintf(buf, sizeof(buf), ", \"pages\": %u, \"size\": %u },\n", m.lightmapPages, m.lightmapSize);
        out += "  \"lightmaps\": { \"bc6h\": " + jsonString(m.lightmaps) + ", \"rgbm\": " + jsonString(m.lightmapsRgbm) +
               ", \"shadowmask\": " + jsonString(m.shadowmask) + buf;
    }
    out += "  \"probes\": " + jsonString(m.probes) + ",\n  \"reflectionProbes\": [";
    for (size_t i = 0; i < m.reflectionProbes.size(); ++i) {
        const MapReflectionProbe& p = m.reflectionProbes[i];
        out += i ? ",\n    " : "\n    ";
        out += "{ \"cube\": " + jsonString(p.cube) + ", \"cubeRgbm\": " + jsonString(p.cubeRgbm) + ", \"position\": " + vec3String(p.position) +
               ", \"boxMin\": " + vec3String(p.boxMin) + ", \"boxMax\": " + vec3String(p.boxMax) + " }";
    }
    out += m.reflectionProbes.empty() ? "],\n" : "\n  ],\n";
    out += "  \"instances\": [";
    for (size_t i = 0; i < m.instances.size(); ++i) {
        const MapInstance& inst = m.instances[i];
        out += i ? ",\n    " : "\n    ";
        out += "{ \"model\": " + jsonString(inst.model);
        if (!inst.materials.empty()) {
            out += ", \"materials\": [";
            for (size_t s = 0; s < inst.materials.size(); ++s) out += (s ? ", " : "") + jsonString(inst.materials[s]);
            out += "]";
        }
        out += ", \"transform\": [";
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 4; ++col) {
                std::snprintf(buf, sizeof(buf), "%s%.9g", row || col ? ", " : "", inst.transform[col][row]);
                out += buf;
            }
        std::snprintf(buf, sizeof(buf), "], \"static\": %s, \"occluder\": %s, \"layer\": %u, \"reflection\": %u",
                      inst.isStatic ? "true" : "false", inst.occluder ? "true" : "false", uint32_t(inst.layer), uint32_t(inst.reflectionProbe));
        out += buf;
        if (!inst.lightmaps.empty()) {
            out += ", \"lightmaps\": [";
            for (size_t l = 0; l < inst.lightmaps.size(); ++l) {
                const MapLightmapRect& r = inst.lightmaps[l];
                std::snprintf(buf, sizeof(buf), "%s[%d, %.9g, %.9g, %.9g, %.9g]", l ? ", " : "", r.page, r.scaleOffset.x, r.scaleOffset.y,
                              r.scaleOffset.z, r.scaleOffset.w);
                out += buf;
            }
            out += "]";
        }
        out += " }";
    }
    out += m.instances.empty() ? "]\n}\n" : "\n  ]\n}\n";
    return out;
}

bool readProbeGrid(std::span<const std::byte> file, ProbeGridData& out, std::string* error) {
    if (file.size() < sizeof(ProbeHeader)) return fail(error, "probe file is truncated");
    ProbeHeader h;
    std::memcpy(&h, file.data(), sizeof(h));
    if (std::memcmp(h.magic, kProbeMagic, 4) != 0) return fail(error, "not a probe file");
    if (h.version != kProbeVersion) return fail(error, "unsupported probe file version");
    const uint64_t count = uint64_t(h.dims[0]) * h.dims[1] * h.dims[2];
    if (count == 0 || count > (1u << 22) || count != h.probeCount) return fail(error, "bad probe grid dimensions");
    if (h.fileSize != file.size() || file.size() != sizeof(ProbeHeader) + count * sizeof(ProbeGridData::Probe))
        return fail(error, "probe file size mismatch");
    for (int a = 0; a < 3; ++a)
        if (!(h.spacing[a] > 0.0f) || !std::isfinite(h.spacing[a]) || !std::isfinite(h.origin[a])) return fail(error, "bad probe grid spacing");
    out.dims[0] = h.dims[0];
    out.dims[1] = h.dims[1];
    out.dims[2] = h.dims[2];
    out.origin = glm::vec3(h.origin[0], h.origin[1], h.origin[2]);
    out.spacing = glm::vec3(h.spacing[0], h.spacing[1], h.spacing[2]);
    out.probes.resize(size_t(count));
    std::memcpy(out.probes.data(), file.data() + sizeof(ProbeHeader), size_t(count) * sizeof(ProbeGridData::Probe));
    for (const ProbeGridData::Probe& p : out.probes)
        for (const glm::vec4& v : p.sh)
            if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z) || !std::isfinite(v.w)) return fail(error, "probe data is not finite");
    return true;
}

std::vector<std::byte> writeProbeGrid(const ProbeGridData& grid) {
    ProbeHeader h{};
    std::memcpy(h.magic, kProbeMagic, 4);
    h.version = kProbeVersion;
    for (int a = 0; a < 3; ++a) {
        h.dims[a] = grid.dims[a];
        h.origin[a] = grid.origin[a];
        h.spacing[a] = grid.spacing[a];
    }
    h.probeCount = uint32_t(grid.probes.size());
    h.fileSize = uint32_t(sizeof(ProbeHeader) + grid.probes.size() * sizeof(ProbeGridData::Probe));
    std::vector<std::byte> out;
    out.reserve(h.fileSize);
    put(out, h);
    for (const ProbeGridData::Probe& p : grid.probes) put(out, p);
    return out;
}

} // namespace assets
