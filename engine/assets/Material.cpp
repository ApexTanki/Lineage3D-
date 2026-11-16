#include "assets/Material.h"

#include <yyjson.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iterator>

namespace assets {
namespace {

constexpr const char* kBlendNames[] = {"opaque", "alphaTest", "alphaBlend", "additive", "premultiplied", "refractive"};
constexpr const char* kFeatureNames[kMaterialFeatureCount] = {"NORMAL_MAP", "SPEC_GLOSS_MAP", "EMISSION_MAP", "DETAIL_LAYER"};
constexpr const char* kFeatureTextures[kMaterialFeatureCount] = {"normal", "specGloss", "emission", "detail"};

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

// Every key of `object` must be one of `allowed`: a typo must not silently drop a texture or parameter.
bool onlyKnownKeys(yyjson_val* object, std::initializer_list<const char*> allowed, const char* where, std::string* error) {
    yyjson_obj_iter it = yyjson_obj_iter_with(object);
    while (yyjson_val* key = yyjson_obj_iter_next(&it)) {
        const char* k = yyjson_get_str(key);
        bool known = false;
        for (const char* a : allowed) known = known || std::strcmp(k, a) == 0;
        if (!known) return fail(error, std::string("unknown key \"") + k + "\" in " + where);
    }
    return true;
}

bool readNumber(yyjson_val* v, const char* name, double lo, double hi, float& out, std::string* error) {
    if (!v) return true;
    if (!yyjson_is_num(v)) return fail(error, std::string(name) + " must be a number");
    const double d = yyjson_get_num(v);
    if (!std::isfinite(d) || d < lo || d > hi) return fail(error, std::string(name) + " is out of range");
    out = float(d);
    return true;
}

// A vector of `count` finite numbers in [lo, hi] (scroll speeds may be negative).
bool readVector(yyjson_val* v, const char* name, float* out, size_t count, double lo, double hi, std::string* error) {
    if (!v) return true;
    if (!yyjson_is_arr(v) || yyjson_arr_size(v) != count)
        return fail(error, std::string(name) + " must be an array of " + std::to_string(count) + " numbers");
    for (size_t i = 0; i < count; ++i) {
        yyjson_val* c = yyjson_arr_get(v, i);
        if (!yyjson_is_num(c)) return fail(error, std::string(name) + " must be an array of numbers");
        const double d = yyjson_get_num(c);
        if (!std::isfinite(d) || d < lo || d > hi) return fail(error, std::string(name) + " is out of range");
        out[i] = float(d);
    }
    return true;
}

// A color: `count` finite, non-negative numbers (linear values; above 1 is allowed for HDR tints and emission).
bool readColor(yyjson_val* v, const char* name, float* out, size_t count, std::string* error) {
    if (!v) return true;
    if (!yyjson_is_arr(v) || yyjson_arr_size(v) != count)
        return fail(error, std::string(name) + " must be an array of " + std::to_string(count) + " numbers");
    for (size_t i = 0; i < count; ++i) {
        yyjson_val* c = yyjson_arr_get(v, i);
        const double d = yyjson_is_num(c) ? yyjson_get_num(c) : -1.0;
        if (!std::isfinite(d) || d < 0.0) return fail(error, std::string(name) + " needs finite, non-negative numbers");
        out[i] = float(d);
    }
    return true;
}

std::string jsonString(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

} // namespace

const char* blendModeName(BlendMode mode) {
    return kBlendNames[uint32_t(mode)];
}

const char* materialFeatureName(uint32_t featureIndex) {
    return featureIndex < kMaterialFeatureCount ? kFeatureNames[featureIndex] : "?";
}

bool parseMaterial(std::span<const std::byte> json, MaterialFile& out, std::string* error) {
    yyjson_doc* doc = yyjson_read(reinterpret_cast<const char*>(json.data()), json.size(), 0);
    if (!doc) return fail(error, "not valid JSON");
    struct Guard {
        yyjson_doc* d;
        ~Guard() { yyjson_doc_free(d); }
    } guard{doc};
    yyjson_val* root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) return fail(error, "the root must be an object");
    if (!onlyKnownKeys(root, {"template", "features", "textures", "params", "blend", "doubleSided", "castShadows"}, "the material", error))
        return false;

    out = MaterialFile{};
    if (yyjson_val* t = yyjson_obj_get(root, "template")) {
        if (!yyjson_is_str(t)) return fail(error, "template must be a string");
        out.templateName = yyjson_get_str(t);
    }
    if (yyjson_val* features = yyjson_obj_get(root, "features")) {
        if (!yyjson_is_arr(features)) return fail(error, "features must be an array of names");
        size_t i = 0, n = 0;
        yyjson_val* f = nullptr;
        yyjson_arr_foreach(features, i, n, f) {
            const char* name = yyjson_get_str(f);
            if (!name) return fail(error, "features must be an array of names");
            if (std::strcmp(name, "ALPHA_TEST") == 0) return fail(error, "ALPHA_TEST follows from \"blend\": \"alphaTest\"; don't list it");
            uint32_t bit = kMaterialFeatureCount;
            for (uint32_t b = 0; b < kMaterialFeatureCount; ++b)
                if (std::strcmp(name, kFeatureNames[b]) == 0) bit = b;
            if (bit == kMaterialFeatureCount) return fail(error, std::string("unknown feature \"") + name + "\"");
            out.features |= 1u << bit;
        }
    }
    if (yyjson_val* textures = yyjson_obj_get(root, "textures")) {
        if (!yyjson_is_obj(textures)) return fail(error, "textures must be an object");
        if (!onlyKnownKeys(textures, {"albedo", "normal", "specGloss", "emission", "detail"}, "textures", error)) return false;
        const char* keys[] = {"albedo", "normal", "specGloss", "emission", "detail"};
        std::string* slots[] = {&out.albedo, &out.normal, &out.specGloss, &out.emission, &out.detail};
        for (size_t i = 0; i < 5; ++i)
            if (yyjson_val* v = yyjson_obj_get(textures, keys[i])) {
                const char* path = yyjson_get_str(v);
                if (!path || !*path) return fail(error, std::string("textures.") + keys[i] + " must be a non-empty path");
                *slots[i] = path;
            }
    }
    const std::string* mapped[kMaterialFeatureCount] = {&out.normal, &out.specGloss, &out.emission, &out.detail};
    for (uint32_t b = 0; b < kMaterialFeatureCount; ++b) {
        const bool feature = ((out.features >> b) & 1u) != 0, texture = !mapped[b]->empty();
        if (feature && !texture)
            return fail(error, std::string("feature ") + kFeatureNames[b] + " needs textures." + kFeatureTextures[b]);
        if (texture && !feature)
            return fail(error, std::string("textures.") + kFeatureTextures[b] + " is set but the feature " + kFeatureNames[b] +
                                   " isn't listed");
    }

    // A parameter multiplies its map when there is one, so it then defaults to 1 (M2 plan D2).
    if (out.features & kFeatureSpecGlossMap) {
        out.specular = glm::vec3(1.0f);
        out.gloss = 1.0f;
    }
    if (out.features & kFeatureEmissionMap) out.emissionColor = glm::vec3(1.0f);
    if (yyjson_val* params = yyjson_obj_get(root, "params")) {
        if (!yyjson_is_obj(params)) return fail(error, "params must be an object");
        if (!onlyKnownKeys(params,
                           {"baseColor", "specular", "gloss", "emission", "emissionIntensity", "normalStrength", "alphaCutoff",
                            "reflection", "uvScroll", "uvRotate", "flipbook", "detailScale", "detailScroll", "detailStrength",
                            "refractionStrength", "refractionTint"},
                           "params", error))
            return false;
        if (!readColor(yyjson_obj_get(params, "baseColor"), "baseColor", &out.baseColor.x, 4, error) ||
            !readColor(yyjson_obj_get(params, "specular"), "specular", &out.specular.x, 3, error) ||
            !readNumber(yyjson_obj_get(params, "gloss"), "gloss", 0.0, 1.0, out.gloss, error) ||
            !readColor(yyjson_obj_get(params, "emission"), "emission", &out.emissionColor.x, 3, error) ||
            !readNumber(yyjson_obj_get(params, "emissionIntensity"), "emissionIntensity", 0.0, 1e6, out.emissionIntensity, error) ||
            !readNumber(yyjson_obj_get(params, "normalStrength"), "normalStrength", 0.0, 10.0, out.normalStrength, error) ||
            !readNumber(yyjson_obj_get(params, "alphaCutoff"), "alphaCutoff", 0.0, 1.0, out.alphaCutoff, error) ||
            !readNumber(yyjson_obj_get(params, "reflection"), "reflection", 0.0, 1.0, out.reflection, error) ||
            !readVector(yyjson_obj_get(params, "uvScroll"), "uvScroll", &out.uvScroll.x, 2, -100.0, 100.0, error) ||
            !readNumber(yyjson_obj_get(params, "uvRotate"), "uvRotate", -3600.0, 3600.0, out.uvRotate, error) ||
            !readVector(yyjson_obj_get(params, "flipbook"), "flipbook", &out.flipbook.x, 3, 0.0, 240.0, error) ||
            !readNumber(yyjson_obj_get(params, "detailScale"), "detailScale", 0.01, 1000.0, out.detailScale, error) ||
            !readVector(yyjson_obj_get(params, "detailScroll"), "detailScroll", &out.detailScroll.x, 2, -100.0, 100.0, error) ||
            !readNumber(yyjson_obj_get(params, "detailStrength"), "detailStrength", 0.0, 1.0, out.detailStrength, error) ||
            !readNumber(yyjson_obj_get(params, "refractionStrength"), "refractionStrength", 0.0, 10.0, out.refractionStrength, error) ||
            !readColor(yyjson_obj_get(params, "refractionTint"), "refractionTint", &out.refractionTint.x, 3, error))
            return false;
        if (out.baseColor.a > 1.0f) return fail(error, "baseColor alpha (opacity) must be at most 1");
        if (out.refractionTint.r > 1.0f || out.refractionTint.g > 1.0f || out.refractionTint.b > 1.0f)
            return fail(error, "refractionTint is the share of each channel the surface lets through: 0 to 1");
        // Flipbook cells are counted in whole columns and rows (at most 64 each).
        for (int i = 0; i < 2; ++i)
            if (out.flipbook[i] < 1.0f || out.flipbook[i] > 64.0f || out.flipbook[i] != std::floor(out.flipbook[i]))
                return fail(error, "flipbook columns and rows must be whole numbers from 1 to 64");
        if (yyjson_obj_get(params, "detailScale") || yyjson_obj_get(params, "detailScroll") || yyjson_obj_get(params, "detailStrength"))
            if (!(out.features & kFeatureDetailLayer)) return fail(error, "detail parameters need the DETAIL_LAYER feature");
        // A flipbook maps the mesh's UVs into one cell; scrolled or rotated UVs would show the neighbouring cells.
        if (out.flipbook.x * out.flipbook.y > 1.0f && (out.uvScroll != glm::vec2(0.0f) || out.uvRotate != 0.0f))
            return fail(error, "a flipbook can't also scroll or rotate its UVs (use the detail layer for a scrolling overlay)");
    }
    if (yyjson_val* blend = yyjson_obj_get(root, "blend")) {
        const char* name = yyjson_get_str(blend);
        bool known = false;
        for (uint32_t i = 0; name && i < std::size(kBlendNames); ++i)
            if (std::strcmp(name, kBlendNames[i]) == 0) {
                out.blend = BlendMode(i);
                known = true;
            }
        if (!known) return fail(error, "unknown blend mode");
    }
    if (yyjson_val* doubleSided = yyjson_obj_get(root, "doubleSided")) {
        if (!yyjson_is_bool(doubleSided)) return fail(error, "doubleSided must be true or false");
        out.doubleSided = yyjson_get_bool(doubleSided);
    }
    if (yyjson_val* params = yyjson_obj_get(root, "params"))
        if ((yyjson_obj_get(params, "refractionStrength") || yyjson_obj_get(params, "refractionTint")) && out.blend != BlendMode::Refractive)
            return fail(error, "refraction parameters need \"blend\": \"refractive\"");
    // Blended and refractive surfaces (water, glass) let light through, so they never cast shadows.
    const bool blended = out.blend == BlendMode::AlphaBlend || out.blend == BlendMode::Additive || out.blend == BlendMode::Premultiplied ||
                         out.blend == BlendMode::Refractive;
    out.castShadows = !blended;
    if (yyjson_val* cast = yyjson_obj_get(root, "castShadows")) {
        if (!yyjson_is_bool(cast)) return fail(error, "castShadows must be true or false");
        if (blended && yyjson_get_bool(cast))
            return fail(error, "blended and refractive materials don't cast shadows (castShadows must be false)");
        out.castShadows = yyjson_get_bool(cast);
    }
    return true;
}

std::string writeMaterial(const MaterialFile& m) {
    std::string out = "{\n  \"template\": " + jsonString(m.templateName) + ",\n  \"features\": [";
    const char* separator = "";
    for (uint32_t b = 0; b < kMaterialFeatureCount; ++b)
        if ((m.features >> b) & 1u) {
            out += separator;
            out += jsonString(kFeatureNames[b]);
            separator = ", ";
        }
    out += "],\n  \"textures\": {";
    separator = " ";
    const std::pair<const char*, const std::string*> textures[] = {{"albedo", &m.albedo},
                                                                   {"normal", &m.normal},
                                                                   {"specGloss", &m.specGloss},
                                                                   {"emission", &m.emission},
                                                                   {"detail", &m.detail}};
    for (const auto& [key, path] : textures)
        if (!path->empty()) {
            out += separator + jsonString(key) + ": " + jsonString(*path);
            separator = ", ";
        }
    out += " },\n";
    char params[1024];
    std::snprintf(params, sizeof(params),
                  "  \"params\": { \"baseColor\": [%.9g, %.9g, %.9g, %.9g], \"specular\": [%.9g, %.9g, %.9g], \"gloss\": %.9g,\n"
                  "              \"emission\": [%.9g, %.9g, %.9g], \"emissionIntensity\": %.9g, \"normalStrength\": %.9g,"
                  " \"alphaCutoff\": %.9g, \"reflection\": %.9g,\n"
                  "              \"uvScroll\": [%.9g, %.9g], \"uvRotate\": %.9g, \"flipbook\": [%.9g, %.9g, %.9g]",
                  m.baseColor.r, m.baseColor.g, m.baseColor.b, m.baseColor.a, m.specular.r, m.specular.g, m.specular.b, m.gloss,
                  m.emissionColor.r, m.emissionColor.g, m.emissionColor.b, m.emissionIntensity, m.normalStrength, m.alphaCutoff,
                  m.reflection, m.uvScroll.x, m.uvScroll.y, m.uvRotate, m.flipbook.x, m.flipbook.y, m.flipbook.z);
    out += params;
    if (m.features & kFeatureDetailLayer) {
        std::snprintf(params, sizeof(params), ",\n              \"detailScale\": %.9g, \"detailScroll\": [%.9g, %.9g], \"detailStrength\": %.9g",
                      m.detailScale, m.detailScroll.x, m.detailScroll.y, m.detailStrength);
        out += params;
    }
    if (m.blend == BlendMode::Refractive) {
        std::snprintf(params, sizeof(params), ",\n              \"refractionStrength\": %.9g, \"refractionTint\": [%.9g, %.9g, %.9g]",
                      m.refractionStrength, m.refractionTint.r, m.refractionTint.g, m.refractionTint.b);
        out += params;
    }
    out += " },\n";
    out += "  \"blend\": " + jsonString(blendModeName(m.blend)) + ",\n  \"doubleSided\": ";
    out += m.doubleSided ? "true" : "false";
    const bool blended = m.blend == BlendMode::AlphaBlend || m.blend == BlendMode::Additive || m.blend == BlendMode::Premultiplied ||
                         m.blend == BlendMode::Refractive;
    out += ",\n  \"castShadows\": ";
    out += m.castShadows && !blended ? "true" : "false";
    out += "\n}\n";
    return out;
}

} // namespace assets
