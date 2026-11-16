#include "assets/TierFormat.h"

#include <yyjson.h>

#include <cmath>
#include <cstring>
#include <initializer_list>
#include <utility>

namespace assets {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

// Messages name the tier and block ("tier \"medium\", shadows: ...") so an author finds the line at once.
struct Reader {
    std::string* error;
    std::string where;

    bool bad(const std::string& message) const { return fail(error, where + ": " + message); }

    // Exactly the keys of `keys`, each once: a tier defines every setting, so a missing key is as wrong as a typo.
    bool exactKeys(yyjson_val* object, std::initializer_list<const char*> keys) const {
        if (!yyjson_is_obj(object)) return bad("must be an object");
        yyjson_obj_iter it = yyjson_obj_iter_with(object);
        size_t index = 0;
        while (yyjson_val* key = yyjson_obj_iter_next(&it)) {
            const char* k = yyjson_get_str(key);
            bool known = false;
            for (const char* a : keys) known = known || std::strcmp(k, a) == 0;
            if (!known) return bad(std::string("unknown key \"") + k + "\"");
            yyjson_obj_iter earlier = yyjson_obj_iter_with(object);
            for (size_t i = 0; i < index; ++i)
                if (std::strcmp(yyjson_get_str(yyjson_obj_iter_next(&earlier)), k) == 0) return bad(std::string("\"") + k + "\" is given twice");
            ++index;
        }
        for (const char* k : keys)
            if (!yyjson_obj_get(object, k)) return bad(std::string("needs \"") + k + "\"");
        return true;
    }
    bool boolean(yyjson_val* object, const char* name, bool& out) const {
        yyjson_val* v = yyjson_obj_get(object, name);
        if (!yyjson_is_bool(v)) return bad(std::string(name) + " must be true or false");
        out = yyjson_get_bool(v);
        return true;
    }
    bool integer(yyjson_val* object, const char* name, uint32_t lo, uint32_t hi, uint32_t& out) const {
        yyjson_val* v = yyjson_obj_get(object, name);
        const double d = yyjson_is_num(v) ? yyjson_get_num(v) : -1.0;
        if (!std::isfinite(d) || d != std::floor(d) || d < double(lo) || d > double(hi))
            return bad(std::string(name) + " must be a whole number from " + std::to_string(lo) + " to " + std::to_string(hi));
        out = uint32_t(d);
        return true;
    }
    bool number(yyjson_val* object, const char* name, double lo, double hi, float& out) const {
        yyjson_val* v = yyjson_obj_get(object, name);
        const double d = yyjson_is_num(v) ? yyjson_get_num(v) : std::nan("");
        if (!std::isfinite(d) || d < lo || d > hi) return bad(std::string(name) + " must be a number from " + std::to_string(lo) + " to " + std::to_string(hi));
        out = float(d);
        return true;
    }
};

bool parseTier(yyjson_val* t, TierDesc& out, std::string* error) {
    Reader r{error, "a tier"};
    yyjson_val* name = yyjson_is_obj(t) ? yyjson_obj_get(t, "name") : nullptr;
    if (yyjson_is_str(name)) r.where = std::string("tier \"") + yyjson_get_str(name) + "\"";
    if (!r.exactKeys(t, {"name", "hdr", "msaa", "fxaa", "shadows", "ssao", "bloom", "softParticles", "refraction", "heatHaze", "p2",
                         "maxLocalLights", "anisotropy", "budgets"}))
        return false;
    out = TierDesc{};
    out.name = yyjson_get_str(name) ? yyjson_get_str(name) : "";
    bool known = false;
    for (const char* n : kTierNames) known = known || out.name == n;
    if (!known) return r.bad("name must be low, medium, high or custom");
    if (!r.boolean(t, "hdr", out.hdr) || !r.integer(t, "msaa", 1, 4, out.msaa) || !r.boolean(t, "fxaa", out.fxaa) ||
        !r.boolean(t, "ssao", out.ssao) || !r.boolean(t, "bloom", out.bloom) || !r.boolean(t, "softParticles", out.softParticles) ||
        !r.boolean(t, "refraction", out.refraction) || !r.boolean(t, "heatHaze", out.heatHaze) ||
        !r.integer(t, "maxLocalLights", 0, 8, out.maxLocalLights) || !r.integer(t, "anisotropy", 1, 16, out.anisotropy))
        return false;
    if (out.msaa == 3) return r.bad("msaa must be 1, 2 or 4");

    yyjson_val* shadows = yyjson_obj_get(t, "shadows");
    const Reader s{error, r.where + ", shadows"};
    if (!s.exactKeys(shadows, {"cascades", "size", "filter", "staticCasters", "spotShadows", "blobShadows", "blobStrength", "distance"}) ||
        !s.integer(shadows, "cascades", 0, 3, out.cascades) || !s.integer(shadows, "size", 256, 4096, out.shadowMapSize) ||
        !s.boolean(shadows, "staticCasters", out.staticCasters) || !s.boolean(shadows, "spotShadows", out.spotShadows) ||
        !s.boolean(shadows, "blobShadows", out.blobShadows) || !s.number(shadows, "blobStrength", 0.0, 1.0, out.blobStrength) ||
        !s.number(shadows, "distance", 10.0, 1000.0, out.shadowDistance))
        return false;
    if ((out.shadowMapSize & (out.shadowMapSize - 1)) != 0) return s.bad("size must be a power of two");
    const char* filter = yyjson_get_str(yyjson_obj_get(shadows, "filter"));
    if (filter && std::strcmp(filter, "hardware") == 0) out.shadowFilter = TierShadowFilter::Hardware;
    else if (filter && std::strcmp(filter, "poisson") == 0) out.shadowFilter = TierShadowFilter::Poisson;
    else return s.bad("filter must be \"hardware\" or \"poisson\"");

    yyjson_val* p2 = yyjson_obj_get(t, "p2");
    const Reader p{error, r.where + ", p2"};
    if (!p.exactKeys(p2, {"depthOfField", "motionBlur", "godRays", "ssr"}) || !p.boolean(p2, "depthOfField", out.depthOfField) ||
        !p.boolean(p2, "motionBlur", out.motionBlur) || !p.boolean(p2, "godRays", out.godRays) || !p.boolean(p2, "ssr", out.ssr))
        return false;

    yyjson_val* budgets = yyjson_obj_get(t, "budgets");
    const Reader b{error, r.where + ", budgets"};
    if (!b.exactKeys(budgets, {"textures", "meshes", "targets", "buffers"}) || !b.integer(budgets, "textures", 16, 65536, out.textureBudgetMiB) ||
        !b.integer(budgets, "meshes", 16, 65536, out.meshBudgetMiB) || !b.integer(budgets, "targets", 16, 65536, out.targetBudgetMiB) ||
        !b.integer(budgets, "buffers", 16, 65536, out.bufferBudgetMiB))
        return false;

    // The LDR path has none of these (M8 plan D8): a tier that asks for them without HDR would promise what it can't show.
    if (!out.hdr && (out.ssao || out.bloom || out.refraction || out.heatHaze || out.depthOfField || out.motionBlur || out.godRays || out.ssr))
        return r.bad("SSAO, bloom, refraction, heat haze and the P2 effects need hdr");
    return true;
}

} // namespace

bool parseTiers(std::span<const std::byte> json, TierFile& out, std::string* error) {
    yyjson_doc* doc = yyjson_read(reinterpret_cast<const char*>(json.data()), json.size(), 0);
    if (!doc) return fail(error, "not valid JSON");
    struct Guard {
        yyjson_doc* d;
        ~Guard() { yyjson_doc_free(d); }
    } guard{doc};
    yyjson_val* root = yyjson_doc_get_root(doc);
    const Reader r{error, "the tiers file"};
    if (!r.exactKeys(root, {"tiers"})) return false;
    yyjson_val* tiers = yyjson_obj_get(root, "tiers");
    constexpr size_t kCount = sizeof(kTierNames) / sizeof(kTierNames[0]);
    if (!yyjson_is_arr(tiers) || yyjson_arr_size(tiers) != kCount) return r.bad("tiers must be an array of the four tiers low, medium, high and custom");
    TierFile file;
    file.tiers.resize(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        if (!parseTier(yyjson_arr_get(tiers, i), file.tiers[i], error)) return false;
        for (size_t j = 0; j < i; ++j)
            if (file.tiers[j].name == file.tiers[i].name) return r.bad("tier \"" + file.tiers[i].name + "\" is given twice");
    }
    out = std::move(file);
    return true;
}

const TierDesc* findTier(const TierFile& file, std::string_view name) {
    for (const TierDesc& t : file.tiers)
        if (t.name == name) return &t;
    return nullptr;
}

} // namespace assets
