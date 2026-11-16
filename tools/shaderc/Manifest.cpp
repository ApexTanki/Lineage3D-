#include "Manifest.h"

#include <yyjson.h>

#include <filesystem>

namespace tools {
namespace fs = std::filesystem;

bool loadManifest(const std::string& path, Manifest& out, std::string& error) {
    yyjson_read_err err{};
    yyjson_doc* doc = yyjson_read_file(path.c_str(), YYJSON_READ_ALLOW_COMMENTS | YYJSON_READ_ALLOW_TRAILING_COMMAS,
                                       nullptr, &err);
    if (!doc) {
        error = path + ": " + (err.msg ? err.msg : "cannot read");
        return false;
    }
    yyjson_val* root = yyjson_doc_get_root(doc);
    const fs::path dir = fs::path(path).parent_path();
    auto str = [&](const char* key) -> std::string {
        const char* s = yyjson_get_str(yyjson_obj_get(root, key));
        return s ? s : "";
    };
    out.templateName = str("template");
    const std::string vertexName = str("vertex"), fragmentName = str("fragment"); // copied before the doc is freed
    out.vertexPath = (dir / vertexName).string();
    out.fragmentPath = (dir / fragmentName).string();
    if (yyjson_val* mv = yyjson_obj_get(root, "maxVariants")) out.maxVariants = uint32_t(yyjson_get_uint(mv));

    uint32_t bit = 0;
    yyjson_val* axes = yyjson_obj_get(root, "axes");
    size_t idx, max;
    yyjson_val* axis;
    yyjson_arr_foreach(axes, idx, max, axis) {
        ManifestAxis a;
        const char* name = yyjson_get_str(yyjson_obj_get(axis, "name"));
        a.name = name ? name : "";
        yyjson_val* values = yyjson_obj_get(axis, "values");
        if (values && (!yyjson_is_arr(values) || yyjson_arr_size(values) < 2)) {
            error = path + ": axis \"" + a.name + "\": \"values\" must be an array of at least two strings";
            yyjson_doc_free(doc);
            return false;
        }
        size_t vi, vmax;
        yyjson_val* v;
        yyjson_arr_foreach(values, vi, vmax, v) {
            const char* s = yyjson_get_str(v);
            if (!s) {
                error = path + ": axis \"" + a.name + "\" has a non-string value";
                yyjson_doc_free(doc);
                return false;
            }
            a.values.push_back(s);
        }
        uint32_t bits = 0;
        while ((1u << bits) < a.valueCount()) ++bits;
        a.bitOffset = uint8_t(bit);
        a.bitCount = uint8_t(bits);
        bit += bits;
        out.axes.push_back(std::move(a));
    }
    yyjson_doc_free(doc);

    if (out.templateName.empty() || vertexName.empty() || fragmentName.empty()) {
        error = path + ": \"template\", \"vertex\" and \"fragment\" are required";
        return false;
    }
    if (bit > 64) {
        error = path + ": permutation axes need more than 64 key bits";
        return false;
    }
    for (const ManifestAxis& a : out.axes)
        if (a.name.empty() || a.values.size() == 1) {
            error = path + ": every axis needs a name, and enum axes need at least two values";
            return false;
        }
    // Checked here, with an early out, so a huge axis set fails cleanly instead of overflowing or exhausting memory.
    uint64_t total = 1;
    for (const ManifestAxis& a : out.axes) {
        total *= a.valueCount();
        if (total > out.maxVariants) {
            error = path + ": more permutations than maxVariants (" + std::to_string(out.maxVariants) + ")";
            return false;
        }
    }
    return true;
}

std::vector<Permutation> enumeratePermutations(const Manifest& m) {
    uint64_t total = 1;
    for (const ManifestAxis& a : m.axes) total *= a.valueCount();
    std::vector<Permutation> result;
    result.reserve(size_t(total));
    for (uint64_t i = 0; i < total; ++i) {
        Permutation p;
        uint64_t rest = i;
        for (const ManifestAxis& a : m.axes) {
            const uint32_t value = uint32_t(rest % a.valueCount());
            rest /= a.valueCount();
            p.key |= uint64_t(value) << a.bitOffset;
            if (a.values.empty()) {
                if (value) p.defines.push_back(a.name);
            } else {
                p.defines.push_back(a.name + "=" + std::to_string(value));
                p.defines.push_back(a.name + "_" + a.values[value]);
            }
        }
        result.push_back(std::move(p));
    }
    return result;
}

} // namespace tools
