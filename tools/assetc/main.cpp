// fl_assetc — offline asset converter (ARCHITECTURE.md §5.7).
//   fl_assetc cook    --manifest <cook.json> --source <dir> --external <dir> --out <data dir> [--force]
//   fl_assetc model   <in.gltf | builtin:...> --out <x.fmodel> --data <data dir> [--material <a.fmat>[,<b.fmat>...]]
//   fl_assetc texture <in.png | builtin:...> --role <role> --out <x.ktx2> [--cutoff <0-1>]
//            roles: albedo, albedo_alpha, albedo_cutout, normal, specgloss, emission, raw_srgb (ImageCook.h)
//   fl_assetc map     <x.map.json> --out <data-relative dir> --data <data dir>   (bakes a map, MapCook.h)

#include "Builtins.h"
#include "Cook.h"
#include "ImageCook.h"
#include "MapCook.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  fl_assetc cook --manifest <cook.json> --source <dir> --external <dir> --out <data dir> [--force]\n"
                 "  fl_assetc model <in.gltf|builtin:...> --out <x.fmodel> --data <data dir> [--material <a.fmat>[,<b.fmat>...]]\n"
                 "  fl_assetc texture <in.png|builtin:...> --role <role> --out <x.ktx2> [--cutoff <0-1>]\n"
                 "    roles: albedo, albedo_alpha, albedo_cutout, normal, specgloss, emission, raw_srgb\n"
                 "  fl_assetc map <x.map.json> --out <data-relative dir> --data <data dir>\n");
    return 2;
}

std::vector<std::string> splitList(const std::string& list) {
    std::vector<std::string> items;
    for (size_t start = 0; !list.empty() && start <= list.size();) {
        const size_t comma = std::min(list.find(',', start), list.size());
        items.push_back(list.substr(start, comma - start));
        start = comma + 1;
    }
    return items;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string command = argv[1];
    std::string input;
    std::map<std::string, std::string> opts;
    bool force = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--force") force = true;
        else if (a.rfind("--", 0) == 0 && i + 1 < argc) opts[a.substr(2)] = argv[++i];
        else if (input.empty()) input = a;
        else return usage();
    }

    if (command == "cook") {
        tools::CookOptions o;
        o.manifest = opts["manifest"];
        o.sourceDir = opts["source"];
        o.externalDir = opts["external"];
        o.outDir = opts["out"];
        o.toolPath = argv[0];
        o.force = force;
        if (o.manifest.empty() || o.sourceDir.empty() || o.outDir.empty()) return usage();
        return tools::runCook(o);
    }
    std::string error;
    if (command == "model") {
        if (input.empty() || opts["out"].empty()) return usage();
        if (!tools::cookModel(input, splitList(opts["material"]), opts["out"], opts.count("data") ? opts["data"] : ".", error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        return 0;
    }
    if (command == "texture") {
        tools::TextureRole role = tools::TextureRole::Albedo;
        if (input.empty() || opts["out"].empty() || !tools::parseTextureRole(opts["role"], role)) return usage();
        tools::Image image;
        const bool loaded = input.rfind("builtin:", 0) == 0 ? tools::makeBuiltinImage(input.substr(8), image, error)
                                                            : tools::loadImageFile(input, image, error);
        float cutoff = 0.5f;
        if (opts.count("cutoff")) {
            char* end = nullptr;
            cutoff = std::strtof(opts["cutoff"].c_str(), &end);
            if (end == opts["cutoff"].c_str() || *end != '\0' || !(cutoff >= 0.0f && cutoff <= 1.0f)) return usage();
        }
        if (!loaded || !tools::cookTexture(image, role, opts["out"], error, 2048, cutoff)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        return 0;
    }
    if (command == "map") {
        if (input.empty() || opts["out"].empty() || opts["data"].empty()) return usage();
        if (!tools::cookMap(input, opts["out"], opts["data"], error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        return 0;
    }
    return usage();
}
