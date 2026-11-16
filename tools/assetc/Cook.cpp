#include "Cook.h"

#include "Builtins.h"
#include "GltfImport.h"
#include "ImageCook.h"
#include "MapCook.h"
#include "MeshBuilder.h"
#include "Rigs.h"
#include "assets/AnimFile.h"
#include "assets/Material.h"
#include "assets/ParticleFormat.h"
#include "assets/TierFormat.h"

#include <yyjson.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace tools {
namespace fs = std::filesystem;

namespace {

using Clock = fs::file_time_type;

Clock fileTime(const fs::path& p) {
    std::error_code ec;
    const Clock t = fs::last_write_time(p, ec);
    return ec ? Clock::min() : t;
}

// glTF files pull in buffers and images next to them, so the whole directory counts as the input.
Clock newestInDirectory(const fs::path& dir) {
    Clock newest = Clock::min();
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec))
        if (it->is_regular_file(ec)) newest = std::max(newest, fileTime(it->path()));
    return newest;
}

template <size_t N>
void copyName(char (&dst)[N], const std::string& src) {
    std::memset(dst, 0, N);
    std::memcpy(dst, src.data(), std::min(src.size(), N - 1));
}

const char* str(yyjson_val* obj, const char* key) {
    const char* s = yyjson_get_str(yyjson_obj_get(obj, key));
    return s ? s : "";
}

bool isBuiltin(const std::string& input) {
    return input.rfind("builtin:", 0) == 0;
}

} // namespace

bool cookModel(const std::string& input, const std::vector<std::string>& materials, const std::string& outPath,
               const std::string& dataDir, std::string& error, bool generateLods, const std::string& animsPath) {
    assets::ModelData model;
    assets::AnimData clips;
    if (input.rfind("rig:", 0) == 0) {
        RigModel rig;
        if (!makeBuiltinRig(input.substr(4), rig, error) || !assembleRig(rig, materials, generateLods, model, error)) return false;
        clips = std::move(rig.clips);
    } else if (isBuiltin(input)) {
        RawMesh raw;
        if (!makeBuiltinMesh(input.substr(8), raw, error)) return false;
        raw.generateLods = generateLods;
        uint32_t slots = 0;
        for (const RawSubmesh& s : raw.submeshes) slots = std::max(slots, s.materialSlot + 1);
        if (materials.size() != slots) {
            error = input + " has " + std::to_string(slots) + " material slot(s) but " + std::to_string(materials.size()) +
                    " material path(s) were given";
            return false;
        }
        assets::ModelData::Mesh mesh;
        if (!buildMesh(std::move(raw), mesh, model.submeshes, error)) return false;
        model.meshes.push_back(std::move(mesh));
        assets::ModelNode node{};
        copyName(node.name, input.substr(8));
        node.parent = -1;
        node.mesh = 0;
        node.rotation[3] = 1.0f;
        node.scale[0] = node.scale[1] = node.scale[2] = 1.0f;
        model.nodes.push_back(node);
        for (uint32_t s = 0; s < slots; ++s) {
            assets::ModelMaterialSlot slot{};
            copyName(slot.name, slots == 1 ? std::string("default") : "slot" + std::to_string(s));
            copyName(slot.path, materials[s]);
            model.materials.push_back(slot);
        }
    } else {
        if (!materials.empty()) std::printf("  note: %s brings its own materials; the given material paths are ignored\n", input.c_str());
        const std::string name = fs::path(outPath).stem().string();
        if (!importGltf(input, name, dataDir, model, error, generateLods, animsPath.empty() ? nullptr : &clips)) return false;
    }
    // The clip library (M6 plan D3) next to the model, when the manifest asks for one.
    if (!animsPath.empty()) {
        if (clips.clips.empty()) {
            error = input + " has no animations for \"anims\"";
            return false;
        }
        std::vector<std::byte> animBytes;
        if (!assets::writeAnim(clips, animBytes, error)) return false;
        assets::AnimView animView;
        std::string animCheck;
        if (!assets::openAnim(animBytes, animView, &animCheck)) { // the runtime reader must accept what the writer produced
            error = "writer produced an invalid clip library: " + animCheck;
            return false;
        }
        if (!writeFileAtomic(animsPath, animBytes.data(), animBytes.size(), error)) return false;
        std::printf("  clips:");
        for (const assets::AnimData::Clip& c : clips.clips) std::printf(" %s (%.2f s)", c.name.c_str(), double(c.duration));
        std::printf("\n");
    }
    // LOD summary (M5 plan D5) for meshes that got more than one level: triangles per level over all submeshes and the
    // screen size (bounding-sphere diameter over viewport height) each level serves up to.
    for (const assets::ModelData::Mesh& mesh : model.meshes) {
        const assets::ModelSubmesh& s = model.submeshes[mesh.info.firstSubmesh];
        if (s.lodCount < 2) continue;
        std::string line = "  LODs: " + std::to_string(s.lodCount) + " levels, triangles";
        for (uint32_t l = 0; l < s.lodCount; ++l) {
            uint32_t triangles = 0;
            for (uint32_t k = 0; k < mesh.info.submeshCount; ++k) triangles += model.submeshes[mesh.info.firstSubmesh + k].lods[l].indexCount / 3;
            line += (l ? " / " : " ") + std::to_string(triangles);
        }
        line += ", up to screen size";
        char size[32];
        for (uint32_t l = 1; l < s.lodCount; ++l) {
            std::snprintf(size, sizeof(size), "%s%.3g", l > 1 ? " / " : " ", s.lods[l].screenSize);
            line += size;
        }
        std::printf("%s\n", line.c_str());
    }
    const std::vector<std::byte> bytes = assets::writeModel(model);
    // Self-check: the runtime reader must accept what the writer produced.
    assets::ModelView view;
    std::string check;
    if (!assets::openModel(bytes, view, &check)) {
        error = "writer produced an invalid model: " + check;
        return false;
    }
    return writeFileAtomic(outPath, bytes.data(), bytes.size(), error);
}

int runCook(const CookOptions& o) {
    std::ifstream file(o.manifest, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "error: cannot open %s\n", o.manifest.c_str());
        return 1;
    }
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    yyjson_doc* doc = yyjson_read(text.data(), text.size(), 0);
    if (!doc) {
        std::fprintf(stderr, "error: %s is not valid JSON\n", o.manifest.c_str());
        return 1;
    }
    struct Guard {
        yyjson_doc* d;
        ~Guard() { yyjson_doc_free(d); }
    } guard{doc};
    yyjson_val* root = yyjson_doc_get_root(doc);

    const Clock baseline = std::max(fileTime(o.manifest), fileTime(o.toolPath));
    const fs::path out(o.outDir);
    auto resolve = [&](const std::string& input) -> fs::path {
        if (input.rfind("external:", 0) == 0) return fs::path(o.externalDir) / input.substr(9);
        return fs::path(o.sourceDir) / input;
    };
    auto stale = [&](const fs::path& output, Clock inputTime) {
        return o.force || !fs::exists(output) || fileTime(output) < std::max(baseline, inputTime);
    };

    int failures = 0, cooked = 0, upToDate = 0, skipped = 0;
    auto report = [&](bool ok, const std::string& what, const std::string& error, double seconds) {
        if (ok) {
            std::printf("cooked %s (%.1f s)\n", what.c_str(), seconds);
            ++cooked;
        } else {
            std::fprintf(stderr, "error: %s: %s\n", what.c_str(), error.c_str());
            ++failures;
        }
    };
    auto timed = [](auto&& fn) {
        const auto start = std::chrono::steady_clock::now();
        const bool ok = fn();
        return std::make_pair(ok, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    };

    yyjson_val* item = nullptr;
    size_t idx = 0, max = 0;
    // The textures this manifest produces: a particle preset may only name one of them (M7 plan D2), so a typo in a
    // preset fails the build instead of drawing white squares.
    std::vector<std::string> textureOutputs;
    yyjson_arr_foreach(yyjson_obj_get(root, "textures"), idx, max, item) textureOutputs.emplace_back(str(item, "out"));
    yyjson_arr_foreach(yyjson_obj_get(root, "copy"), idx, max, item) {
        const fs::path from = fs::path(o.sourceDir) / str(item, "from");
        const fs::path to = out / str(item, "to");
        std::error_code ec;
        for (auto it = fs::recursive_directory_iterator(from, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            const fs::path dest = to / fs::relative(it->path(), from, ec);
            if (!stale(dest, fileTime(it->path()))) { // a newer tool or manifest re-copies (and so re-validates) too
                ++upToDate;
                continue;
            }
            if (it->path().extension() == ".fmat") { // an invalid material fails the build here, not at runtime
                std::ifstream in(it->path(), std::ios::binary);
                const std::string materialText((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                assets::MaterialFile material;
                std::string parseError;
                if (!assets::parseMaterial(std::as_bytes(std::span(materialText.data(), materialText.size())), material, &parseError)) {
                    std::fprintf(stderr, "error: %s: %s\n", it->path().string().c_str(), parseError.c_str());
                    ++failures;
                    continue;
                }
            }
            if (it->path().extension() == ".fpart") { // likewise a particle preset, and the textures it names
                std::ifstream in(it->path(), std::ios::binary);
                const std::string presetText((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                assets::ParticleEffectFile preset;
                std::string parseError;
                if (assets::parseParticleEffect(std::as_bytes(std::span(presetText.data(), presetText.size())), preset, &parseError))
                    for (const assets::ParticleEmitterDesc& e : preset.emitters)
                        if (!e.texture.empty() && std::find(textureOutputs.begin(), textureOutputs.end(), e.texture) == textureOutputs.end())
                            parseError = "emitter \"" + e.name + "\": texture " + e.texture + " isn't produced by the manifest";
                if (!parseError.empty()) {
                    std::fprintf(stderr, "error: %s: %s\n", it->path().string().c_str(), parseError.c_str());
                    ++failures;
                    continue;
                }
            }
            if (it->path().filename() == "tiers.json") { // the quality tiers (M9 plan D1): a bad file fails the build too
                std::ifstream in(it->path(), std::ios::binary);
                const std::string tiersText((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                assets::TierFile tiers;
                std::string parseError;
                if (!assets::parseTiers(std::as_bytes(std::span(tiersText.data(), tiersText.size())), tiers, &parseError)) {
                    std::fprintf(stderr, "error: %s: %s\n", it->path().string().c_str(), parseError.c_str());
                    ++failures;
                    continue;
                }
            }
            fs::create_directories(dest.parent_path(), ec);
            fs::copy_file(it->path(), dest, fs::copy_options::overwrite_existing, ec);
            // copy_file keeps the source's time, which is older than the tool: without a fresh time the copy would look
            // stale on every later cook and be copied (and re-validated) again.
            if (!ec) fs::last_write_time(dest, Clock::clock::now(), ec);
            if (ec) {
                std::fprintf(stderr, "error: copy %s: %s\n", it->path().string().c_str(), ec.message().c_str());
                ++failures;
            } else {
                ++cooked;
            }
        }
    }

    yyjson_arr_foreach(yyjson_obj_get(root, "textures"), idx, max, item) {
        const std::string input = str(item, "input");
        const fs::path dest = out / str(item, "out");
        TextureRole role = TextureRole::Albedo;
        if (!parseTextureRole(str(item, "role"), role)) {
            report(false, dest.string(), "unknown role", 0.0);
            continue;
        }
        float cutoff = 0.5f;
        if (yyjson_val* c = yyjson_obj_get(item, "cutoff")) {
            const double value = yyjson_is_num(c) ? yyjson_get_num(c) : -1.0;
            if (!(value >= 0.0 && value <= 1.0)) {
                report(false, dest.string(), "\"cutoff\" must be a number from 0 to 1", 0.0);
                continue;
            }
            cutoff = float(value);
        }
        const Clock inputTime = isBuiltin(input) ? Clock::min() : fileTime(resolve(input));
        if (!stale(dest, inputTime)) {
            ++upToDate;
            continue;
        }
        std::string error;
        const auto [ok, seconds] = timed([&] {
            Image image;
            const bool loaded = isBuiltin(input) ? makeBuiltinImage(input.substr(8), image, error)
                                                 : loadImageFile(resolve(input).string(), image, error);
            return loaded && cookTexture(image, role, dest.string(), error, 2048, cutoff);
        });
        report(ok, str(item, "out"), error, seconds);
    }

    yyjson_arr_foreach(yyjson_obj_get(root, "models"), idx, max, item) {
        const std::string input = str(item, "input");
        const fs::path dest = out / str(item, "out");
        const std::string animsRel = str(item, "anims"); // optional clip library (M6 plan D4)
        const fs::path animsDest = animsRel.empty() ? fs::path() : out / animsRel;
        Clock inputTime = Clock::min();
        std::string source = input;
        if (!isBuiltin(input) && input.rfind("rig:", 0) != 0) {
            const fs::path path = resolve(input);
            if (!fs::exists(path)) {
                if (yyjson_get_bool(yyjson_obj_get(item, "optional"))) {
                    std::printf("skipped %s: %s isn't downloaded (run scripts\\fetch_assets.ps1)\n", str(item, "out"),
                                path.string().c_str());
                    ++skipped;
                } else {
                    report(false, str(item, "out"), "missing input " + path.string(), 0.0);
                }
                continue;
            }
            inputTime = newestInDirectory(path.parent_path());
            source = path.string();
        }
        if (!stale(dest, inputTime) && (animsRel.empty() || !stale(animsDest, inputTime))) {
            ++upToDate;
            continue;
        }
        std::string error;
        std::vector<std::string> materials;
        if (yyjson_val* list = yyjson_obj_get(item, "materials")) {
            size_t mi = 0, mn = 0;
            yyjson_val* path = nullptr;
            yyjson_arr_foreach(list, mi, mn, path) materials.emplace_back(yyjson_get_str(path) ? yyjson_get_str(path) : "");
        } else if (*str(item, "material")) {
            materials.emplace_back(str(item, "material"));
        }
        // "lods": false keeps one level (M5 plan D5); anything but a boolean is an error.
        bool lods = true;
        if (yyjson_val* l = yyjson_obj_get(item, "lods")) {
            if (!yyjson_is_bool(l)) {
                report(false, str(item, "out"), "\"lods\" must be true or false", 0.0);
                continue;
            }
            lods = yyjson_get_bool(l);
        }
        const auto [ok, seconds] = timed([&] {
            return cookModel(source, materials, dest.string(), o.outDir, error, lods, animsRel.empty() ? std::string() : animsDest.string());
        });
        report(ok, str(item, "out"), error, seconds);
    }

    // Maps bake last: they read the cooked models, materials and textures (MapCook.h). A map is re-baked when its source,
    // the manifest, the tool or anything cooked into models/, materials/ or textures/ is newer than its map.json.
    yyjson_arr_foreach(yyjson_obj_get(root, "maps"), idx, max, item) {
        const std::string input = str(item, "input");
        const std::string outRel = str(item, "out");
        const fs::path manifestPath = out / outRel / "map.json";
        Clock inputTime = fileTime(resolve(input));
        for (const char* dir : {"models", "materials", "textures"}) inputTime = std::max(inputTime, newestInDirectory(out / dir));
        if (!stale(manifestPath, inputTime)) {
            ++upToDate;
            continue;
        }
        std::printf("baking %s ...\n", outRel.c_str());
        std::fflush(stdout);
        std::string error;
        const auto [ok, seconds] = timed([&] { return cookMap(resolve(input).string(), outRel, o.outDir, error); });
        report(ok, outRel, error, seconds);
    }

    std::printf("fl_assetc cook: %d cooked, %d up to date, %d skipped, %d failed\n", cooked, upToDate, skipped, failures);
    return failures == 0 ? 0 : 1;
}

} // namespace tools
