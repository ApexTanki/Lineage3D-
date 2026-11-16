// fl_shaderc — build-time shader compiler (ARCHITECTURE.md §4.10, §5.7).
//   fl_shaderc --manifest <x.perm.json> --include <dir> [--include <dir>...] --out <x.fsp>
//              [--depfile <x.fsp.d> --depfile-target <path as the build system names the output>]

#include "Manifest.h"
#include "PackageWriter.h"
#include "ShaderCompiler.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Makefile-style depfiles escape spaces with a backslash; the project path itself contains one.
std::string escapeDepPath(const std::string& path) {
    std::string out;
    for (char c : path) {
        if (c == ' ') out += "\\ ";
        else if (c == '\\') out += '/';
        else out += c;
    }
    return out;
}

bool writeBytes(const std::string& path, const void* data, size_t size) {
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file.write(static_cast<const char*>(data), std::streamsize(size));
    return bool(file);
}

} // namespace

int main(int argc, char** argv) {
    std::string manifestPath, outPath, depfilePath, depTarget;
    std::vector<std::string> includeDirs;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (arg == "--manifest") manifestPath = next();
        else if (arg == "--include") includeDirs.push_back(next());
        else if (arg == "--out") outPath = next();
        else if (arg == "--depfile") depfilePath = next();
        else if (arg == "--depfile-target") depTarget = next();
        else {
            std::fprintf(stderr, "fl_shaderc: unknown argument %s\n", arg.c_str());
            return 2;
        }
    }
    if (manifestPath.empty() || outPath.empty()) {
        std::fprintf(stderr, "usage: fl_shaderc --manifest <x.perm.json> --include <dir> --out <x.fsp> "
                             "[--depfile <file> --depfile-target <path>]\n");
        return 2;
    }

    tools::Manifest manifest;
    std::string error;
    if (!tools::loadManifest(manifestPath, manifest, error)) {
        std::fprintf(stderr, "fl_shaderc: %s\n", error.c_str());
        return 1;
    }
    const std::vector<tools::Permutation> permutations = tools::enumeratePermutations(manifest);
    if (permutations.size() > manifest.maxVariants) {
        std::fprintf(stderr, "fl_shaderc: %s has %zu permutations, more than maxVariants %u\n",
                     manifest.templateName.c_str(), permutations.size(), manifest.maxVariants);
        return 1;
    }

    std::set<std::string> dependencies;
    dependencies.insert(fs::weakly_canonical(manifestPath).generic_string());
    tools::ShaderCompiler compiler;
    std::vector<tools::CompiledPermutation> compiled;
    compiled.reserve(permutations.size());
    for (const tools::Permutation& p : permutations) {
        tools::VariantInput input{manifest.vertexPath, manifest.fragmentPath, p.defines, includeDirs};
        tools::CompiledPermutation c;
        c.key = p.key;
        if (!compiler.compile(input, c.output, error, dependencies)) {
            std::string defs;
            for (const std::string& d : p.defines) defs += " " + d;
            std::fprintf(stderr, "fl_shaderc: %s (permutation key 0x%llx:%s) failed:\n%s\n", manifest.templateName.c_str(),
                         (unsigned long long)p.key, defs.empty() ? " none" : defs.c_str(), error.c_str());
            return 1;
        }
        compiled.push_back(std::move(c));
    }

    const std::vector<std::byte> package = tools::writePackage(manifest, std::move(compiled));
    if (!writeBytes(outPath, package.data(), package.size())) {
        std::fprintf(stderr, "fl_shaderc: cannot write %s\n", outPath.c_str());
        return 1;
    }
    if (!depfilePath.empty()) {
        std::string dep = escapeDepPath(depTarget.empty() ? outPath : depTarget) + ":";
        for (const std::string& d : dependencies) dep += " \\\n  " + escapeDepPath(d);
        dep += "\n";
        if (!writeBytes(depfilePath, dep.data(), dep.size())) {
            std::fprintf(stderr, "fl_shaderc: cannot write %s\n", depfilePath.c_str());
            return 1;
        }
    }
    std::printf("fl_shaderc: %s -> %zu permutation(s), %zu bytes\n", manifest.templateName.c_str(), permutations.size(),
                package.size());
    return 0;
}
