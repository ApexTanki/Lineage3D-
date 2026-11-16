#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tools {

// Permutation manifest (<template>.perm.json, ARCHITECTURE.md §7.5):
// { "template": "lit", "vertex": "lit.vert", "fragment": "lit.frag",
//   "axes": [ { "name": "NORMAL_MAP" }, { "name": "OBJECT", "values": ["STATIC", "DYNAMIC"] } ],
//   "maxVariants": 256 }
// A boolean axis takes 1 key bit and defines NAME when on. An enum axis takes ceil(log2(n)) bits and
// defines NAME=<index> plus NAME_<VALUE>.
struct ManifestAxis {
    std::string name;
    std::vector<std::string> values; // empty → boolean
    uint8_t bitOffset = 0;
    uint8_t bitCount = 0;
    uint32_t valueCount() const { return values.empty() ? 2u : uint32_t(values.size()); }
};

struct Manifest {
    std::string templateName;
    std::string vertexPath;   // resolved against the manifest's directory
    std::string fragmentPath;
    std::vector<ManifestAxis> axes;
    uint32_t maxVariants = 256;
};

struct Permutation {
    uint64_t key = 0;
    std::vector<std::string> defines;
};

bool loadManifest(const std::string& path, Manifest& out, std::string& error);
std::vector<Permutation> enumeratePermutations(const Manifest& manifest);

} // namespace tools
