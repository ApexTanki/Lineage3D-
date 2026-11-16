#pragma once

#include "Manifest.h"
#include "ShaderCompiler.h"

#include <cstddef>
#include <vector>

namespace tools {

struct CompiledPermutation {
    uint64_t key = 0;
    VariantOutput output;
};

// Serializes compiled permutations into the .fsp layout read by shaderformat::PackageReader.
std::vector<std::byte> writePackage(const Manifest& manifest, std::vector<CompiledPermutation> permutations);

} // namespace tools
