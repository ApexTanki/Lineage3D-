#pragma once

#include <shaderformat/ShaderPackage.h>

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace tools {

struct VariantInput {
    std::string vertexPath;
    std::string fragmentPath;
    std::vector<std::string> defines; // "NAME" or "NAME=VALUE"
    std::vector<std::string> includeDirs;
};

struct VariantOutput {
    std::vector<uint32_t> spirv[shaderformat::kStageCount];
    std::string glsl330[shaderformat::kStageCount];
    shaderformat::PushBlock pushBlock{};
    std::vector<shaderformat::UniformBlock> uniformBlocks;
    std::vector<shaderformat::Sampler> samplers;
    std::vector<shaderformat::VertexInput> vertexInputs;
};

// Compiles one permutation: glslang to SPIR-V 1.3 (Vulkan 1.1), SPIRV-Cross to GLSL 330 with
// the options pinned in ARCHITECTURE.md §4.10, reflection, the forbidden-feature checks, and a final
// re-compile of the generated GLSL with glslang.
class ShaderCompiler {
public:
    ShaderCompiler();
    ~ShaderCompiler();
    ShaderCompiler(const ShaderCompiler&) = delete;
    ShaderCompiler& operator=(const ShaderCompiler&) = delete;

    bool compile(const VariantInput& input, VariantOutput& output, std::string& error, std::set<std::string>& dependencies);
};

} // namespace tools
