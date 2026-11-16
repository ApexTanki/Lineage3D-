#include "ShaderCompiler.h"

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <shaderformat/BindingModel.h>
#include <spirv_glsl.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

namespace tools {
namespace fs = std::filesystem;
using shaderformat::Stage;

namespace {

bool readText(const std::string& path, std::string& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::ostringstream ss;
    ss << file.rdbuf();
    out = ss.str();
    return true;
}

void copyName(char (&dst)[shaderformat::kNameLength], const std::string& src) {
    std::memset(dst, 0, sizeof(dst));
    std::memcpy(dst, src.data(), std::min(src.size(), size_t(shaderformat::kNameLength - 1)));
}

std::string normalized(const fs::path& p) {
    std::error_code ec;
    fs::path abs = fs::weakly_canonical(p, ec);
    return (ec ? p : abs).generic_string();
}

// Resolves #include against the including file's directory, then the include directories, and records
// every file read so the build can write a depfile.
class Includer final : public glslang::TShader::Includer {
public:
    Includer(const std::vector<std::string>& dirs, std::set<std::string>& deps) : m_dirs(dirs), m_deps(deps) {}

    IncludeResult* includeLocal(const char* header, const char* includer, size_t) override {
        if (includer && *includer)
            if (IncludeResult* r = tryOpen(fs::path(includer).parent_path() / header)) return r;
        return includeSystem(header, includer, 0);
    }
    IncludeResult* includeSystem(const char* header, const char*, size_t) override {
        for (const std::string& dir : m_dirs)
            if (IncludeResult* r = tryOpen(fs::path(dir) / header)) return r;
        return nullptr;
    }
    void releaseInclude(IncludeResult* result) override {
        if (!result) return;
        delete static_cast<std::string*>(result->userData);
        delete result;
    }

private:
    IncludeResult* tryOpen(const fs::path& path) {
        auto* text = new std::string();
        if (!readText(path.string(), *text)) {
            delete text;
            return nullptr;
        }
        const std::string name = normalized(path);
        m_deps.insert(name);
        return new IncludeResult(name, text->data(), text->size(), text);
    }
    const std::vector<std::string>& m_dirs;
    std::set<std::string>& m_deps;
};

EShLanguage toLanguage(Stage s) {
    return s == Stage::Vertex ? EShLangVertex : EShLangFragment;
}

shaderformat::SamplerType samplerTypeOf(const spirv_cross::SPIRType& type) {
    const auto& img = type.image;
    if (img.ms) return shaderformat::SamplerType::Sampler2DMS; // only 2D, non-arrayed ones get here (checked by the caller)
    if (img.dim == spv::DimCube) return shaderformat::SamplerType::SamplerCube;
    if (img.depth) return img.arrayed ? shaderformat::SamplerType::Sampler2DArrayShadow : shaderformat::SamplerType::Sampler2DShadow;
    return img.arrayed ? shaderformat::SamplerType::Sampler2DArray : shaderformat::SamplerType::Sampler2D;
}

std::string crossCompile(const std::vector<uint32_t>& spirv, std::string& error) {
    // SPIRV-Cross reports problems by throwing; tools may use exceptions (ARCHITECTURE.md A11).
    try {
        spirv_cross::CompilerGLSL glsl(spirv);
        spirv_cross::CompilerGLSL::Options opts = glsl.get_common_options();
        opts.version = 330;
        opts.es = false;
        opts.enable_420pack_extension = false;           // no layout(binding): the GL backend binds by name
        opts.emit_push_constant_as_uniform_buffer = true; // per-draw block → UBO on GL
        opts.vertex.fixup_clipspace = false;              // clip space is handled in ClipSpace only (C7)
        opts.vertex.flip_vert_y = false;
        opts.vertex.support_nonzero_base_instance = false;
        glsl.set_common_options(opts);
        return glsl.compile();
    } catch (const std::exception& e) {
        error = std::string("SPIRV-Cross: ") + e.what();
        return {};
    }
}

// Re-compiles generated GLSL with glslang to catch SPIRV-Cross output that the target GLSL rejects.
bool validateGlsl(const std::string (&text)[shaderformat::kStageCount], std::string& error) {
    std::unique_ptr<glslang::TShader> shaders[2];
    glslang::TProgram program;
    for (uint32_t s = 0; s < shaderformat::kStageCount; ++s) {
        const EShLanguage lang = toLanguage(Stage(s));
        shaders[s] = std::make_unique<glslang::TShader>(lang);
        const char* src = text[s].c_str();
        shaders[s]->setStrings(&src, 1);
        shaders[s]->setEnvInput(glslang::EShSourceGlsl, lang, glslang::EShClientNone, 0);
        if (!shaders[s]->parse(GetDefaultResources(), 330, ECoreProfile, false, false, EShMsgDefault)) {
            error = std::string("GLSL 330 validation failed:\n") + shaders[s]->getInfoLog() +
                    "\n--- generated source ---\n" + text[s];
            return false;
        }
        program.addShader(shaders[s].get());
    }
    if (!program.link(EShMsgDefault)) {
        error = std::string("GLSL 330 link validation failed:\n") + program.getInfoLog();
        return false;
    }
    return true;
}

bool reflectStage(const std::vector<uint32_t>& spirv, Stage stage, VariantOutput& out, std::string& error,
                  spirv_cross::ShaderResources* outResources) {
    try {
        spirv_cross::Compiler comp(spirv);
        const auto active = comp.get_active_interface_variables();
        spirv_cross::ShaderResources res = comp.get_shader_resources(active);
        const uint8_t bit = stage == Stage::Vertex ? shaderformat::kVertexBit : shaderformat::kFragmentBit;

        // Forbidden on the GL 3.3 floor (ARCHITECTURE.md §4.10).
        if (!res.storage_buffers.empty() || !res.storage_images.empty()) {
            error = "SSBOs and image load/store are not available on the GL floor";
            return false;
        }
        if (!res.separate_images.empty() || !res.separate_samplers.empty()) {
            error = "use combined image samplers (sampler2D ...), not separate images and samplers";
            return false;
        }
        if (!comp.get_specialization_constants().empty()) {
            error = "specialization constants are not allowed; use permutation #defines";
            return false;
        }
        for (const auto& b : res.builtin_inputs)
            if (b.builtin == spv::BuiltInBaseInstance || b.builtin == spv::BuiltInBaseVertex || b.builtin == spv::BuiltInDrawIndex) {
                error = "gl_BaseInstance / gl_BaseVertex / gl_DrawID are not available on the GL floor";
                return false;
            }

        for (const auto& ub : res.uniform_buffers) {
            const uint32_t set = comp.get_decoration(ub.id, spv::DecorationDescriptorSet);
            const uint32_t binding = comp.get_decoration(ub.id, spv::DecorationBinding);
            if (shaderformat::glUniformBufferPoint(set, binding) == shaderformat::kInvalidSlot) {
                error = "uniform block " + ub.name + " uses set " + std::to_string(set) + " binding " +
                        std::to_string(binding) + ", which has no GL slot (ARCHITECTURE.md §4.6)";
                return false;
            }
            auto it = std::find_if(out.uniformBlocks.begin(), out.uniformBlocks.end(),
                                   [&](const auto& b) { return b.set == set && b.binding == binding; });
            if (it == out.uniformBlocks.end()) {
                shaderformat::UniformBlock b{};
                copyName(b.name, comp.get_name(ub.base_type_id));
                b.set = uint8_t(set);
                b.binding = uint8_t(binding);
                b.size = uint32_t(comp.get_declared_struct_size(comp.get_type(ub.base_type_id)));
                b.stageMask = bit;
                out.uniformBlocks.push_back(b);
            } else {
                it->stageMask |= bit;
            }
        }
        for (const auto& si : res.sampled_images) {
            const auto& image = comp.get_type(si.type_id).image;
            if (image.ms && (image.dim != spv::Dim2D || image.arrayed || image.depth)) {
                error = "sampler " + si.name + " is a multisampled sampler other than sampler2DMS, which the reflection has no type for";
                return false;
            }
            const uint32_t set = comp.get_decoration(si.id, spv::DecorationDescriptorSet);
            const uint32_t binding = comp.get_decoration(si.id, spv::DecorationBinding);
            if (shaderformat::glTextureUnit(set, binding) == shaderformat::kInvalidSlot) {
                error = "sampler " + si.name + " uses set " + std::to_string(set) + " binding " + std::to_string(binding) +
                        ", which has no GL texture unit (ARCHITECTURE.md §4.6)";
                return false;
            }
            auto it = std::find_if(out.samplers.begin(), out.samplers.end(),
                                   [&](const auto& s) { return s.set == set && s.binding == binding; });
            if (it == out.samplers.end()) {
                shaderformat::Sampler s{};
                copyName(s.name, si.name);
                s.set = uint8_t(set);
                s.binding = uint8_t(binding);
                s.type = samplerTypeOf(comp.get_type(si.type_id));
                s.stageMask = bit;
                out.samplers.push_back(s);
            } else {
                it->stageMask |= bit;
            }
        }
        if (!res.push_constant_buffers.empty()) {
            const auto& pc = res.push_constant_buffers[0];
            const uint32_t size = uint32_t(comp.get_declared_struct_size(comp.get_type(pc.base_type_id)));
            if (size > shaderformat::kMaxPerDrawBytes) {
                error = "per-draw block is " + std::to_string(size) + " bytes; the limit is 128";
                return false;
            }
            copyName(out.pushBlock.name, comp.get_name(pc.base_type_id));
            out.pushBlock.size = std::max(out.pushBlock.size, size);
            out.pushBlock.stageMask |= bit;
        }
        if (stage == Stage::Vertex) {
            for (const auto& in : res.stage_inputs) {
                const auto& type = comp.get_type(in.type_id);
                shaderformat::VertexInput v{};
                copyName(v.name, in.name);
                v.location = uint8_t(comp.get_decoration(in.id, spv::DecorationLocation));
                v.baseType = type.basetype == spirv_cross::SPIRType::Int    ? shaderformat::BaseType::Int
                             : type.basetype == spirv_cross::SPIRType::UInt ? shaderformat::BaseType::Uint
                                                                            : shaderformat::BaseType::Float;
                v.components = uint8_t(type.vecsize);
                out.vertexInputs.push_back(v);
            }
        }
        if (outResources) *outResources = res;
        return true;
    } catch (const std::exception& e) {
        error = std::string("SPIRV-Cross reflection: ") + e.what();
        return false;
    }
}

// GLSL 330 matches varyings by name, so VS outputs and FS inputs must agree exactly (§4.10).
bool checkVaryings(const std::vector<uint32_t>& vs, const std::vector<uint32_t>& fsSpirv, std::string& error) {
    try {
        spirv_cross::Compiler v(vs), f(fsSpirv);
        const auto outs = v.get_shader_resources().stage_outputs;
        for (const auto& in : f.get_shader_resources().stage_inputs) {
            auto it = std::find_if(outs.begin(), outs.end(), [&](const auto& o) { return o.name == in.name; });
            if (it == outs.end()) {
                error = "fragment input '" + in.name + "' has no vertex output with the same name";
                return false;
            }
            const auto& a = v.get_type(it->type_id);
            const auto& b = f.get_type(in.type_id);
            if (a.basetype != b.basetype || a.vecsize != b.vecsize || a.columns != b.columns) {
                error = "varying '" + in.name + "' has different types in the vertex and fragment stages";
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("SPIRV-Cross: ") + e.what();
        return false;
    }
}

} // namespace

ShaderCompiler::ShaderCompiler() {
    glslang::InitializeProcess();
}

ShaderCompiler::~ShaderCompiler() {
    glslang::FinalizeProcess();
}

bool ShaderCompiler::compile(const VariantInput& input, VariantOutput& out, std::string& error,
                             std::set<std::string>& dependencies) {
    const std::string paths[2] = {input.vertexPath, input.fragmentPath};
    std::string sources[2];
    std::string preambles[2];
    for (uint32_t s = 0; s < 2; ++s) {
        if (!readText(paths[s], sources[s])) {
            error = "cannot read " + paths[s];
            return false;
        }
        dependencies.insert(normalized(paths[s]));
        std::string& p = preambles[s];
        p = "#extension GL_GOOGLE_include_directive : require\n";
        p += s == 0 ? "#define FL_STAGE_VERTEX 1\n" : "#define FL_STAGE_FRAGMENT 1\n";
        for (const std::string& d : input.defines) {
            const size_t eq = d.find('=');
            p += "#define " + (eq == std::string::npos ? d + " 1" : d.substr(0, eq) + " " + d.substr(eq + 1)) + "\n";
        }
    }

    Includer includer(input.includeDirs, dependencies);
    std::unique_ptr<glslang::TShader> shaders[2];
    glslang::TProgram program;
    const EShMessages messages = EShMessages(EShMsgSpvRules | EShMsgVulkanRules);
    for (uint32_t s = 0; s < 2; ++s) {
        const EShLanguage lang = toLanguage(Stage(s));
        shaders[s] = std::make_unique<glslang::TShader>(lang);
        const char* src = sources[s].c_str();
        const int length = int(sources[s].size());
        const std::string name = normalized(paths[s]);
        const char* names[] = {name.c_str()};
        shaders[s]->setStringsWithLengthsAndNames(&src, &length, names, 1);
        shaders[s]->setPreamble(preambles[s].c_str());
        shaders[s]->setEnvInput(glslang::EShSourceGlsl, lang, glslang::EShClientVulkan, 100);
        shaders[s]->setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
        shaders[s]->setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);
        if (!shaders[s]->parse(GetDefaultResources(), 450, ECoreProfile, false, false, messages, includer)) {
            error = shaders[s]->getInfoLog();
            return false;
        }
        program.addShader(shaders[s].get());
    }
    if (!program.link(messages)) {
        error = program.getInfoLog();
        return false;
    }

    glslang::SpvOptions spvOptions;
    spvOptions.generateDebugInfo = false; // OpNames are kept regardless: GL binds blocks and samplers by name
    spvOptions.disableOptimizer = true;
    spvOptions.validate = true;
    for (uint32_t s = 0; s < 2; ++s) {
        out.spirv[s].clear();
        glslang::GlslangToSpv(*program.getIntermediate(toLanguage(Stage(s))), out.spirv[s], &spvOptions);
    }

    out.pushBlock = {};
    out.uniformBlocks.clear();
    out.samplers.clear();
    out.vertexInputs.clear();
    for (uint32_t s = 0; s < 2; ++s)
        if (!reflectStage(out.spirv[s], Stage(s), out, error, nullptr)) return false;
    if (!checkVaryings(out.spirv[0], out.spirv[1], error)) return false;

    for (uint32_t s = 0; s < 2; ++s) {
        out.glsl330[s] = crossCompile(out.spirv[s], error);
        if (out.glsl330[s].empty()) return false;
    }
    return validateGlsl(out.glsl330, error);
}

} // namespace tools
