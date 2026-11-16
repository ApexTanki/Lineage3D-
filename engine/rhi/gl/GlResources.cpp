#include "rhi/gl/GlDevice.h"

#include "core/Hash.h"
#include "core/Log.h"
#include "platform/FileSystem.h"
#include "platform/Timer.h"

#include <shaderformat/BindingModel.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>

namespace rhi::gl {
namespace {

GLenum toGlTarget(TextureType type) {
    switch (type) {
    case TextureType::Tex2D: return GL_TEXTURE_2D;
    case TextureType::Tex2DArray: return GL_TEXTURE_2D_ARRAY;
    case TextureType::Cube: return GL_TEXTURE_CUBE_MAP;
    }
    return GL_TEXTURE_2D;
}

GLenum toGlCompare(CompareOp op) {
    switch (op) {
    case CompareOp::Never: return GL_NEVER;
    case CompareOp::Less: return GL_LESS;
    case CompareOp::Equal: return GL_EQUAL;
    case CompareOp::LessOrEqual: return GL_LEQUAL;
    case CompareOp::Greater: return GL_GREATER;
    case CompareOp::NotEqual: return GL_NOTEQUAL;
    case CompareOp::GreaterOrEqual: return GL_GEQUAL;
    case CompareOp::Always: return GL_ALWAYS;
    }
    return GL_ALWAYS;
}

GLenum toGlWrap(AddressMode mode) {
    switch (mode) {
    case AddressMode::Repeat: return GL_REPEAT;
    case AddressMode::ClampToEdge: return GL_CLAMP_TO_EDGE;
    case AddressMode::MirroredRepeat: return GL_MIRRORED_REPEAT;
    }
    return GL_REPEAT;
}

std::string infoLog(GLuint object, bool program) {
    GLint length = 0;
    if (program) glGetProgramiv(object, GL_INFO_LOG_LENGTH, &length);
    else glGetShaderiv(object, GL_INFO_LOG_LENGTH, &length);
    std::string log(size_t(length > 0 ? length : 1), '\0');
    if (program) glGetProgramInfoLog(object, length, nullptr, log.data());
    else glGetShaderInfoLog(object, length, nullptr, log.data());
    return log;
}

} // namespace

BufferHandle GlDevice::createBuffer(const BufferDesc& desc) {
    GlBuffer res;
    res.size = desc.size;
    res.category = memoryCategory(desc);
    if (!m_memory.reserve(res.category, desc.size, desc.debugName)) return {}; // over its budget (M9 plan D6)
    MemoryReservation reservation(m_memory, res.category, desc.size);
    glGenBuffers(1, &res.id);
    // COPY_WRITE_BUFFER avoids disturbing the VAO's element-array binding.
    glBindBuffer(GL_COPY_WRITE_BUFFER, res.id);
    glBufferData(GL_COPY_WRITE_BUFFER, desc.size, desc.initialData,
                 desc.memory == MemoryKind::GpuOnly ? GL_STATIC_DRAW : GL_STREAM_DRAW);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    label(GL_BUFFER, res.id, desc.debugName);
    const BufferHandle h = m_buffers.create(res);
    if (!h) {
        FL_LOG_ERROR("Buffer pool exhausted");
        glDeleteBuffers(1, &res.id);
        return h;
    }
    reservation.keep();
    return h;
}

TextureHandle GlDevice::createTexture(const TextureDesc& desc) {
    const GlTextureFormat f = glTextureFormat(desc.format);
    if (!f.internalFormat) {
        FL_LOG_ERROR("Texture %s: format %s has no GL equivalent", desc.debugName, formatName(desc.format));
        return {};
    }
    GlTexture res;
    res.desc = desc;
    res.target = toGlTarget(desc.type);
    const GpuMemoryCategory category = memoryCategory(desc);
    const uint64_t memory = memoryBytes(desc);
    if (!m_memory.reserve(category, memory, desc.debugName)) return {}; // over its budget (M9 plan D6)
    MemoryReservation reservation(m_memory, category, memory);
    if (desc.samples > 1) {
        // Multisample textures are GL 3.2 core; they serve as render targets that get resolved by blits.
        if (desc.type != TextureType::Tex2D || desc.mips != 1 || desc.samples > m_caps.maxMsaaSamples) {
            FL_LOG_ERROR("Texture %s: multisampled textures must be single-level 2D with at most %u samples", desc.debugName,
                         m_caps.maxMsaaSamples);
            return {};
        }
        res.target = GL_TEXTURE_2D_MULTISAMPLE;
        glGenTextures(1, &res.id);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(res.target, res.id);
        glTexImage2DMultisample(res.target, desc.samples, f.internalFormat, GLsizei(desc.width), GLsizei(desc.height), GL_TRUE);
        glBindTexture(res.target, 0);
        m_state.texture[0] = 0;
        label(GL_TEXTURE, res.id, desc.debugName);
        const TextureHandle h = m_textures.create(res);
        if (!h) {
            FL_LOG_ERROR("Texture pool exhausted");
            glDeleteTextures(1, &res.id);
            return h;
        }
        reservation.keep();
        return h;
    }
    glGenTextures(1, &res.id);
    glActiveTexture(GL_TEXTURE0); // the bind below would otherwise clobber whatever unit replay left active
    glBindTexture(res.target, res.id);
    // Immutable storage where available (ARB_texture_storage, core in GL 4.2), else per-mip allocation.
    const bool storage = GLAD_GL_ARB_texture_storage != 0;
    if (storage) {
        if (desc.type == TextureType::Tex2DArray)
            glTexStorage3D(res.target, desc.mips, f.internalFormat, desc.width, desc.height, desc.layers);
        else glTexStorage2D(res.target, desc.mips, f.internalFormat, desc.width, desc.height);
    } else {
        // Specific compressed formats can only be allocated through the CompressedTexImage entry points.
        const FormatInfo fi = formatInfo(desc.format);
        for (uint32_t mip = 0; mip < desc.mips; ++mip) {
            const GLsizei w = GLsizei(std::max(1u, desc.width >> mip)), h = GLsizei(std::max(1u, desc.height >> mip));
            const GLsizei bytes = GLsizei(((uint32_t(w) + 3) / 4) * ((uint32_t(h) + 3) / 4) * fi.blockBytes);
            if (desc.type == TextureType::Tex2DArray) {
                if (f.compressed)
                    glCompressedTexImage3D(res.target, GLint(mip), f.internalFormat, w, h, desc.layers, 0, bytes * desc.layers, nullptr);
                else glTexImage3D(res.target, GLint(mip), GLint(f.internalFormat), w, h, desc.layers, 0, f.format, f.type, nullptr);
            } else {
                const uint32_t faces = desc.type == TextureType::Cube ? 6 : 1;
                for (uint32_t face = 0; face < faces; ++face) {
                    const GLenum target = desc.type == TextureType::Cube ? GL_TEXTURE_CUBE_MAP_POSITIVE_X + face : res.target;
                    if (f.compressed) glCompressedTexImage2D(target, GLint(mip), f.internalFormat, w, h, 0, bytes, nullptr);
                    else glTexImage2D(target, GLint(mip), GLint(f.internalFormat), w, h, 0, f.format, f.type, nullptr);
                }
            }
        }
    }
    glTexParameteri(res.target, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(res.target, GL_TEXTURE_MAX_LEVEL, GLint(desc.mips) - 1);
    glBindTexture(res.target, 0);
    m_state.texture[0] = 0; // texture unit 0's binding changed behind the cache
    label(GL_TEXTURE, res.id, desc.debugName);
    const TextureHandle h = m_textures.create(res);
    if (!h) {
        FL_LOG_ERROR("Texture pool exhausted");
        glDeleteTextures(1, &res.id);
        return h;
    }
    reservation.keep();
    return h;
}

core::Result<void> GlDevice::uploadTexture(TextureHandle handle, const TextureUpload& upload) {
    const GlTexture* t = m_textures.get(handle);
    if (!t) return core::Error::NotFound;
    const GlTextureFormat f = glTextureFormat(t->desc.format);
    const FormatInfo fi = formatInfo(t->desc.format);
    const uint32_t texels = upload.rowLength ? upload.rowLength : upload.width;
    const uint32_t rowLength = (texels + fi.blockSize - 1) / fi.blockSize * fi.blockSize; // whole blocks (see Vulkan)
    const size_t bytes = size_t(rowLength / fi.blockSize) * ((upload.height + fi.blockSize - 1) / fi.blockSize) * fi.blockBytes;
    if (upload.data.size() < bytes) return core::Error::InvalidData;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(t->target, t->id);
    m_state.texture[0] = 0;
    glPixelStorei(GL_UNPACK_ROW_LENGTH, f.compressed ? 0 : GLint(rowLength));
    const GLenum faceTarget = t->desc.type == TextureType::Cube ? GL_TEXTURE_CUBE_MAP_POSITIVE_X + upload.layer : t->target;
    if (t->desc.type == TextureType::Tex2DArray) {
        if (f.compressed)
            glCompressedTexSubImage3D(t->target, GLint(upload.mip), GLint(upload.x), GLint(upload.y), GLint(upload.layer),
                                      GLsizei(upload.width), GLsizei(upload.height), 1, f.internalFormat, GLsizei(bytes),
                                      upload.data.data());
        else
            glTexSubImage3D(t->target, GLint(upload.mip), GLint(upload.x), GLint(upload.y), GLint(upload.layer),
                            GLsizei(upload.width), GLsizei(upload.height), 1, f.format, f.type, upload.data.data());
    } else {
        if (f.compressed)
            glCompressedTexSubImage2D(faceTarget, GLint(upload.mip), GLint(upload.x), GLint(upload.y), GLsizei(upload.width),
                                      GLsizei(upload.height), f.internalFormat, GLsizei(bytes), upload.data.data());
        else
            glTexSubImage2D(faceTarget, GLint(upload.mip), GLint(upload.x), GLint(upload.y), GLsizei(upload.width),
                            GLsizei(upload.height), f.format, f.type, upload.data.data());
    }
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glBindTexture(t->target, 0);
    m_stats.uploadedBytes += bytes;
    return {};
}

SamplerHandle GlDevice::createSampler(const SamplerDesc& desc) {
    GlSampler res;
    glGenSamplers(1, &res.id);
    const bool linear = desc.filter == Filter::Linear;
    const GLenum minFilter = desc.mip == MipMode::Linear ? (linear ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_LINEAR)
                                                         : (linear ? GL_LINEAR_MIPMAP_NEAREST : GL_NEAREST_MIPMAP_NEAREST);
    glSamplerParameteri(res.id, GL_TEXTURE_MIN_FILTER, GLint(minFilter));
    glSamplerParameteri(res.id, GL_TEXTURE_MAG_FILTER, linear ? GL_LINEAR : GL_NEAREST);
    const GLint wrap = GLint(toGlWrap(desc.address));
    glSamplerParameteri(res.id, GL_TEXTURE_WRAP_S, wrap);
    glSamplerParameteri(res.id, GL_TEXTURE_WRAP_T, wrap);
    glSamplerParameteri(res.id, GL_TEXTURE_WRAP_R, wrap);
    if (desc.compare != CompareOp::Never) {
        glSamplerParameteri(res.id, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        glSamplerParameteri(res.id, GL_TEXTURE_COMPARE_FUNC, GLint(toGlCompare(desc.compare)));
    }
    const float aniso = std::min(desc.maxAnisotropy, m_caps.maxAnisotropy);
    if (aniso > 1.0f) glSamplerParameterf(res.id, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);
    return m_samplers.create(res);
}

ShaderHandle GlDevice::createShader(const ShaderDesc& desc) {
    if (desc.code.empty() || !desc.reflection) {
        FL_LOG_ERROR("Shader %s has no GLSL source for this backend", desc.debugName);
        return {};
    }
    GlShader res;
    res.stage = desc.stage;
    res.source.assign(reinterpret_cast<const char*>(desc.code.data()), desc.code.size());
    res.name = desc.debugName ? desc.debugName : "";
    res.hash = core::hash64(res.source);
    // The reflection is copied: the GL program needs it at link time to assign bindings by name.
    res.pushBlock = desc.reflection->pushBlock;
    res.uniformBlocks.assign(desc.reflection->uniformBlocks.begin(), desc.reflection->uniformBlocks.end());
    res.samplers.assign(desc.reflection->samplers.begin(), desc.reflection->samplers.end());
    res.vertexInputs.assign(desc.reflection->vertexInputs.begin(), desc.reflection->vertexInputs.end());
    // Without a binary cache the shader compiles now, so errors show where it is made; with one, a pipeline that misses
    // the cache compiles it (M9 plan D5).
    if (!m_programCache && !compileShader(res)) return {};
    return m_shaders.create(std::move(res));
}

// ---------------------------------------------------------------- program binaries (M9 plan D5)

namespace {

// A program binary file: this header, then the driver's binary. A driver update changes the key (the driver's strings
// are part of it), and the driver may still refuse a binary, which then compiles and is rewritten. The header repeats
// what the key was made from, so a binary is never taken on the strength of a 64-bit hash alone.
struct ProgramBinaryHeader {
    char magic[4];         // "FLGP"
    uint32_t version;      // kProgramBinaryVersion
    uint64_t key;          // the program's key, also its file name
    uint64_t vertexHash;   // FNV-1a 64 of the vertex shader's GLSL
    uint64_t fragmentHash; // FNV-1a 64 of the fragment shader's GLSL
    uint64_t driverHash;   // FNV-1a 64 of the driver's vendor, renderer and version strings
    uint64_t dataHash;     // FNV-1a 64 of the binary
    uint32_t format;       // the driver's binary format
    uint32_t length;       // bytes of binary that follow
};
constexpr uint32_t kProgramBinaryVersion = 2; // 2: the source and driver hashes

} // namespace

void GlDevice::initProgramCache() {
    m_programCache = false;
    if (!m_caps.programBinaryCache || !m_desc.cacheDirectory) return;
    GLint count = 0;
    glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &count);
    if (count <= 0) return; // a driver that lists no formats can't load what it writes
    m_binaryFormats.resize(size_t(count));
    glGetIntegerv(GL_PROGRAM_BINARY_FORMATS, m_binaryFormats.data());
    m_programCacheDir = std::string(m_desc.cacheDirectory) + "gl/";
    if (!platform::createDirectories(m_programCacheDir)) {
        FL_LOG_WARN("Program binary cache off: %s can't be created", m_programCacheDir.c_str());
        return;
    }
    auto text = [](GLenum name) {
        const GLubyte* s = glGetString(name);
        return std::string(s ? reinterpret_cast<const char*>(s) : "");
    };
    m_driverKey = core::hash64(text(GL_VENDOR) + "|" + text(GL_RENDERER) + "|" + text(GL_VERSION));
    m_programCache = true;
}

bool GlDevice::compileShader(GlShader& s) {
    if (s.id) return true;
    if (s.compileFailed) return false;
    s.id = glCreateShader(s.stage == shaderformat::Stage::Vertex ? GL_VERTEX_SHADER : GL_FRAGMENT_SHADER);
    const GLchar* source = s.source.c_str();
    const GLint length = GLint(s.source.size());
    glShaderSource(s.id, 1, &source, &length);
    glCompileShader(s.id);
    GLint ok = GL_FALSE;
    glGetShaderiv(s.id, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        FL_LOG_ERROR("GL shader %s failed to compile:\n%s", s.name.c_str(), infoLog(s.id, false).c_str());
        glDeleteShader(s.id);
        s.id = 0;
        s.compileFailed = true;
        return false;
    }
    label(GL_SHADER, s.id, s.name.c_str());
    return true;
}

uint64_t GlDevice::programKey(const GlShader& vs, const GlShader& fs) const {
    const uint64_t parts[2] = {vs.hash, fs.hash};
    return core::hash64(std::as_bytes(std::span(parts)), m_driverKey);
}

std::string GlDevice::programBinaryPath(uint64_t key) const {
    char name[32];
    std::snprintf(name, sizeof(name), "%016llx.bin", static_cast<unsigned long long>(key));
    return m_programCacheDir + name;
}

GLuint GlDevice::loadProgramBinary(uint64_t key, const GlShader& vs, const GlShader& fs) {
    auto file = platform::readFile(programBinaryPath(key));
    if (!file.ok()) return 0;
    const std::vector<std::byte>& bytes = file.value();
    ProgramBinaryHeader h{};
    if (bytes.size() < sizeof(h)) return 0;
    std::memcpy(&h, bytes.data(), sizeof(h));
    const std::span<const std::byte> binary(bytes.data() + sizeof(h), bytes.size() - sizeof(h));
    if (std::memcmp(h.magic, "FLGP", 4) != 0 || h.version != kProgramBinaryVersion || h.key != key || h.vertexHash != vs.hash ||
        h.fragmentHash != fs.hash || h.driverHash != m_driverKey || h.length != binary.size() || core::hash64(binary) != h.dataHash ||
        std::find(m_binaryFormats.begin(), m_binaryFormats.end(), GLint(h.format)) == m_binaryFormats.end())
        return 0;
    const GLuint program = glCreateProgram();
    glProgramBinary(program, GLenum(h.format), binary.data(), GLsizei(h.length));
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

void GlDevice::saveProgramBinary(GLuint program, uint64_t key, const GlShader& vs, const GlShader& fs) {
    GLint length = 0;
    glGetProgramiv(program, GL_PROGRAM_BINARY_LENGTH, &length);
    if (length <= 0) return;
    constexpr size_t kHeader = sizeof(ProgramBinaryHeader);
    std::vector<std::byte> file(kHeader + size_t(length));
    GLsizei written = 0;
    GLenum format = 0;
    glGetProgramBinary(program, length, &written, &format, file.data() + kHeader);
    if (written <= 0) return;
    file.resize(kHeader + size_t(written));
    ProgramBinaryHeader h{};
    std::memcpy(h.magic, "FLGP", 4);
    h.version = kProgramBinaryVersion;
    h.key = key;
    h.vertexHash = vs.hash;
    h.fragmentHash = fs.hash;
    h.driverHash = m_driverKey;
    h.dataHash = core::hash64(std::span<const std::byte>(file.data() + kHeader, size_t(written)));
    h.format = uint32_t(format);
    h.length = uint32_t(written);
    std::memcpy(file.data(), &h, kHeader);
    if (platform::writeFileAtomic(programBinaryPath(key), file).ok()) ++m_stats.programBinaryWrites;
    else FL_LOG_WARN("Could not write a program binary to %s", m_programCacheDir.c_str());
}

// A linked program with its bindings applied, from the binary cache or compiled and linked (M9 plan D5); 0 on failure.
GLuint GlDevice::makeProgram(GlShader& vs, GlShader& fs, const PipelineDesc& desc) {
    const uint64_t key = m_programCache ? programKey(vs, fs) : 0;
    GLuint program = m_programCache ? loadProgramBinary(key, vs, fs) : 0;
    if (program) {
        ++m_stats.programBinaryHits;
    } else {
        if (!compileShader(vs) || !compileShader(fs)) return 0;
        program = glCreateProgram();
        if (m_programCache) glProgramParameteri(program, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
        glAttachShader(program, vs.id);
        glAttachShader(program, fs.id);
        glLinkProgram(program);
        GLint ok = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            FL_LOG_ERROR("GL program %s failed to link:\n%s", desc.debugName, infoLog(program, true).c_str());
            glDeleteProgram(program);
            return 0;
        }
        glDetachShader(program, vs.id);
        glDetachShader(program, fs.id);
        if (m_programCache) saveProgramBinary(program, key, vs, fs);
    }

    // GLSL 330 can't declare bindings, so assign them by name from the reflection (ARCHITECTURE.md §4.6). A program loaded
    // from a binary starts with default uniforms and block bindings, so this runs for it too.
    auto bindBlock = [&](const char* name, uint8_t point) {
        const GLuint index = glGetUniformBlockIndex(program, name);
        if (index != GL_INVALID_INDEX) glUniformBlockBinding(program, index, point);
    };
    auto fail = [&] {
        glUseProgram(0);
        m_state.program = 0;
        glDeleteProgram(program);
        return GLuint(0);
    };
    glUseProgram(program);
    m_state.program = program;
    for (const GlShader* s : {&vs, &fs}) {
        for (const shaderformat::UniformBlock& b : s->uniformBlocks) {
            const uint8_t point = shaderformat::glUniformBufferPoint(b.set, b.binding);
            if (point == shaderformat::kInvalidSlot) {
                FL_LOG_ERROR("Pipeline %s: block %s (set %u, binding %u) has no GL slot", desc.debugName, b.name, b.set, b.binding);
                return fail();
            }
            bindBlock(b.name, point);
        }
        if (s->pushBlock.size > 0) bindBlock(s->pushBlock.name, shaderformat::kPerDrawUboPoint);
        for (const shaderformat::Sampler& smp : s->samplers) {
            const uint8_t unit = shaderformat::glTextureUnit(smp.set, smp.binding);
            if (unit == shaderformat::kInvalidSlot) {
                FL_LOG_ERROR("Pipeline %s: sampler %s (set %u, binding %u) has no GL slot", desc.debugName, smp.name, smp.set, smp.binding);
                return fail();
            }
            const GLint location = glGetUniformLocation(program, smp.name);
            if (location >= 0) glUniform1i(location, unit);
        }
    }
    // Leave no program current: the next BindPipeline makes its own current. A just-created program left current was
    // validated against later, unrelated state changes (uploads, other passes' textures), which NVIDIA reported (M4).
    glUseProgram(0);
    m_state.program = 0;
    label(GL_PROGRAM, program, desc.debugName);
    return program;
}

PipelineHandle GlDevice::createPipeline(const PipelineDesc& desc) {
    GlShader* vs = m_shaders.get(desc.vertex);
    GlShader* fs = m_shaders.get(desc.fragment);
    if (!vs || !fs) {
        FL_LOG_ERROR("Pipeline %s: invalid shader handle", desc.debugName);
        return {};
    }
    // Integer attributes must be fed with glVertexAttribIPointer; catch format/shader mismatches here (P4).
    for (uint32_t i = 0; i < desc.vertexLayout.attributeCount; ++i) {
        const VertexAttribute& a = desc.vertexLayout.attributes[i];
        for (const shaderformat::VertexInput& in : vs->vertexInputs) {
            if (in.location != a.location) continue;
            const bool shaderInteger = in.baseType != shaderformat::BaseType::Float;
            if (shaderInteger != glVertexFormat(a.format).integer) {
                FL_LOG_ERROR("Pipeline %s: attribute %u format %s doesn't match the shader input type", desc.debugName,
                             a.location, formatName(a.format));
                return {};
            }
        }
    }
    const uint64_t start = platform::nowTicks();
    GlPipeline res;
    res.desc = desc;
    res.program = makeProgram(*vs, *fs, desc);
    if (!res.program) return {};
    const PipelineHandle handle = m_pipelines.create(res);
    if (!handle) {
        FL_LOG_ERROR("Pipeline pool exhausted (%s)", desc.debugName);
        glDeleteProgram(res.program);
        return {};
    }
    ++m_stats.pipelinesCreated; // M9 plan D3
    m_stats.pipelineCreateMs += platform::ticksToMilliseconds(platform::nowTicks() - start);
    return handle;
}

// Hot reload (M9 plan D9): both shaders compile from their new sources, and every pipeline made with either links a new
// program; handles stay valid. If anything fails (an edit that changed the shaders' interface), nothing changes.
bool GlDevice::replaceShaders(ShaderHandle vertex, const ShaderDesc& vertexDesc, ShaderHandle fragment, const ShaderDesc& fragmentDesc) {
    GlShader* shaders[2] = {m_shaders.get(vertex), m_shaders.get(fragment)};
    const ShaderDesc* descs[2] = {&vertexDesc, &fragmentDesc};
    GlShader fresh[2];
    auto undo = [&] {
        for (GlShader& f : fresh)
            if (f.id) glDeleteShader(f.id);
        return false;
    };
    for (uint32_t i = 0; i < 2; ++i) {
        const ShaderDesc& d = *descs[i];
        if (!shaders[i] || d.code.empty() || !d.reflection || d.stage != shaders[i]->stage) return undo();
        GlShader& f = fresh[i];
        f.stage = d.stage;
        f.source.assign(reinterpret_cast<const char*>(d.code.data()), d.code.size());
        f.name = d.debugName ? d.debugName : "";
        f.hash = core::hash64(f.source);
        f.pushBlock = d.reflection->pushBlock;
        f.uniformBlocks.assign(d.reflection->uniformBlocks.begin(), d.reflection->uniformBlocks.end());
        f.samplers.assign(d.reflection->samplers.begin(), d.reflection->samplers.end());
        f.vertexInputs.assign(d.reflection->vertexInputs.begin(), d.reflection->vertexInputs.end());
        if (!compileShader(f)) return undo();
    }
    std::vector<std::pair<GlPipeline*, GLuint>> relinked;
    bool ok = true;
    m_pipelines.forEach([&](PipelineHandle, GlPipeline& p) {
        if (!ok || (p.desc.vertex != vertex && p.desc.fragment != fragment)) return;
        GlShader* vs = p.desc.vertex == vertex ? &fresh[0] : m_shaders.get(p.desc.vertex);
        GlShader* fs = p.desc.fragment == fragment ? &fresh[1] : m_shaders.get(p.desc.fragment);
        const GLuint program = vs && fs ? makeProgram(*vs, *fs, p.desc) : 0;
        if (program) relinked.emplace_back(&p, program);
        else ok = false;
    });
    if (!ok) {
        for (auto& [p, program] : relinked) glDeleteProgram(program);
        return undo();
    }
    for (auto& [p, program] : relinked) {
        defer(Deferred::Kind::Program, p->program); // frames in flight still use it
        if (m_pipeline == p) m_pipeline = nullptr;   // the next BindPipeline makes the new program current
        p->program = program;
    }
    for (uint32_t i = 0; i < 2; ++i) {
        if (shaders[i]->id) defer(Deferred::Kind::Shader, shaders[i]->id);
        *shaders[i] = std::move(fresh[i]);
    }
    m_stats.pipelinesRebuilt += uint32_t(relinked.size());
    return true;
}

ResourceSetLayoutHandle GlDevice::createResourceSetLayout(const ResourceSetLayoutDesc& desc) {
    if (desc.count > kMaxSetBindings) {
        FL_LOG_ERROR("Set %u layout has %u bindings; the limit is %u", desc.set, desc.count, kMaxSetBindings);
        return {};
    }
    for (uint32_t i = 0; i < desc.count; ++i) {
        const ResourceBinding& b = desc.bindings[i];
        for (uint32_t j = 0; j < i; ++j)
            if (desc.bindings[j].binding == b.binding) {
                FL_LOG_ERROR("Set %u layout declares binding %u twice", desc.set, b.binding);
                return {};
            }
        const uint8_t slot = b.kind == BindingKind::Texture ? shaderformat::glTextureUnit(desc.set, b.binding)
                                                            : shaderformat::glUniformBufferPoint(desc.set, b.binding);
        if (slot == shaderformat::kInvalidSlot) {
            FL_LOG_ERROR("Set %u binding %u doesn't fit the binding model (ARCHITECTURE.md §4.6)", desc.set, b.binding);
            return {};
        }
    }
    return m_setLayouts.create({desc});
}

ResourceSetHandle GlDevice::createResourceSet(const ResourceSetDesc& desc) {
    const GlSetLayout* layout = m_setLayouts.get(desc.layout);
    if (!layout) return {};
    if (desc.writes.size() > kMaxSetBindings) {
        FL_LOG_ERROR("Resource set %s has %zu writes; the limit is %u", desc.debugName, desc.writes.size(), kMaxSetBindings);
        return {};
    }
    GlResourceSet res;
    for (const ResourceWrite& w : desc.writes) {
        const ResourceBinding* binding = nullptr;
        for (uint32_t i = 0; i < layout->desc.count; ++i)
            if (layout->desc.bindings[i].binding == w.binding) binding = &layout->desc.bindings[i];
        if (!binding) {
            FL_LOG_ERROR("Resource set %s: binding %u isn't in its layout", desc.debugName, w.binding);
            return {};
        }
        if (binding->kind == BindingKind::Texture) {
            const GlTexture* t = m_textures.get(w.texture);
            const GlSampler* s = m_samplers.get(w.sampler);
            if (!t || !s) {
                FL_LOG_ERROR("Resource set %s: invalid texture or sampler at binding %u", desc.debugName, w.binding);
                return {};
            }
            // A multisample texture is read with texelFetch and has no sampler state. A sampler object on its unit makes
            // NVIDIA apply a mipmapped min filter to it and treat it as incomplete (texelFetch returned 0; M7 soft
            // particles), so its unit gets no sampler object.
            const GLuint sampler = t->target == GL_TEXTURE_2D_MULTISAMPLE ? 0u : s->id;
            res.textures[res.textureCount++] = {shaderformat::glTextureUnit(layout->desc.set, w.binding), t->target, t->id, sampler};
        } else {
            const GlBuffer* b = m_buffers.get(w.buffer.buffer);
            if (!b) {
                FL_LOG_ERROR("Resource set %s: invalid buffer at binding %u", desc.debugName, w.binding);
                return {};
            }
            res.ubos[res.uboCount++] = {shaderformat::glUniformBufferPoint(layout->desc.set, w.binding), b->id,
                                        w.buffer.offset, w.buffer.size ? w.buffer.size : b->size - w.buffer.offset,
                                        binding->kind == BindingKind::UniformBufferDynamic};
        }
    }
    return m_resourceSets.create(res);
}

QueryPoolHandle GlDevice::createTimestampPool(uint32_t count) {
    GlQueryPool res;
    res.count = count;
    for (uint32_t s = 0; s < kFramesInFlight; ++s) {
        res.ids[s].resize(count);
        res.written[s].assign(count, 0);
        glGenQueries(GLsizei(count), res.ids[s].data());
    }
    return m_queryPools.create(std::move(res));
}

void GlDevice::destroy(BufferHandle h) {
    if (GlBuffer* r = m_buffers.get(h); r && !r->ownedByRing) {
        m_memory.release(r->category, r->size);
        defer(Deferred::Kind::Buffer, r->id);
        m_buffers.destroy(h);
    }
}

void GlDevice::destroy(TextureHandle h) {
    if (GlTexture* r = m_textures.get(h)) {
        m_memory.release(memoryCategory(r->desc), memoryBytes(r->desc));
        defer(Deferred::Kind::Texture, r->id);
        m_textures.destroy(h);
    }
}

void GlDevice::destroy(SamplerHandle h) {
    if (GlSampler* r = m_samplers.get(h)) {
        defer(Deferred::Kind::Sampler, r->id);
        m_samplers.destroy(h);
    }
}

void GlDevice::destroy(ShaderHandle h) {
    if (GlShader* r = m_shaders.get(h)) {
        if (r->id) defer(Deferred::Kind::Shader, r->id); // 0: never compiled (every pipeline came from a binary)
        m_shaders.destroy(h);
    }
}

void GlDevice::destroy(PipelineHandle h) {
    if (GlPipeline* r = m_pipelines.get(h)) {
        if (m_pipeline == r) m_pipeline = nullptr;
        defer(Deferred::Kind::Program, r->program);
        m_pipelines.destroy(h);
    }
}

void GlDevice::destroy(ResourceSetLayoutHandle h) {
    if (m_setLayouts.get(h)) m_setLayouts.destroy(h); // invalid handles are ignored, as on Vulkan
}

void GlDevice::destroy(ResourceSetHandle h) {
    if (m_resourceSets.get(h)) m_resourceSets.destroy(h);
}

void GlDevice::destroy(QueryPoolHandle h) {
    if (GlQueryPool* r = m_queryPools.get(h)) {
        for (auto& ids : r->ids)
            for (GLuint id : ids) defer(Deferred::Kind::Query, id);
        m_queryPools.destroy(h);
    }
}

} // namespace rhi::gl
