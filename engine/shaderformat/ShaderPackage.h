#pragma once

// .fsp shader package: every permutation of one shader template, compiled offline into SPIR-V and
// GLSL 330, plus reflection (ARCHITECTURE.md §7.5). The file is read into memory once
// and accessed through views; the only fix-up is turning offsets into pointers.
//
// Layout (little-endian, sections 16-byte aligned):
//   FileHeader | Axis[axisCount] | VariantEntry[variantCount] (sorted by key) | variant blobs
//   variant blob: VariantHeader | code blobs | UniformBlock[] | Sampler[] | VertexInput[]

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>

namespace shaderformat {

enum class Target : uint8_t { SpirV, Glsl330, Count };
enum class Stage : uint8_t { Vertex, Fragment, Count };
enum class BaseType : uint8_t { Float, Int, Uint };
// Sampler2DMS (M7 plan D8): a multisampled 2D texture read with texelFetch (the soft particles' depth copy).
enum class SamplerType : uint8_t { Sampler2D, Sampler2DArray, SamplerCube, Sampler2DShadow, Sampler2DArrayShadow, Sampler2DMS };

inline constexpr uint32_t kTargetCount = uint32_t(Target::Count);
inline constexpr uint32_t kStageCount = uint32_t(Stage::Count);
inline constexpr uint32_t kNameLength = 48;
inline constexpr char kMagic[4] = {'F', 'S', 'P', 'K'};
inline constexpr uint16_t kVersion = 2; // 2: ESSL 300 target removed (ARCHITECTURE.md DEV-7)

enum StageBits : uint8_t { kVertexBit = 1, kFragmentBit = 2 };

struct UniformBlock {
    char name[kNameLength]; // block type name, as GLSL declares it (used by glGetUniformBlockIndex)
    uint8_t set;
    uint8_t binding;
    uint8_t stageMask;
    uint8_t pad;
    uint32_t size;
};

struct Sampler {
    char name[kNameLength];
    uint8_t set;
    uint8_t binding;
    uint8_t stageMask;
    SamplerType type;
};

struct VertexInput {
    char name[kNameLength];
    uint8_t location;
    BaseType baseType;
    uint8_t components;
    uint8_t pad;
};

struct PushBlock {
    char name[kNameLength]; // empty when the variant has no per-draw block
    uint32_t size;
    uint8_t stageMask;
    uint8_t pad[3];
};

struct FileHeader {
    char magic[4];
    uint16_t version;
    uint16_t axisCount;
    uint32_t variantCount;
    uint32_t fileSize;
    char templateName[kNameLength];
};

// One permutation axis: a boolean feature (valueCount 2) or an enum (valueCount n), packed into the
// 64-bit variant key at bitOffset.
struct Axis {
    char name[kNameLength];
    uint8_t bitOffset;
    uint8_t bitCount;
    uint8_t valueCount;
    uint8_t pad[5];
};

struct VariantEntry {
    uint64_t key;
    uint32_t offset; // from the start of the file
    uint32_t size;
};

struct VariantHeader {
    uint32_t codeOffset[kTargetCount][kStageCount]; // from the start of the variant blob
    uint32_t codeSize[kTargetCount][kStageCount];
    PushBlock pushBlock;
    uint32_t uniformBlockOffset;
    uint32_t samplerOffset;
    uint32_t vertexInputOffset;
    uint16_t uniformBlockCount;
    uint16_t samplerCount;
    uint16_t vertexInputCount;
    uint16_t pad[3];
};

struct Reflection {
    PushBlock pushBlock{};
    std::span<const UniformBlock> uniformBlocks;
    std::span<const Sampler> samplers;
    std::span<const VertexInput> vertexInputs;
};

struct VariantView {
    uint64_t key = 0;
    std::span<const std::byte> code[kTargetCount][kStageCount];
    Reflection reflection;
};

// Whether two builds of a variant bind and take the same things (hot reload, M9 plan D9: new code must fit the pipelines
// made for the old): the per-draw block, every uniform block and sampler (by set and binding: name, stages, size or
// type) and every vertex input (by location: name, type, components). Declaration order doesn't matter.
inline bool sameInterface(const Reflection& a, const Reflection& b) {
    auto sameName = [](const char* x, const char* y) { return std::strncmp(x, y, kNameLength) == 0; };
    if (a.pushBlock.size != b.pushBlock.size || a.pushBlock.stageMask != b.pushBlock.stageMask ||
        !sameName(a.pushBlock.name, b.pushBlock.name) || a.uniformBlocks.size() != b.uniformBlocks.size() ||
        a.samplers.size() != b.samplers.size() || a.vertexInputs.size() != b.vertexInputs.size())
        return false;
    for (const UniformBlock& x : a.uniformBlocks) {
        bool found = false;
        for (const UniformBlock& y : b.uniformBlocks)
            found = found || (x.set == y.set && x.binding == y.binding && x.stageMask == y.stageMask && x.size == y.size &&
                              sameName(x.name, y.name));
        if (!found) return false;
    }
    for (const Sampler& x : a.samplers) {
        bool found = false;
        for (const Sampler& y : b.samplers)
            found = found || (x.set == y.set && x.binding == y.binding && x.stageMask == y.stageMask && x.type == y.type &&
                              sameName(x.name, y.name));
        if (!found) return false;
    }
    for (const VertexInput& x : a.vertexInputs) {
        bool found = false;
        for (const VertexInput& y : b.vertexInputs)
            found = found || (x.location == y.location && x.baseType == y.baseType && x.components == y.components && sameName(x.name, y.name));
        if (!found) return false;
    }
    return true;
}

inline std::string nameOf(const char (&name)[kNameLength]) {
    return std::string(name, strnlen(name, kNameLength));
}

// Validating reader. Every offset is bounds-checked, so a truncated or corrupt file yields an error
// instead of a crash.
class PackageReader {
public:
    bool open(std::span<const std::byte> file, std::string* error) {
        m_file = file;
        if (file.size() < sizeof(FileHeader)) return fail(error, "file smaller than its header");
        std::memcpy(&m_header, file.data(), sizeof(FileHeader));
        if (std::memcmp(m_header.magic, kMagic, 4) != 0) return fail(error, "bad magic");
        if (m_header.version != kVersion) return fail(error, "unsupported package version");
        if (m_header.fileSize != file.size()) return fail(error, "size mismatch (truncated file?)");
        if (!terminated(m_header.templateName)) return fail(error, "unterminated template name");
        m_axisOffset = sizeof(FileHeader);
        m_entryOffset = m_axisOffset + m_header.axisCount * sizeof(Axis);
        const size_t tableEnd = m_entryOffset + size_t(m_header.variantCount) * sizeof(VariantEntry);
        if (tableEnd > file.size()) return fail(error, "variant table out of bounds");
        for (const Axis& a : axes())
            if (!terminated(a.name)) return fail(error, "unterminated axis name");
        for (uint32_t i = 0; i < m_header.variantCount; ++i) {
            const VariantEntry e = entry(i);
            if (size_t(e.offset) + e.size > file.size() || e.size < sizeof(VariantHeader))
                return fail(error, "variant blob out of bounds");
            if (i > 0 && entry(i - 1).key >= e.key) return fail(error, "variant keys are not sorted");
            if (e.offset % 16 != 0) return fail(error, "misaligned variant blob");
            VariantHeader vh;
            std::memcpy(&vh, file.data() + e.offset, sizeof(vh));
            for (uint32_t s = 0; s < kStageCount; ++s)
                if (vh.codeOffset[uint32_t(Target::SpirV)][s] % 4 != 0 || vh.codeSize[uint32_t(Target::SpirV)][s] % 4 != 0)
                    return fail(error, "misaligned SPIR-V");
            if (vh.uniformBlockOffset % 4 != 0 || vh.samplerOffset % 4 != 0 || vh.vertexInputOffset % 4 != 0)
                return fail(error, "misaligned reflection");
            if (!terminated(vh.pushBlock.name)) return fail(error, "unterminated name");
            for (uint32_t t = 0; t < kTargetCount; ++t)
                for (uint32_t s = 0; s < kStageCount; ++s)
                    if (size_t(vh.codeOffset[t][s]) + vh.codeSize[t][s] > e.size)
                        return fail(error, "code blob out of bounds");
            if (vh.uniformBlockOffset + size_t(vh.uniformBlockCount) * sizeof(UniformBlock) > e.size ||
                vh.samplerOffset + size_t(vh.samplerCount) * sizeof(Sampler) > e.size ||
                vh.vertexInputOffset + size_t(vh.vertexInputCount) * sizeof(VertexInput) > e.size)
                return fail(error, "reflection out of bounds");
            // Names go straight to glGetUniformBlockIndex / glGetUniformLocation, so they must be terminated.
            const VariantView v = variant(i);
            for (const auto& b : v.reflection.uniformBlocks)
                if (!terminated(b.name)) return fail(error, "unterminated name");
            for (const auto& s : v.reflection.samplers)
                if (!terminated(s.name)) return fail(error, "unterminated name");
            for (const auto& in : v.reflection.vertexInputs)
                if (!terminated(in.name)) return fail(error, "unterminated name");
        }
        return true;
    }

    const FileHeader& header() const { return m_header; }
    std::string templateName() const { return nameOf(m_header.templateName); }
    uint32_t variantCount() const { return m_header.variantCount; }

    std::span<const Axis> axes() const {
        return {reinterpret_cast<const Axis*>(m_file.data() + m_axisOffset), m_header.axisCount};
    }

    VariantView variant(uint32_t index) const {
        const VariantEntry e = entry(index);
        const std::byte* blob = m_file.data() + e.offset;
        const auto* vh = reinterpret_cast<const VariantHeader*>(blob);
        VariantView v;
        v.key = e.key;
        for (uint32_t t = 0; t < kTargetCount; ++t)
            for (uint32_t s = 0; s < kStageCount; ++s)
                v.code[t][s] = {blob + vh->codeOffset[t][s], vh->codeSize[t][s]};
        v.reflection.pushBlock = vh->pushBlock;
        v.reflection.uniformBlocks = {reinterpret_cast<const UniformBlock*>(blob + vh->uniformBlockOffset),
                                      vh->uniformBlockCount};
        v.reflection.samplers = {reinterpret_cast<const Sampler*>(blob + vh->samplerOffset), vh->samplerCount};
        v.reflection.vertexInputs = {reinterpret_cast<const VertexInput*>(blob + vh->vertexInputOffset),
                                     vh->vertexInputCount};
        return v;
    }

    // Binary search; entries are sorted by key.
    std::optional<VariantView> find(uint64_t key) const {
        uint32_t lo = 0, hi = m_header.variantCount;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            const uint64_t k = entry(mid).key;
            if (k == key) return variant(mid);
            if (k < key) lo = mid + 1;
            else hi = mid;
        }
        return std::nullopt;
    }

private:
    VariantEntry entry(uint32_t index) const {
        VariantEntry e;
        std::memcpy(&e, m_file.data() + m_entryOffset + index * sizeof(VariantEntry), sizeof(e));
        return e;
    }
    static bool terminated(const char (&name)[kNameLength]) { return std::memchr(name, 0, kNameLength) != nullptr; }
    static bool fail(std::string* error, const char* message) {
        if (error) *error = message;
        return false;
    }

    std::span<const std::byte> m_file;
    FileHeader m_header{};
    size_t m_axisOffset = 0;
    size_t m_entryOffset = 0;
};

} // namespace shaderformat
