#include "PackageWriter.h"

#include <algorithm>
#include <cstring>

namespace tools {
namespace {

using namespace shaderformat;

size_t align16(size_t v) { return (v + 15) & ~size_t(15); }

template <typename T>
void put(std::vector<std::byte>& out, size_t offset, const T& value) {
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

void copyName(char (&dst)[kNameLength], const std::string& src) {
    std::memset(dst, 0, sizeof(dst));
    std::memcpy(dst, src.data(), std::min(src.size(), size_t(kNameLength - 1)));
}

std::vector<std::byte> variantBlob(const VariantOutput& v) {
    VariantHeader h{};
    std::vector<std::byte> blob(align16(sizeof(VariantHeader)));
    auto append = [&](const void* data, size_t bytes) -> uint32_t {
        const size_t offset = blob.size();
        blob.resize(align16(offset + bytes));
        if (bytes) std::memcpy(blob.data() + offset, data, bytes);
        return uint32_t(offset);
    };
    for (uint32_t s = 0; s < kStageCount; ++s) {
        const auto& spv = v.spirv[s];
        h.codeOffset[uint32_t(Target::SpirV)][s] = append(spv.data(), spv.size() * 4);
        h.codeSize[uint32_t(Target::SpirV)][s] = uint32_t(spv.size() * 4);
        h.codeOffset[uint32_t(Target::Glsl330)][s] = append(v.glsl330[s].data(), v.glsl330[s].size());
        h.codeSize[uint32_t(Target::Glsl330)][s] = uint32_t(v.glsl330[s].size());
    }
    h.pushBlock = v.pushBlock;
    h.uniformBlockOffset = append(v.uniformBlocks.data(), v.uniformBlocks.size() * sizeof(UniformBlock));
    h.uniformBlockCount = uint16_t(v.uniformBlocks.size());
    h.samplerOffset = append(v.samplers.data(), v.samplers.size() * sizeof(Sampler));
    h.samplerCount = uint16_t(v.samplers.size());
    h.vertexInputOffset = append(v.vertexInputs.data(), v.vertexInputs.size() * sizeof(VertexInput));
    h.vertexInputCount = uint16_t(v.vertexInputs.size());
    std::memcpy(blob.data(), &h, sizeof(h));
    return blob;
}

} // namespace

std::vector<std::byte> writePackage(const Manifest& manifest, std::vector<CompiledPermutation> permutations) {
    std::sort(permutations.begin(), permutations.end(), [](const auto& a, const auto& b) { return a.key < b.key; });

    const size_t axesOffset = sizeof(FileHeader);
    const size_t entriesOffset = axesOffset + manifest.axes.size() * sizeof(Axis);
    size_t dataOffset = align16(entriesOffset + permutations.size() * sizeof(VariantEntry));

    std::vector<std::vector<std::byte>> blobs;
    blobs.reserve(permutations.size());
    for (const CompiledPermutation& p : permutations) blobs.push_back(variantBlob(p.output));
    size_t total = dataOffset;
    for (const auto& b : blobs) total += b.size();

    std::vector<std::byte> out(total);
    FileHeader header{};
    std::memcpy(header.magic, kMagic, 4);
    header.version = kVersion;
    header.axisCount = uint16_t(manifest.axes.size());
    header.variantCount = uint32_t(permutations.size());
    header.fileSize = uint32_t(total);
    copyName(header.templateName, manifest.templateName);
    put(out, 0, header);

    for (size_t i = 0; i < manifest.axes.size(); ++i) {
        Axis a{};
        copyName(a.name, manifest.axes[i].name);
        a.bitOffset = manifest.axes[i].bitOffset;
        a.bitCount = manifest.axes[i].bitCount;
        a.valueCount = uint8_t(manifest.axes[i].valueCount());
        put(out, axesOffset + i * sizeof(Axis), a);
    }
    for (size_t i = 0; i < permutations.size(); ++i) {
        const VariantEntry e{permutations[i].key, uint32_t(dataOffset), uint32_t(blobs[i].size())};
        put(out, entriesOffset + i * sizeof(VariantEntry), e);
        std::memcpy(out.data() + dataOffset, blobs[i].data(), blobs[i].size());
        dataOffset += blobs[i].size();
    }
    return out;
}

} // namespace tools
