#include "rhi/Types.h"

namespace rhi {

FormatInfo formatInfo(Format format) {
    switch (format) {
    case Format::Undefined: return {0, 1, false, false};
    case Format::R8Unorm: return {1, 1, false, false};
    case Format::RG8Unorm: return {2, 1, false, false};
    case Format::RGBA8Unorm: return {4, 1, false, false};
    case Format::RGBA8Srgb: return {4, 1, false, true};
    case Format::BGRA8Unorm: return {4, 1, false, false};
    case Format::BGRA8Srgb: return {4, 1, false, true};
    case Format::R16F: return {2, 1, false, false};
    case Format::RG16F: return {4, 1, false, false};
    case Format::RGBA16F: return {8, 1, false, false};
    case Format::R11G11B10F: return {4, 1, false, false};
    case Format::R32F: return {4, 1, false, false};
    case Format::D16: return {2, 1, true, false};
    case Format::D24S8: return {4, 1, true, false};
    case Format::D32F: return {4, 1, true, false};
    case Format::BC1Srgb: return {8, 4, false, true};
    case Format::BC3Srgb: return {16, 4, false, true};
    case Format::BC3Unorm: return {16, 4, false, false};
    case Format::BC4: return {8, 4, false, false};
    case Format::BC5: return {16, 4, false, false};
    case Format::BC6HUfloat: return {16, 4, false, false};
    case Format::BC7Unorm: return {16, 4, false, false};
    case Format::BC7Srgb: return {16, 4, false, true};
    case Format::RG32F: return {8, 1, false, false};
    case Format::RGB32F: return {12, 1, false, false};
    case Format::RGBA32F: return {16, 1, false, false};
    case Format::RGBA16Sint: return {8, 1, false, false};
    case Format::RGBA8Snorm: return {4, 1, false, false};
    case Format::RGBA8Uint: return {4, 1, false, false};
    case Format::RG16Unorm: return {4, 1, false, false};
    case Format::RGBA32Uint: return {16, 1, false, false};
    case Format::Count: break;
    }
    return {0, 1, false, false};
}

const char* formatName(Format format) {
    static constexpr const char* kNames[] = {
        "Undefined", "R8Unorm", "RG8Unorm", "RGBA8Unorm", "RGBA8Srgb", "BGRA8Unorm", "BGRA8Srgb", "R16F", "RG16F",
        "RGBA16F", "R11G11B10F", "R32F", "D16", "D24S8", "D32F", "BC1Srgb", "BC3Srgb", "BC3Unorm", "BC4", "BC5", "BC6HUfloat",
        "BC7Unorm", "BC7Srgb", "RG32F", "RGB32F", "RGBA32F", "RGBA16Sint", "RGBA8Snorm", "RGBA8Uint", "RG16Unorm",
        "RGBA32Uint"};
    static_assert(sizeof(kNames) / sizeof(kNames[0]) == kFormatCount, "format name table out of sync");
    const auto i = static_cast<uint32_t>(format);
    return i < kFormatCount ? kNames[i] : "?";
}

} // namespace rhi
