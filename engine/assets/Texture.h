#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace assets {

// GPU-ready pixel formats a decoded texture can have. fl_assets has no RHI dependency; the renderer maps these.
enum class PixelFormat : uint8_t { Unknown, RGBA8Unorm, RGBA8Srgb, BC1Srgb, BC3Srgb, BC3Unorm, BC4, BC5, BC6H, BC7Unorm, BC7Srgb };

enum class TextureShape : uint8_t { Tex2D, Array, Cube };

struct DecodedTexture {
    // One image: a mip level of one array layer or cube face (faces in the order +X, −X, +Y, −Y, +Z, −Z).
    struct Level {
        uint32_t offset; // into data
        uint32_t size;
        uint32_t width;
        uint32_t height;
        uint32_t mip = 0;
        uint32_t layer = 0;
    };
    PixelFormat format = PixelFormat::Unknown;
    TextureShape shape = TextureShape::Tex2D;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t layers = 1; // array layers, or 6 for a cube
    uint32_t mips = 1;
    bool hasAlpha = false;
    std::vector<Level> levels; // mip-major: every layer of mip 0, then mip 1, ...; a plain 2D texture has one per mip
    std::vector<std::byte> data;
};

// What the device can sample, decided once from rhi::Caps.
struct TranscodeTargets {
    bool bc = true;   // BC1–BC5
    bool bptc = true; // BC7 and BC6H
};

// Reads a KTX2 file (2D, 2D array or cube) and, for Basis Universal payloads, transcodes it (ARCHITECTURE.md §7.3, M1
// plan D3): ETC1S → BC1, UASTC → BC7 (BC3 for alpha without BPTC), 1/2-channel linear data → BC4/BC5, and RGBA8 only
// when the device has no BC support. Pre-compressed payloads (BC6H lightmaps and environment cubes, M3 plan D5) are
// taken as they are and need the matching device support. Thread-safe; runs on loader threads.
bool decodeKtx2(std::span<const std::byte> file, const TranscodeTargets& targets, DecodedTexture& out, std::string* error);

uint32_t pixelFormatBlockBytes(PixelFormat format); // per 4×4 block, or per pixel for RGBA8
bool pixelFormatIsBlockCompressed(PixelFormat format);

} // namespace assets
