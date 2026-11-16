#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace tools {

struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgba; // 8-bit RGBA, rows top to bottom
    bool hasTranslucency() const;
};

bool loadImageFile(const std::string& path, Image& out, std::string& error);
bool loadImageMemory(std::span<const std::byte> bytes, Image& out, std::string& error);

// Resamples with sRGB-correct filtering of RGB (`srgb`, alpha linear and alpha-weighted) or plain linear filtering of
// all four channels.
Image resizeImage(const Image& src, uint32_t width, uint32_t height, bool srgb);

float srgbToLinear(uint8_t value);
uint8_t linearToSrgb(float value);

// Texture roles (ARCHITECTURE.md §7.3, M2 plan D8):
//   albedo        sRGB RGB, Basis ETC1S                         → BC1
//   albedo_alpha  sRGB RGBA, Basis UASTC + Zstandard            → BC7 (BC3 without BPTC)
//   albedo_cutout as albedo_alpha, with coverage-preserving mips for the alpha-test cutoff
//   normal        input RGB = XYZ; mips averaged + renormalized; stored as 2-channel linear UASTC (normal-map mode)
//                                                                → BC5, Z reconstructed in the shader
//   specgloss     sRGB rgb specular color + linear alpha gloss, no alpha weighting, UASTC + Zstandard → BC7/BC3 sRGB
//   emission      sRGB RGB, Basis ETC1S                         → BC1
//   raw_srgb      uncompressed RGBA8 sRGB (test patterns that must survive bit-exactly)
enum class TextureRole : uint8_t { Albedo, AlbedoAlpha, AlbedoCutout, Normal, SpecGloss, Emission, RawSrgb };
bool parseTextureRole(const std::string& name, TextureRole& out);
const char* textureRoleName(TextureRole role);

// Offline texture cooking (M1 plan D3): downscale to at most maxSize, build the full mip chain the role needs, then
// encode into KTX2. `alphaCutoff` is the cutoff whose coverage albedo_cutout preserves.
bool cookTexture(const Image& image, TextureRole role, const std::string& outPath, std::string& error, uint32_t maxSize = 2048,
                 float alphaCutoff = 0.5f);

// Fraction of texels whose alpha × scale reaches the cutoff (exposed for tests).
float alphaCoverage(const Image& image, float alphaCutoff, float scale = 1.0f);

// Generic KTX2 writing (2D, 2D arrays and cubes; M3 plan D5). One image is mip `mip` of array layer or cube face
// `layer` (faces +X, −X, +Y, −Y, +Z, −Z), tightly packed: 8-bit texels for the Basis encodings, or blocks for a
// pre-compressed vkFormat (BC6H) written Raw. `zstd` adds Zstandard supercompression (not for ETC1S, which has its own).
// The file is replaced atomically.
struct KtxImage {
    uint32_t mip = 0;
    uint32_t layer = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> bytes;
};
enum class KtxEncoding : uint8_t { Etc1s, Uastc, Raw };
enum class KtxShape : uint8_t { Tex2D, Array, Cube };
inline constexpr uint32_t kVkR8Unorm = 9;
inline constexpr uint32_t kVkR8G8B8A8Unorm = 37;
inline constexpr uint32_t kVkBc6hUfloat = 143;
bool writeKtx2Images(uint32_t vkFormat, KtxShape shape, uint32_t width, uint32_t height, uint32_t mips, uint32_t layers,
                     const std::vector<KtxImage>& images, KtxEncoding encoding, bool normalMap, bool zstd, const std::string& outPath,
                     std::string& error);

} // namespace tools
