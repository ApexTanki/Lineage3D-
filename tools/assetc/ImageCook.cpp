#include "ImageCook.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <ktx.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <thread>

namespace tools {
namespace {

bool fromStb(unsigned char* pixels, int w, int h, Image& out, std::string& error) {
    if (!pixels) {
        error = std::string("image decode failed: ") + stbi_failure_reason();
        return false;
    }
    out.width = uint32_t(w);
    out.height = uint32_t(h);
    out.rgba.assign(pixels, pixels + size_t(w) * size_t(h) * 4);
    stbi_image_free(pixels);
    return true;
}

// One KTX2 mip level, tightly packed.
struct Level {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> bytes;
};

using Encoding = KtxEncoding;

constexpr uint32_t kVkR8G8Unorm = 16;
constexpr uint32_t kVkR8G8B8Srgb = 29;
constexpr uint32_t kVkR8G8B8A8Srgb = 43;

Image resizedSrgb(const Image& src, uint32_t w, uint32_t h, stbir_pixel_layout layout) {
    Image dst;
    dst.width = w;
    dst.height = h;
    dst.rgba.resize(size_t(w) * h * 4);
    // RGB is filtered in linear light; alpha is linear. STBIR_RGBA weights colors by alpha (coverage); STBIR_RGBA_PM
    // doesn't, for textures whose alpha is data (gloss).
    stbir_resize_uint8_srgb(src.rgba.data(), int(src.width), int(src.height), 0, dst.rgba.data(), int(w), int(h), 0, layout);
    return dst;
}

std::vector<Image> srgbChain(Image base, stbir_pixel_layout layout) {
    std::vector<Image> levels;
    levels.push_back(std::move(base));
    while (levels.back().width > 1 || levels.back().height > 1) {
        const Image& prev = levels.back();
        Image next = resizedSrgb(prev, std::max(1u, prev.width / 2), std::max(1u, prev.height / 2), layout);
        levels.push_back(std::move(next));
    }
    return levels;
}

// Alpha-tested foliage and fences thin out with distance because averaging lowers alpha below the cutoff. Each mip's
// alpha is scaled so the fraction of texels passing the cutoff matches mip 0 (coverage preservation).
void preserveCoverage(std::vector<Image>& levels, float cutoff) {
    const float target = alphaCoverage(levels[0], cutoff);
    for (size_t l = 1; l < levels.size(); ++l) {
        float lo = 0.0f, hi = 4.0f;
        for (int iteration = 0; iteration < 20; ++iteration) {
            const float mid = (lo + hi) * 0.5f;
            if (alphaCoverage(levels[l], cutoff, mid) < target) lo = mid;
            else hi = mid;
        }
        std::vector<uint8_t>& rgba = levels[l].rgba;
        for (size_t i = 3; i < rgba.size(); i += 4) rgba[i] = uint8_t(std::min(255.0f, std::round(float(rgba[i]) * hi)));
    }
}

// Normal maps: XYZ decoded to floats, averaged per mip in linear space and renormalized, then stored as X, Y bytes.
std::vector<Level> normalChain(const Image& image, uint32_t width, uint32_t height) {
    std::vector<float> xyz(size_t(image.width) * image.height * 3);
    for (size_t p = 0, n = size_t(image.width) * image.height; p < n; ++p)
        for (size_t c = 0; c < 3; ++c) xyz[p * 3 + c] = float(image.rgba[p * 4 + c]) / 255.0f * 2.0f - 1.0f;
    uint32_t w = image.width, h = image.height;
    std::vector<Level> levels;
    std::vector<float> next;
    while (true) {
        if (w != width || h != height) {
            next.resize(size_t(width) * height * 3);
            stbir_resize_float_linear(xyz.data(), int(w), int(h), 0, next.data(), int(width), int(height), 0, STBIR_RGB);
            xyz.swap(next);
            w = width;
            h = height;
        }
        Level level;
        level.width = w;
        level.height = h;
        level.bytes.resize(size_t(w) * h * 2);
        for (size_t p = 0, n = size_t(w) * h; p < n; ++p) {
            float* v = &xyz[p * 3];
            const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            const float inv = len > 1e-6f ? 1.0f / len : 0.0f;
            if (len <= 1e-6f) v[2] = 1.0f; // a degenerate average points straight out
            for (int c = 0; c < 3; ++c) v[c] = len > 1e-6f ? v[c] * inv : v[c];
            level.bytes[p * 2 + 0] = uint8_t(std::clamp(std::round((v[0] * 0.5f + 0.5f) * 255.0f), 0.0f, 255.0f));
            level.bytes[p * 2 + 1] = uint8_t(std::clamp(std::round((v[1] * 0.5f + 0.5f) * 255.0f), 0.0f, 255.0f));
        }
        levels.push_back(std::move(level));
        if (w == 1 && h == 1) break;
        width = std::max(1u, w / 2);
        height = std::max(1u, h / 2);
    }
    return levels;
}

// Plain 2D textures with a full mip chain.
bool writeKtx2(const std::vector<Level>& levels, uint32_t vkFormat, Encoding encoding, bool normalMap, const std::string& outPath,
               std::string& error) {
    std::vector<KtxImage> images;
    images.reserve(levels.size());
    for (uint32_t level = 0; level < levels.size(); ++level) images.push_back({level, 0, levels[level].width, levels[level].height, levels[level].bytes});
    return writeKtx2Images(vkFormat, KtxShape::Tex2D, levels[0].width, levels[0].height, uint32_t(levels.size()), 1, images, encoding,
                           normalMap, encoding == Encoding::Uastc, outPath, error);
}

} // namespace

bool writeKtx2Images(uint32_t vkFormat, KtxShape shape, uint32_t width, uint32_t height, uint32_t mips, uint32_t layers,
                     const std::vector<KtxImage>& images, KtxEncoding encoding, bool normalMap, bool zstd, const std::string& outPath,
                     std::string& error) {
    ktxTextureCreateInfo ci{};
    ci.vkFormat = vkFormat;
    ci.baseWidth = width;
    ci.baseHeight = height;
    ci.baseDepth = 1;
    ci.numDimensions = 2;
    ci.numLevels = mips;
    ci.numLayers = shape == KtxShape::Array ? layers : 1;
    ci.numFaces = shape == KtxShape::Cube ? 6 : 1;
    ci.isArray = shape == KtxShape::Array ? KTX_TRUE : KTX_FALSE;
    ci.generateMipmaps = KTX_FALSE;
    ktxTexture2* texture = nullptr;
    if (ktxTexture2_Create(&ci, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture) != KTX_SUCCESS) {
        error = "ktxTexture2_Create failed";
        return false;
    }
    struct Guard {
        ktxTexture2* t;
        ~Guard() { ktxTexture_Destroy(ktxTexture(t)); }
    } guard{texture};
    if (images.size() != size_t(mips) * (shape == KtxShape::Cube ? 6 : shape == KtxShape::Array ? layers : 1)) {
        error = "KTX2 image count doesn't match the layout";
        return false;
    }
    for (const KtxImage& image : images) {
        const uint32_t layer = shape == KtxShape::Array ? image.layer : 0, face = shape == KtxShape::Cube ? image.layer : 0;
        if (ktxTexture_SetImageFromMemory(ktxTexture(texture), image.mip, layer, face, image.bytes.data(), image.bytes.size()) !=
            KTX_SUCCESS) {
            error = "ktxTexture_SetImageFromMemory failed";
            return false;
        }
    }

    if (encoding != KtxEncoding::Raw) {
        ktxBasisParams params{};
        params.structSize = sizeof(params);
        params.uastc = encoding == KtxEncoding::Uastc ? KTX_TRUE : KTX_FALSE;
        params.threadCount = std::max(1u, std::thread::hardware_concurrency());
        params.compressionLevel = KTX_ETC1S_DEFAULT_COMPRESSION_LEVEL;
        params.qualityLevel = 192;
        params.uastcFlags = KTX_PACK_UASTC_LEVEL_DEFAULT;
        params.normalMap = normalMap ? KTX_TRUE : KTX_FALSE; // 2-channel input is swizzled to "rrrg" (→ BC5 RG)
        if (ktxTexture2_CompressBasisEx(texture, &params) != KTX_SUCCESS) {
            error = "Basis Universal encoding failed";
            return false;
        }
    }
    if (zstd && encoding != KtxEncoding::Etc1s && ktxTexture2_DeflateZstd(texture, 18) != KTX_SUCCESS) {
        error = "Zstandard supercompression failed";
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(outPath).parent_path(), ec);
    const std::string temp = outPath + ".tmp";
    if (ktxTexture_WriteToNamedFile(ktxTexture(texture), temp.c_str()) != KTX_SUCCESS) {
        error = "cannot write " + outPath;
        return false;
    }
    std::filesystem::rename(temp, outPath, ec); // replace atomically so a failed cook never leaves half a file
    if (ec) {
        error = "cannot replace " + outPath + ": " + ec.message();
        return false;
    }
    return true;
}

bool Image::hasTranslucency() const {
    for (size_t i = 3; i < rgba.size(); i += 4)
        if (rgba[i] != 255) return true;
    return false;
}

// The decode runs before w and h are read: function arguments have no evaluation order, so the call must not also
// pass them by value.
bool loadImageFile(const std::string& path, Image& out, std::string& error) {
    int w = 0, h = 0, n = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &n, 4);
    return fromStb(pixels, w, h, out, error);
}

bool loadImageMemory(std::span<const std::byte> bytes, Image& out, std::string& error) {
    int w = 0, h = 0, n = 0;
    unsigned char* pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()), int(bytes.size()), &w, &h, &n, 4);
    return fromStb(pixels, w, h, out, error);
}

Image resizeImage(const Image& src, uint32_t width, uint32_t height, bool srgb) {
    if (srgb) return resizedSrgb(src, width, height, STBIR_RGBA);
    Image dst;
    dst.width = width;
    dst.height = height;
    dst.rgba.resize(size_t(width) * height * 4);
    stbir_resize_uint8_linear(src.rgba.data(), int(src.width), int(src.height), 0, dst.rgba.data(), int(width), int(height), 0,
                              STBIR_4CHANNEL);
    return dst;
}

float srgbToLinear(uint8_t value) {
    const float c = float(value) / 255.0f;
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

uint8_t linearToSrgb(float value) {
    const float c = std::clamp(value, 0.0f, 1.0f);
    const float s = c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
    return uint8_t(std::clamp(std::round(s * 255.0f), 0.0f, 255.0f));
}

bool parseTextureRole(const std::string& name, TextureRole& out) {
    static const std::pair<const char*, TextureRole> kRoles[] = {
        {"albedo", TextureRole::Albedo},       {"albedo_alpha", TextureRole::AlbedoAlpha}, {"albedo_cutout", TextureRole::AlbedoCutout},
        {"normal", TextureRole::Normal},       {"specgloss", TextureRole::SpecGloss},      {"emission", TextureRole::Emission},
        {"raw_srgb", TextureRole::RawSrgb}};
    for (const auto& [n, role] : kRoles)
        if (name == n) {
            out = role;
            return true;
        }
    return false;
}

const char* textureRoleName(TextureRole role) {
    static constexpr const char* kNames[] = {"albedo", "albedo_alpha", "albedo_cutout", "normal", "specgloss", "emission", "raw_srgb"};
    return kNames[uint32_t(role)];
}

float alphaCoverage(const Image& image, float alphaCutoff, float scale) {
    const size_t texels = image.rgba.size() / 4;
    if (texels == 0) return 0.0f;
    const float threshold = alphaCutoff * 255.0f;
    size_t covered = 0;
    for (size_t i = 3; i < image.rgba.size(); i += 4) covered += float(image.rgba[i]) * scale >= threshold ? 1u : 0u;
    return float(covered) / float(texels);
}

bool cookTexture(const Image& image, TextureRole role, const std::string& outPath, std::string& error, uint32_t maxSize,
                 float alphaCutoff) {
    if (image.width == 0 || image.height == 0 || image.rgba.size() != size_t(image.width) * image.height * 4) {
        error = "empty image";
        return false;
    }
    // Level 0 is downscaled so the larger side fits maxSize.
    uint32_t width = image.width, height = image.height;
    if (const uint32_t larger = std::max(width, height); larger > maxSize) {
        const double s = double(maxSize) / double(larger);
        width = std::max(1u, uint32_t(width * s + 0.5));
        height = std::max(1u, uint32_t(height * s + 0.5));
    }
    if (role == TextureRole::Normal) return writeKtx2(normalChain(image, width, height), kVkR8G8Unorm, Encoding::Uastc, true, outPath, error);

    // Opaque color maps drop alpha so ETC1S encodes (and BC1 holds) RGB only. They are filtered as opaque too: leftover
    // alpha in the source (glTF says to ignore it for opaque materials) would otherwise weight texels away in the mips.
    const bool opaque = role == TextureRole::Albedo || role == TextureRole::Emission;
    Image opaqueSource;
    const Image* source = &image;
    if (opaque && image.hasTranslucency()) {
        opaqueSource = image;
        for (size_t i = 3; i < opaqueSource.rgba.size(); i += 4) opaqueSource.rgba[i] = 255;
        source = &opaqueSource;
    }
    const stbir_pixel_layout layout = role == TextureRole::SpecGloss ? STBIR_RGBA_PM : STBIR_RGBA;
    std::vector<Image> chain =
        srgbChain(width != source->width || height != source->height ? resizedSrgb(*source, width, height, layout) : *source, layout);
    if (role == TextureRole::AlbedoCutout) preserveCoverage(chain, alphaCutoff);

    const uint32_t channels = opaque ? 3 : 4;
    std::vector<Level> levels;
    for (const Image& l : chain) {
        Level level;
        level.width = l.width;
        level.height = l.height;
        level.bytes.resize(size_t(l.width) * l.height * channels);
        for (size_t p = 0, n = size_t(l.width) * l.height; p < n; ++p)
            for (uint32_t c = 0; c < channels; ++c) level.bytes[p * channels + c] = l.rgba[p * 4 + c];
        levels.push_back(std::move(level));
    }
    const Encoding encoding = role == TextureRole::RawSrgb ? Encoding::Raw : channels == 3 ? Encoding::Etc1s : Encoding::Uastc;
    return writeKtx2(levels, channels == 3 ? kVkR8G8B8Srgb : kVkR8G8B8A8Srgb, encoding, false, outPath, error);
}

} // namespace tools
