#include "assets/Texture.h"

#include <ktx.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>

namespace assets {
namespace {

bool fail(std::string* error, const char* message) {
    if (error) *error = message;
    return false;
}

// libktx initializes the Basis transcoder's global tables on first use behind an unsynchronized static flag.
// Transcodes run under a lock until one has succeeded (a failing one may stop before the tables exist); after
// that they run in parallel.
std::mutex g_firstTranscodeMutex;
std::atomic<bool> g_transcoderReady{false};

KTX_error_code transcode(ktxTexture2* texture, ktx_transcode_fmt_e format) {
    if (g_transcoderReady.load(std::memory_order_acquire)) return ktxTexture2_TranscodeBasis(texture, format, 0);
    std::lock_guard<std::mutex> lock(g_firstTranscodeMutex);
    const KTX_error_code result = ktxTexture2_TranscodeBasis(texture, format, 0);
    if (result == KTX_SUCCESS) g_transcoderReady.store(true, std::memory_order_release);
    return result;
}

PixelFormat fromVkFormat(uint32_t vkFormat) {
    switch (vkFormat) {
    case 37: return PixelFormat::RGBA8Unorm;  // VK_FORMAT_R8G8B8A8_UNORM
    case 43: return PixelFormat::RGBA8Srgb;   // VK_FORMAT_R8G8B8A8_SRGB
    case 132:                                 // VK_FORMAT_BC1_RGB_SRGB_BLOCK
    case 134: return PixelFormat::BC1Srgb;    // VK_FORMAT_BC1_RGBA_SRGB_BLOCK
    case 137: return PixelFormat::BC3Unorm;   // VK_FORMAT_BC3_UNORM_BLOCK
    case 138: return PixelFormat::BC3Srgb;    // VK_FORMAT_BC3_SRGB_BLOCK
    case 139: return PixelFormat::BC4;        // VK_FORMAT_BC4_UNORM_BLOCK
    case 141: return PixelFormat::BC5;        // VK_FORMAT_BC5_UNORM_BLOCK
    case 143: return PixelFormat::BC6H;       // VK_FORMAT_BC6H_UFLOAT_BLOCK
    case 145: return PixelFormat::BC7Unorm;   // VK_FORMAT_BC7_UNORM_BLOCK
    case 146: return PixelFormat::BC7Srgb;    // VK_FORMAT_BC7_SRGB_BLOCK
    default: return PixelFormat::Unknown;
    }
}

struct Choice {
    ktx_transcode_fmt_e transcode;
    PixelFormat format;
};

Choice chooseTarget(uint32_t components, bool srgb, bool etc1s, const TranscodeTargets& t) {
    const Choice rgba8{KTX_TTF_RGBA32, srgb ? PixelFormat::RGBA8Srgb : PixelFormat::RGBA8Unorm};
    if (!t.bc) return rgba8;
    if (!srgb && components == 1) return {KTX_TTF_BC4_R, PixelFormat::BC4};
    if (!srgb && components == 2) return {KTX_TTF_BC5_RG, PixelFormat::BC5};
    if (components == 4) {
        if (t.bptc) return {KTX_TTF_BC7_RGBA, srgb ? PixelFormat::BC7Srgb : PixelFormat::BC7Unorm};
        return {KTX_TTF_BC3_RGBA, srgb ? PixelFormat::BC3Srgb : PixelFormat::BC3Unorm};
    }
    // RGB: ETC1S carries no more detail than BC1 holds; UASTC is worth BC7 where available.
    if (srgb && (etc1s || !t.bptc)) return {KTX_TTF_BC1_RGB, PixelFormat::BC1Srgb};
    if (t.bptc) return {KTX_TTF_BC7_RGBA, srgb ? PixelFormat::BC7Srgb : PixelFormat::BC7Unorm};
    return rgba8;
}

} // namespace

uint32_t pixelFormatBlockBytes(PixelFormat format) {
    switch (format) {
    case PixelFormat::BC1Srgb:
    case PixelFormat::BC4: return 8;
    case PixelFormat::BC3Srgb:
    case PixelFormat::BC3Unorm:
    case PixelFormat::BC5:
    case PixelFormat::BC6H:
    case PixelFormat::BC7Unorm:
    case PixelFormat::BC7Srgb: return 16;
    case PixelFormat::RGBA8Unorm:
    case PixelFormat::RGBA8Srgb: return 4;
    case PixelFormat::Unknown: break;
    }
    return 0;
}

bool pixelFormatIsBlockCompressed(PixelFormat format) {
    return format != PixelFormat::RGBA8Unorm && format != PixelFormat::RGBA8Srgb && format != PixelFormat::Unknown;
}

bool decodeKtx2(std::span<const std::byte> file, const TranscodeTargets& targets, DecodedTexture& out, std::string* error) {
    ktxTexture2* texture = nullptr;
    if (ktxTexture2_CreateFromMemory(reinterpret_cast<const ktx_uint8_t*>(file.data()), file.size(),
                                     KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture) != KTX_SUCCESS)
        return fail(error, "not a valid KTX2 file");
    struct Guard {
        ktxTexture2* t;
        ~Guard() { ktxTexture_Destroy(ktxTexture(t)); }
    } guard{texture};

    if (texture->numDimensions != 2 || texture->baseDepth > 1) return fail(error, "only 2D textures, 2D arrays and cubes are supported");
    if (texture->numFaces != 1 && texture->numFaces != 6) return fail(error, "a cube needs 6 faces");
    if (texture->numFaces == 6 && texture->isArray) return fail(error, "cube arrays aren't on the GL 3.3 floor");
    const uint32_t components = ktxTexture2_GetNumComponents(texture);
    const bool srgb = ktxTexture2_GetOETF(texture) == KHR_DF_TRANSFER_SRGB;
    out.hasAlpha = components == 4;

    if (ktxTexture2_NeedsTranscoding(texture)) {
        const bool etc1s = texture->supercompressionScheme == KTX_SS_BASIS_LZ;
        const Choice choice = chooseTarget(components, srgb, etc1s, targets);
        if (transcode(texture, choice.transcode) != KTX_SUCCESS) return fail(error, "Basis transcoding failed");
        out.format = choice.format;
    } else {
        out.format = fromVkFormat(texture->vkFormat);
        if (out.format == PixelFormat::Unknown) return fail(error, "unsupported KTX2 pixel format");
        if (pixelFormatIsBlockCompressed(out.format) && !targets.bc) return fail(error, "the device can't sample BC textures");
        if ((out.format == PixelFormat::BC6H || out.format == PixelFormat::BC7Unorm || out.format == PixelFormat::BC7Srgb) &&
            !targets.bptc)
            return fail(error, "the device can't sample BC6H/BC7 textures (load the RGBM variant)");
    }

    out.width = texture->baseWidth;
    out.height = texture->baseHeight;
    out.shape = texture->numFaces == 6 ? TextureShape::Cube : texture->isArray ? TextureShape::Array : TextureShape::Tex2D;
    out.layers = texture->numFaces == 6 ? 6u : std::max(1u, texture->numLayers);
    out.mips = texture->numLevels;
    out.levels.clear();
    out.data.clear();
    const ktx_uint8_t* pixels = ktxTexture_GetData(ktxTexture(texture));
    const size_t dataSize = ktxTexture_GetDataSize(ktxTexture(texture));
    for (uint32_t level = 0; level < texture->numLevels; ++level) {
        const size_t size = ktxTexture_GetImageSize(ktxTexture(texture), level);
        for (uint32_t layer = 0; layer < out.layers; ++layer) {
            const uint32_t arrayLayer = texture->numFaces == 6 ? 0 : layer;
            const uint32_t face = texture->numFaces == 6 ? layer : 0;
            ktx_size_t offset = 0;
            if (ktxTexture_GetImageOffset(ktxTexture(texture), level, arrayLayer, face, &offset) != KTX_SUCCESS)
                return fail(error, "missing image");
            if (offset + size > dataSize) return fail(error, "image out of bounds");
            DecodedTexture::Level l;
            l.offset = uint32_t(out.data.size());
            l.size = uint32_t(size);
            l.width = std::max(1u, out.width >> level);
            l.height = std::max(1u, out.height >> level);
            l.mip = level;
            l.layer = layer;
            out.levels.push_back(l);
            out.data.insert(out.data.end(), reinterpret_cast<const std::byte*>(pixels + offset),
                            reinterpret_cast<const std::byte*>(pixels + offset + size));
        }
    }
    // Basis stores 2-channel data (normal maps) as "rrrg": BC5 picks R and A, but the RGBA8 fallback keeps that layout.
    // Moving A into G makes shaders read .rg on every path (M2 plan D7).
    if (components == 2 && !pixelFormatIsBlockCompressed(out.format))
        for (size_t i = 0; i + 3 < out.data.size(); i += 4) out.data[i + 1] = out.data[i + 3];
    return true;
}

} // namespace assets
