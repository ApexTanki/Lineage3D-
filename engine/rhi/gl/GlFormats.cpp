#include "rhi/gl/GlDevice.h"

namespace rhi::gl {

// Compressed internal formats are spelled as raw values, so they don't depend on which extension
// headers glad generated.
GlTextureFormat glTextureFormat(Format format) {
    switch (format) {
    case Format::R8Unorm: return {GL_R8, GL_RED, GL_UNSIGNED_BYTE};
    case Format::RG8Unorm: return {GL_RG8, GL_RG, GL_UNSIGNED_BYTE};
    case Format::RGBA8Unorm: return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};
    case Format::RGBA8Srgb: return {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE};
    case Format::R16F: return {GL_R16F, GL_RED, GL_HALF_FLOAT};
    case Format::RG16F: return {GL_RG16F, GL_RG, GL_HALF_FLOAT};
    case Format::RGBA16F: return {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT};
    case Format::R11G11B10F: return {GL_R11F_G11F_B10F, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV};
    case Format::R32F: return {GL_R32F, GL_RED, GL_FLOAT};
    case Format::D16: return {GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT};
    case Format::D24S8: return {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8};
    case Format::D32F: return {GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT};
    case Format::BC1Srgb: return {0x8C4D, 0, 0, true};      // COMPRESSED_SRGB_ALPHA_S3TC_DXT1
    case Format::BC3Srgb: return {0x8C4F, 0, 0, true};      // COMPRESSED_SRGB_ALPHA_S3TC_DXT5
    case Format::BC3Unorm: return {0x83F3, 0, 0, true};     // COMPRESSED_RGBA_S3TC_DXT5
    case Format::BC4: return {0x8DBB, 0, 0, true};          // COMPRESSED_RED_RGTC1
    case Format::BC5: return {0x8DBD, 0, 0, true};          // COMPRESSED_RG_RGTC2
    case Format::BC6HUfloat: return {0x8E8F, 0, 0, true};   // COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT
    case Format::BC7Unorm: return {0x8E8C, 0, 0, true};     // COMPRESSED_RGBA_BPTC_UNORM
    case Format::BC7Srgb: return {0x8E8D, 0, 0, true};      // COMPRESSED_SRGB_ALPHA_BPTC_UNORM
    default: return {};
    }
}

GlVertexFormat glVertexFormat(Format format) {
    switch (format) {
    case Format::RG32F: return {2, GL_FLOAT, GL_FALSE, false};
    case Format::RGB32F: return {3, GL_FLOAT, GL_FALSE, false};
    case Format::RGBA32F: return {4, GL_FLOAT, GL_FALSE, false};
    case Format::RGBA16Sint: return {4, GL_SHORT, GL_FALSE, true};
    case Format::RGBA8Snorm: return {4, GL_BYTE, GL_TRUE, false};
    case Format::RGBA8Uint: return {4, GL_UNSIGNED_BYTE, GL_FALSE, true};
    case Format::RG16Unorm: return {2, GL_UNSIGNED_SHORT, GL_TRUE, false};
    case Format::RGBA32Uint: return {4, GL_UNSIGNED_INT, GL_FALSE, true};
    case Format::RGBA8Unorm: return {4, GL_UNSIGNED_BYTE, GL_TRUE, false};
    case Format::RGBA16F: return {4, GL_HALF_FLOAT, GL_FALSE, false}; // core since GL 3.0 (particle instances, M7)
    default: return {};
    }
}

// GL 3.3 has no per-format query, so support follows the GL 3.3 core rules plus extensions. Float color
// targets (renderable and blendable) and RGTC are core since GL 3.0.
void fillGlFormatSupport(Caps& caps) {
    auto set = [&](Format f, bool sampled, bool filterable, bool color, bool blend, bool depth) {
        caps.formats[uint32_t(f)] = {sampled, filterable, color, blend, depth, false};
    };
    for (Format f : {Format::R8Unorm, Format::RG8Unorm, Format::RGBA8Unorm, Format::RGBA8Srgb, Format::R16F,
                     Format::RG16F, Format::RGBA16F, Format::R11G11B10F, Format::R32F})
        set(f, true, true, true, true, false);
    for (Format f : {Format::D16, Format::D24S8, Format::D32F}) set(f, true, true, false, false, true);

    const bool s3tc = GLAD_GL_EXT_texture_compression_s3tc != 0;
    const bool bptc = GLAD_GL_ARB_texture_compression_bptc != 0;
    for (Format f : {Format::BC1Srgb, Format::BC3Srgb, Format::BC3Unorm}) set(f, s3tc, s3tc, false, false, false);
    for (Format f : {Format::BC4, Format::BC5}) set(f, true, true, false, false, false);
    for (Format f : {Format::BC6HUfloat, Format::BC7Unorm, Format::BC7Srgb}) set(f, bptc, bptc, false, false, false);

    for (Format f : {Format::RG32F, Format::RGB32F, Format::RGBA32F, Format::RGBA16Sint, Format::RGBA8Snorm,
                     Format::RGBA8Uint, Format::RG16Unorm, Format::RGBA8Unorm, Format::RGBA32Uint, Format::RGBA16F})
        caps.formats[uint32_t(f)].vertex = true;

    caps.bc1to5 = s3tc;
    caps.bptc = bptc;
}

} // namespace rhi::gl
