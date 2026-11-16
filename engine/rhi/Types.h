#pragma once

#include "core/Handle.h"

#include <cstddef>
#include <cstdint>

namespace rhi {

inline constexpr uint32_t kFramesInFlight = 2;
inline constexpr uint32_t kMaxColorAttachments = 4;
inline constexpr uint32_t kMaxVertexStreams = 4;
inline constexpr uint32_t kMaxVertexAttributes = 13;
inline constexpr uint32_t kMaxSetBindings = 8;
inline constexpr uint32_t kMaxDynamicOffsets = 4;

using BufferHandle = core::Handle<struct BufferTag>;
using TextureHandle = core::Handle<struct TextureTag>;
using SamplerHandle = core::Handle<struct SamplerTag>;
using ShaderHandle = core::Handle<struct ShaderTag>;
using PipelineHandle = core::Handle<struct PipelineTag>;
using ResourceSetLayoutHandle = core::Handle<struct ResourceSetLayoutTag>;
using ResourceSetHandle = core::Handle<struct ResourceSetTag>;
using QueryPoolHandle = core::Handle<struct QueryPoolTag>;

enum class Backend : uint8_t { Vulkan, GL33 };

constexpr const char* backendName(Backend b) {
    switch (b) {
    case Backend::Vulkan: return "Vulkan";
    case Backend::GL33: return "OpenGL 3.3";
    }
    return "?";
}

enum class Format : uint8_t {
    Undefined,
    // color / render targets
    R8Unorm, RG8Unorm, RGBA8Unorm, RGBA8Srgb, BGRA8Unorm, BGRA8Srgb, R16F, RG16F, RGBA16F, R11G11B10F, R32F,
    // depth
    D16, D24S8, D32F,
    // compressed
    BC1Srgb, BC3Srgb, BC3Unorm, BC4, BC5, BC6HUfloat, BC7Unorm, BC7Srgb,
    // vertex-only
    RG32F, RGB32F, RGBA32F, RGBA16Sint, RGBA8Snorm, RGBA8Uint, RG16Unorm, RGBA32Uint,
    Count
};

inline constexpr uint32_t kFormatCount = uint32_t(Format::Count);

enum class IndexType : uint8_t { U16, U32 };
enum class MemoryKind : uint8_t { GpuOnly, Upload, Readback };
enum class TextureType : uint8_t { Tex2D, Tex2DArray, Cube };
enum class LoadOp : uint8_t { Load, Clear, DontCare };
enum class StoreOp : uint8_t { Store, DontCare };
enum class CompareOp : uint8_t { Never, Less, Equal, LessOrEqual, Greater, NotEqual, GreaterOrEqual, Always };
enum class CullMode : uint8_t { None, Front, Back };
enum class Topology : uint8_t { Triangles, Lines };
enum class Filter : uint8_t { Nearest, Linear };
enum class MipMode : uint8_t { Nearest, Linear };
enum class AddressMode : uint8_t { Repeat, ClampToEdge, MirroredRepeat };
enum class BlendFactor : uint8_t { Zero, One, SrcAlpha, OneMinusSrcAlpha, DstAlpha, OneMinusDstAlpha, SrcColor, DstColor };
enum class BlendOp : uint8_t { Add, Subtract, ReverseSubtract, Min, Max };
enum class TransientKind : uint8_t { Uniform, Vertex, Index, Count };

enum class BufferUsage : uint8_t { None = 0, Vertex = 1, Index = 2, Uniform = 4, TransferSrc = 8, TransferDst = 16 };
enum class TextureUsage : uint8_t { None = 0, Sampled = 1, ColorTarget = 2, DepthTarget = 4, TransferSrc = 8, TransferDst = 16 };
enum class ShaderStageMask : uint8_t { None = 0, Vertex = 1, Fragment = 2, All = 3 };

#define FL_RHI_FLAGS(T)                                                                                                \
    constexpr T operator|(T a, T b) { return T(uint32_t(a) | uint32_t(b)); }                                           \
    constexpr T operator&(T a, T b) { return T(uint32_t(a) & uint32_t(b)); }                                           \
    constexpr bool any(T a) { return uint32_t(a) != 0; }
FL_RHI_FLAGS(BufferUsage)
FL_RHI_FLAGS(TextureUsage)
FL_RHI_FLAGS(ShaderStageMask)
#undef FL_RHI_FLAGS

struct Extent2D {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct BufferSlice {
    BufferHandle buffer;
    uint32_t offset = 0;
    uint32_t size = 0;
};

// Pixel rectangle in attachment texel rows: y = 0 is the first row in memory on every backend.
// Top-left UI rectangles are converted by renderer::ClipSpace, never inside a backend (ARCHITECTURE.md §4.2, C6).
struct Rect {
    int32_t x = 0;
    int32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

// Data only. Interpreted exclusively by renderer::ClipSpace.
struct ClipSpaceConvention {
    bool depthZeroToOne = true; // Vulkan: true. GL: false (NDC z in [-1, 1])
    bool originTopLeft = true;  // texel row 0 is the top of the screen. Vulkan: true. GL: false
};

struct FormatInfo {
    uint8_t blockBytes;  // bytes per pixel, or per 4x4 block for compressed formats
    uint8_t blockSize;   // 1 for uncompressed, 4 for block-compressed
    bool depth;
    bool srgb;
};

FormatInfo formatInfo(Format format);
const char* formatName(Format format);

} // namespace rhi
