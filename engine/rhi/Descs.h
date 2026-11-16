#pragma once

#include "rhi/Types.h"

#include <shaderformat/BindingModel.h>
#include <shaderformat/ShaderPackage.h>

#include <span>

namespace rhi {

struct BufferDesc {
    const char* debugName = "";
    uint32_t size = 0;
    BufferUsage usage = BufferUsage::None;
    MemoryKind memory = MemoryKind::GpuOnly;
    const void* initialData = nullptr;
};

struct TextureDesc {
    const char* debugName = "";
    TextureType type = TextureType::Tex2D;
    Format format = Format::RGBA8Unorm;
    uint32_t width = 1;
    uint32_t height = 1;
    uint16_t layers = 1; // array layers (6 for a cube)
    uint16_t mips = 1;
    uint8_t samples = 1;
    TextureUsage usage = TextureUsage::Sampled;
};

// A region upload; pixels are tightly packed rows of rowLength texels (0 = width).
struct TextureUpload {
    uint32_t mip = 0;
    uint32_t layer = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t rowLength = 0;
    std::span<const std::byte> data;
};

struct SamplerDesc {
    Filter filter = Filter::Linear;
    MipMode mip = MipMode::Linear;
    AddressMode address = AddressMode::Repeat;
    float maxAnisotropy = 1.0f;
    CompareOp compare = CompareOp::Never; // Never = no depth comparison
};

// One compiled variant from a .fsp package: SPIR-V for Vulkan, GLSL text for GL, plus its reflection.
struct ShaderDesc {
    const char* debugName = "";
    shaderformat::Stage stage = shaderformat::Stage::Vertex;
    std::span<const std::byte> code;
    const shaderformat::Reflection* reflection = nullptr; // must stay valid while the shader exists
};

struct VertexAttribute {
    uint8_t location = 0;
    uint8_t stream = 0;
    Format format = Format::RGB32F;
    uint16_t offset = 0;
};

struct VertexStream {
    uint16_t stride = 0;
    bool perInstance = false;
};

struct VertexLayout {
    VertexStream streams[kMaxVertexStreams];
    VertexAttribute attributes[kMaxVertexAttributes];
    uint8_t streamCount = 0;
    uint8_t attributeCount = 0;
};

struct RasterState {
    CullMode cull = CullMode::Back;
    // Front faces are counter-clockwise on both backends (ARCHITECTURE.md §4.2); mirrored transforms (det < 0)
    // turn them clockwise.
    bool frontFaceClockwise = false;
    bool wireframe = false;
    bool alphaToCoverage = false;
    float depthBiasConstant = 0.0f;
    float depthBiasSlope = 0.0f;
};

struct DepthState {
    bool test = false;
    bool write = false;
    CompareOp compare = CompareOp::LessOrEqual;
};

struct BlendState {
    bool enable = false;
    BlendFactor srcColor = BlendFactor::One;
    BlendFactor dstColor = BlendFactor::Zero;
    BlendOp colorOp = BlendOp::Add;
    BlendFactor srcAlpha = BlendFactor::One;
    BlendFactor dstAlpha = BlendFactor::Zero;
    BlendOp alphaOp = BlendOp::Add;
    uint8_t writeMask = 0xF;
};

// Pipeline ↔ render pass compatibility key (formats + sample count), not a specific pass instance.
struct AttachmentLayout {
    Format color[kMaxColorAttachments] = {};
    uint8_t colorCount = 0;
    Format depth = Format::Undefined;
    uint8_t samples = 1;
};

struct PipelineDesc {
    const char* debugName = "";
    ShaderHandle vertex;
    ShaderHandle fragment;
    VertexLayout vertexLayout;
    Topology topology = Topology::Triangles;
    RasterState raster;
    DepthState depth;
    BlendState blend[kMaxColorAttachments];
    AttachmentLayout attachments;
    ResourceSetLayoutHandle setLayouts[shaderformat::kMaxSets]; // sets 0–3 (ARCHITECTURE.md §4.6)
    uint8_t perDrawBytes = 0;                                   // ≤ 128
};

enum class BindingKind : uint8_t { UniformBuffer, UniformBufferDynamic, Texture };

struct ResourceBinding {
    uint8_t binding = 0;
    BindingKind kind = BindingKind::UniformBuffer;
    ShaderStageMask stages = ShaderStageMask::All;
    TextureType textureType = TextureType::Tex2D;
    bool shadowCompare = false;
};

struct ResourceSetLayoutDesc {
    uint8_t set = 0;
    ResourceBinding bindings[kMaxSetBindings];
    uint8_t count = 0;
};

struct ResourceWrite {
    uint8_t binding = 0;
    BufferSlice buffer;    // uniform buffers: the bound range (dynamic offsets are added at bind time)
    TextureHandle texture; // textures
    SamplerHandle sampler;
};

struct ResourceSetDesc {
    const char* debugName = "";
    ResourceSetLayoutHandle layout;
    std::span<const ResourceWrite> writes;
};

struct ColorAttachment {
    TextureHandle texture; // ignored when RenderPassDesc::toBackbuffer is set (color[0] is the backbuffer)
    uint16_t mip = 0;
    uint16_t layer = 0;
    LoadOp load = LoadOp::Clear;
    StoreOp store = StoreOp::Store;
    float clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    // Multisampled `texture` only: single-sample target that receives the resolve when the pass ends.
    // Render targets rest in the shader-read state between passes; multisampled ones stay attachments.
    TextureHandle resolveTo;
};

struct DepthAttachment {
    TextureHandle texture;
    uint16_t layer = 0;
    LoadOp load = LoadOp::Clear;
    StoreOp store = StoreOp::DontCare;
    float clearDepth = 1.0f;
};

struct RenderPassDesc {
    const char* debugName = ""; // must outlive the frame (string literals)
    ColorAttachment color[kMaxColorAttachments];
    uint8_t colorCount = 0;
    DepthAttachment depth;
    bool useDepth = false;
    bool toBackbuffer = false;
};

} // namespace rhi
