#pragma once

#include "rhi/Device.h"

#include <glad/glad.h>

#include <memory>
#include <string>
#include <vector>

namespace rhi::gl {

struct GlTextureFormat {
    GLenum internalFormat = 0;
    GLenum format = 0;
    GLenum type = 0;
    bool compressed = false;
};

struct GlVertexFormat {
    GLint components = 0;
    GLenum type = 0;
    GLboolean normalized = GL_FALSE;
    bool integer = false; // needs glVertexAttribIPointer (ARCHITECTURE.md §9, P4)
};

GlTextureFormat glTextureFormat(Format format);
GlVertexFormat glVertexFormat(Format format);
void fillGlFormatSupport(Caps& caps);

struct GlBuffer {
    GLuint id = 0;
    uint32_t size = 0;
    bool ownedByRing = false;
    GpuMemoryCategory category = GpuMemoryCategory::Buffers; // M9 plan D6
};

struct GlTexture {
    GLuint id = 0;
    GLenum target = 0;
    TextureDesc desc;
};

struct GlSampler { GLuint id = 0; };

struct GlShader {
    // With the program binary cache on, the shader compiles on first use by a pipeline that missed the cache, so a warm
    // cache never compiles anything (M9 plan D5); without it, createShader compiles at once.
    GLuint id = 0;
    bool compileFailed = false;
    std::string source; // GLSL 330
    std::string name;   // debug name, for compile errors
    uint64_t hash = 0;  // FNV-1a 64 of the source (part of a program binary's key)
    shaderformat::Stage stage = shaderformat::Stage::Vertex;
    shaderformat::PushBlock pushBlock{};
    std::vector<shaderformat::UniformBlock> uniformBlocks;
    std::vector<shaderformat::Sampler> samplers;
    std::vector<shaderformat::VertexInput> vertexInputs;
};

struct GlPipeline {
    GLuint program = 0;
    PipelineDesc desc;
};

struct GlSetLayout { ResourceSetLayoutDesc desc; };

struct GlResourceSet {
    struct Ubo { uint8_t point; GLuint buffer; uint32_t offset; uint32_t size; bool dynamic; };
    struct Tex { uint8_t unit; GLenum target; GLuint texture; GLuint sampler; };
    Ubo ubos[kMaxSetBindings];
    Tex textures[kMaxSetBindings];
    uint8_t uboCount = 0;
    uint8_t textureCount = 0;
};

struct GlQueryPool {
    std::vector<GLuint> ids[kFramesInFlight];
    std::vector<uint8_t> written[kFramesInFlight];
    uint32_t count = 0;
};

class GlDevice final : public Device {
public:
    explicit GlDevice(const DeviceDesc& desc);
    ~GlDevice() override;
    core::Result<void> init();

    const Caps& caps() const override { return m_caps; }

    BufferHandle createBuffer(const BufferDesc& desc) override;
    TextureHandle createTexture(const TextureDesc& desc) override;
    SamplerHandle createSampler(const SamplerDesc& desc) override;
    ShaderHandle createShader(const ShaderDesc& desc) override;
    PipelineHandle createPipeline(const PipelineDesc& desc) override;
    bool replaceShaders(ShaderHandle vertex, const ShaderDesc& vertexDesc, ShaderHandle fragment, const ShaderDesc& fragmentDesc) override;
    ResourceSetLayoutHandle createResourceSetLayout(const ResourceSetLayoutDesc& desc) override;
    ResourceSetHandle createResourceSet(const ResourceSetDesc& desc) override;
    QueryPoolHandle createTimestampPool(uint32_t count) override;

    void destroy(BufferHandle handle) override;
    void destroy(TextureHandle handle) override;
    void destroy(SamplerHandle handle) override;
    void destroy(ShaderHandle handle) override;
    void destroy(PipelineHandle handle) override;
    void destroy(ResourceSetLayoutHandle handle) override;
    void destroy(ResourceSetHandle handle) override;
    void destroy(QueryPoolHandle handle) override;

    core::Result<void> uploadTexture(TextureHandle texture, const TextureUpload& upload) override;

    core::Result<bool> beginFrame() override;
    TransientRing& transient(TransientKind kind) override { return m_rings[uint32_t(kind)]; }
    CommandList* allocateCommandList() override;
    void reserveCommandBytes(size_t bytes) override;
    void submit(std::span<CommandList* const> lists) override;
    void present() override;

    Extent2D backbufferExtent() const override { return m_backbuffer; }
    Format backbufferFormat() const override { return m_caps.srgbBackbuffer ? Format::RGBA8Srgb : Format::RGBA8Unorm; }
    void readTimestamps(QueryPoolHandle pool, std::span<double> outMilliseconds) override;
    core::Result<void> readbackBackbuffer(std::vector<uint8_t>& outRgba, Extent2D& outExtent) override;
    void waitIdle() override {
        if (!m_lost) glFinish(); // M9 plan D7: a hung GPU would never finish
    }
    const DeviceStats& stats() const override { return m_stats; }
    void persistCaches() override {} // program binaries are written as each program links

    void onDebugMessage(GLenum type, GLenum severity, const char* message);

private:
    struct Deferred {
        enum class Kind : uint8_t { Buffer, Texture, Sampler, Shader, Program, Query } kind;
        GLuint id;
        uint64_t serial; // deleted once this submission has completed
    };

    // Redundant-state filter: GL drivers are slow at absorbing repeated binds.
    struct StateCache {
        GLuint program = 0;
        int cull = -1;
        GLenum frontFace = 0;
        GLenum cullFace = 0;
        int depthTest = -1;
        int depthWrite = -1;
        GLenum depthFunc = 0;
        int blend = -1;
        GLenum blendSrc = 0, blendDst = 0, blendSrcA = 0, blendDstA = 0, blendEq = 0, blendEqA = 0;
        uint8_t colorMask = 0xFF;
        int scissor = -1;
        int alphaToCoverage = -1;
        int polygonOffset = -1;
        int polygonOffsetLine = -1;
        int wireframe = -1;
        int framebufferSrgb = -1; // GL_FRAMEBUFFER_SRGB: on only while an sRGB target is bound (M8 plan D8)
        GLuint ubo[16] = {};
        uint32_t uboOffset[16] = {};
        uint32_t uboSize[16] = {};
        GLuint texture[16] = {};
        GLuint sampler[16] = {};
        uint32_t enabledAttributes = 0;
    };

    bool queryCaps();
    void applyCapsFloor();
    bool createRings();
    void defer(Deferred::Kind kind, GLuint id);
    void runDeferred(bool all);
    void deleteNow(const Deferred& d);
    void label(GLenum identifier, GLuint name, const char* text);

    // Framebuffer objects are cached by attachment set; entries are purged when an attached texture is deleted.
    struct FboKey {
        GLuint color[kMaxColorAttachments] = {};
        uint16_t mip[kMaxColorAttachments] = {};
        uint16_t layer[kMaxColorAttachments] = {};
        GLuint depth = 0;
        uint16_t depthLayer = 0;
        uint8_t colorCount = 0;
        bool operator==(const FboKey&) const = default;
    };
    struct FboEntry {
        FboKey key;
        GLuint fbo;
    };
    struct PendingResolve {
        GLuint sourceFbo;
        uint32_t attachment;
        const GlTexture* target;
        uint32_t width, height;
    };

    void replay(const CommandList& list, uint32_t& perDrawIndex);
    void beginRenderPass(const RenderPassDesc& desc);
    void endRenderPass();
    GLuint framebufferFor(const FboKey& key, const GlTexture* const* colors, const GlTexture* depth);
    void applyPipelineState(const GlPipeline& pipeline);
    void applyVertexInput();
    void bindUniformRange(uint8_t point, GLuint buffer, uint32_t offset, uint32_t size);
    void setCap(GLenum cap, int& cached, bool enable);
    // Program binaries (M9 plan D5): <cache directory>gl/<key as 16 hex digits>.bin.
    void initProgramCache();
    bool compileShader(GlShader& shader);
    GLuint makeProgram(GlShader& vs, GlShader& fs, const PipelineDesc& desc); // linked, bindings applied; 0 on failure
    uint64_t programKey(const GlShader& vs, const GlShader& fs) const;
    GLuint loadProgramBinary(uint64_t key, const GlShader& vs, const GlShader& fs);
    void saveProgramBinary(GLuint program, uint64_t key, const GlShader& vs, const GlShader& fs);
    std::string programBinaryPath(uint64_t key) const;

    DeviceDesc m_desc;
    Caps m_caps;
    DeviceStats m_stats;
    bool m_khrDebug = false;
    bool m_timerQueries = false;
    bool m_inReadback = false;
    bool m_lost = false;                // M9 plan D7: the device is lost; beginFrame fails and nothing waits on the GPU
    PFNGLGETGRAPHICSRESETSTATUSPROC m_resetStatus = nullptr; // with reset notification only (init)
    uint32_t m_framesBegun = 0;         // for DeviceDesc::simulateLossAtFrame
    bool m_programCache = false;        // binaries are read and written (caps.programBinaryCache and a cache directory)
    std::string m_programCacheDir;      // <cache directory>gl/
    uint64_t m_driverKey = 0;           // FNV-1a 64 of vendor, renderer and version: a driver update invalidates binaries
    std::vector<GLint> m_binaryFormats; // formats the driver accepts in glProgramBinary
    Extent2D m_backbuffer;
    GLuint m_vao = 0;

    uint32_t m_slot = 0;
    bool m_slotUsed = false;    // a frame was submitted in m_slot; the next beginFrame moves on
    bool m_frameActive = false; // between a successful beginFrame and submit
    GLsync m_fences[kFramesInFlight] = {};
    uint64_t m_frameSerial[kFramesInFlight] = {};
    uint64_t m_submitSerial = 0;
    uint64_t m_completedSerial = 0;
    std::vector<Deferred> m_deferred;
    std::vector<std::unique_ptr<CommandList>> m_commandLists[kFramesInFlight];
    uint32_t m_commandListsUsed = 0;
    size_t m_commandBytes = 256 * 1024; // every command list holds at least this (reserveCommandBytes)

    TransientRing m_rings[uint32_t(TransientKind::Count)];
    BufferHandle m_ringBuffers[uint32_t(TransientKind::Count)];
    std::vector<std::byte> m_ringStaging[uint32_t(TransientKind::Count)];
    // Ring offsets of this frame's per-draw blocks that fit. When they don't all fit, the first m_perDrawHead blocks and the
    // last ones (from m_perDrawTailStart) keep theirs and the middle is dropped: the end of a frame (heat haze, the post
    // chain, the debug UI) must draw or the backbuffer shows garbage (ARCHITECTURE.md R40).
    std::vector<uint32_t> m_perDrawOffsets;
    uint32_t m_perDrawHead = 0, m_perDrawTailStart = 0;
    static constexpr uint32_t kPerDrawTailBlocks = 256; // kept for the end of an overflowing frame
    static constexpr uint32_t kNoPerDraw = ~0u;         // the per-draw block didn't fit: its draws are skipped
    bool m_skipDraws = false;

    StateCache m_state;
    std::vector<FboEntry> m_fbos;
    PendingResolve m_pendingResolves[kMaxColorAttachments] = {};
    uint32_t m_pendingResolveCount = 0;
    const GlPipeline* m_pipeline = nullptr;
    BufferSlice m_streams[kMaxVertexStreams];
    BufferSlice m_indexBuffer;
    IndexType m_indexType = IndexType::U16;
    bool m_vertexDirty = true;

    core::HandlePool<GlBuffer, BufferTag> m_buffers{4096};
    core::HandlePool<GlTexture, TextureTag> m_textures{4096};
    core::HandlePool<GlSampler, SamplerTag> m_samplers{256};
    core::HandlePool<GlShader, ShaderTag> m_shaders{2048};
    core::HandlePool<GlPipeline, PipelineTag> m_pipelines{2048};
    core::HandlePool<GlSetLayout, ResourceSetLayoutTag> m_setLayouts{256};
    core::HandlePool<GlResourceSet, ResourceSetTag> m_resourceSets{8192};
    core::HandlePool<GlQueryPool, QueryPoolTag> m_queryPools{64};
};

} // namespace rhi::gl
