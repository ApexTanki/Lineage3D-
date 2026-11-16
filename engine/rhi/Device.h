#pragma once

#include "core/Result.h"
#include "rhi/Caps.h"
#include "rhi/CommandList.h"
#include "rhi/Descs.h"
#include "rhi/MemoryTracker.h"
#include "rhi/TransientRing.h"

#include <memory>
#include <span>
#include <vector>

namespace platform {
class Window;
}

namespace rhi {

enum class GlStreamStrategy : uint8_t { Orphan, Fenced }; // GL uniform streaming (ARCHITECTURE.md R11)

struct DeviceDesc {
    Backend backend = Backend::Vulkan;
    platform::Window* window = nullptr;
    bool validation = false;     // Vulkan validation layers / GL debug output
    bool capsFloor = false;      // GL: behave like a minimal DX10-class OpenGL 3.3 core device (testing)
    bool vulkanBaseline = false; // Vulkan: enable no optional extensions (testing)
    GlStreamStrategy glStream = GlStreamStrategy::Orphan;
    // M9 plan D4, D5: the Vulkan pipeline cache (vk_pipelines.bin) and GL program binaries (gl/<key>.bin) live here,
    // created on demand; with a trailing separator. Null: no caches.
    const char* cacheDirectory = nullptr;
    const char* bundledLayerPath = nullptr;  // Vulkan: dev-build validation layer directory, used if none is installed
    // Tests (M9 plan D7): the device reports itself lost from this beginFrame on (0 = never), so the loss path runs.
    uint32_t simulateLossAtFrame = 0;
};

// How long a frame may take on the GPU before the device counts as lost (M9 plan D7): a fence that hasn't signaled after
// this is a hung or removed GPU, and waiting forever would freeze the application instead of reporting it.
inline constexpr uint64_t kGpuHangTimeoutNs = 10'000'000'000ull;

struct DeviceStats {
    uint32_t validationErrors = 0;
    uint32_t validationWarnings = 0;
    uint32_t drawCalls = 0; // last submitted frame
    uint64_t uploadedBytes = 0;
    uint32_t swapchainRecreations = 0; // Vulkan: rebuilds after the first (resize, suboptimal, out of date)
    uint32_t skippedDraws = 0;         // GL: draws of the last frame skipped because per-draw constants overflowed the ring
    uint64_t commandListBytes = 0;     // capacity of each command list (reserveCommandBytes; two lists, one per frame in flight)
    // Pipelines (M9 plan D3–D5): every successful createPipeline since start, and the CPU time spent in them.
    uint32_t pipelinesCreated = 0;
    double pipelineCreateMs = 0.0;
    uint32_t programBinaryHits = 0;    // GL: programs loaded from a cached binary (no compile, no link)
    uint32_t programBinaryWrites = 0;  // GL: binaries written after a link
    uint32_t pipelinesRebuilt = 0;     // made again in place by replaceShaders (hot reload)
    // GL with debug output: the driver reported compiling a shader again for the GL state of a draw (NVIDIA's "recompiled
    // based on GL state"), a hitch like a pipeline creation (ARCHITECTURE.md P30).
    uint32_t driverRecompiles = 0;
};

// Coarse-grained interface: one virtual call per resource operation or per submit, never per draw.
class Device {
public:
    virtual ~Device() = default;

    virtual const Caps& caps() const = 0;

    virtual BufferHandle createBuffer(const BufferDesc& desc) = 0;
    virtual TextureHandle createTexture(const TextureDesc& desc) = 0;
    virtual SamplerHandle createSampler(const SamplerDesc& desc) = 0;
    virtual ShaderHandle createShader(const ShaderDesc& desc) = 0;
    virtual PipelineHandle createPipeline(const PipelineDesc& desc) = 0;
    // Hot reload (dev; M9 plan D9): new code for an existing vertex and fragment shader pair (both at once, so an edit
    // that changes what passes between the stages links). Every pipeline made with either is made again from its
    // description, in place, so every handle stays valid; call between frames. The new code must keep the shaders'
    // interface (bindings, per-draw block, vertex inputs): the descriptions were made for the old one, and graphics APIs
    // don't check the match (ShaderLibrary::reload does, with shaderformat::sameInterface). False (and nothing changed) if
    // the code doesn't compile or any of those pipelines can't be made.
    virtual bool replaceShaders(ShaderHandle vertex, const ShaderDesc& vertexDesc, ShaderHandle fragment, const ShaderDesc& fragmentDesc) = 0;
    virtual ResourceSetLayoutHandle createResourceSetLayout(const ResourceSetLayoutDesc& desc) = 0;
    virtual ResourceSetHandle createResourceSet(const ResourceSetDesc& desc) = 0;
    virtual QueryPoolHandle createTimestampPool(uint32_t count) = 0;

    // Destruction is deferred until every frame that may still use the object has retired.
    virtual void destroy(BufferHandle handle) = 0;
    virtual void destroy(TextureHandle handle) = 0;
    virtual void destroy(SamplerHandle handle) = 0;
    virtual void destroy(ShaderHandle handle) = 0;
    virtual void destroy(PipelineHandle handle) = 0;
    virtual void destroy(ResourceSetLayoutHandle handle) = 0;
    virtual void destroy(ResourceSetHandle handle) = 0;
    virtual void destroy(QueryPoolHandle handle) = 0;

    // Copies the data now; on Vulkan the GPU copy runs at the start of the next submit.
    virtual core::Result<void> uploadTexture(TextureHandle texture, const TextureUpload& upload) = 0;

    // Waits for this frame slot's fence and acquires the backbuffer.
    // Returns false when there is nothing to render to (e.g. the window is minimized), and Error::DeviceLost once the GPU
    // device is lost (a driver reset, a removed GPU, a frame that never finished; M9 plan D7): from then on every
    // beginFrame fails and nothing waits on the GPU, so the application can report it and shut down cleanly.
    virtual core::Result<bool> beginFrame() = 0;
    virtual TransientRing& transient(TransientKind kind) = 0;
    virtual CommandList* allocateCommandList() = 0; // reset by the next beginFrame() of this slot
    // Every command list (current and future) holds at least `bytes` without growing. The renderer calls it with a bound
    // from the scene and settings, so recording never allocates when the camera merely sees more (M5).
    virtual void reserveCommandBytes(size_t bytes) = 0;
    virtual void submit(std::span<CommandList* const> lists) = 0;
    virtual void present() = 0;

    virtual Extent2D backbufferExtent() const = 0;
    virtual Format backbufferFormat() const = 0;

    // Timestamps written into this frame slot two frames ago, in milliseconds (NaN where unavailable).
    // Call after beginFrame() and before recording new timestamps into the same pool.
    virtual void readTimestamps(QueryPoolHandle pool, std::span<double> outMilliseconds) = 0;

    // Debug/test only (screenshots, golden images): stalls the GPU. Call between submit() and present().
    // Returns top-down RGBA8 rows on every backend.
    virtual core::Result<void> readbackBackbuffer(std::vector<uint8_t>& outRgba, Extent2D& outExtent) = 0;

    virtual void waitIdle() = 0;
    virtual const DeviceStats& stats() const = 0;

    // Writes the pipeline cache to the cache directory now (Vulkan, atomically; it is also written at shutdown). Call it
    // once loading has finished, so a crash later doesn't cost the next run its warm cache. GL writes each program binary
    // as it links, so there it does nothing.
    virtual void persistCaches() = 0;

    // GPU memory per category and its budgets (M9 plan D6); both backends count every buffer and texture here.
    MemoryTracker& memory() { return m_memory; }
    const MemoryTracker& memory() const { return m_memory; }
    // What the driver allocator really holds, for comparison with memory(): Vulkan, VMA's memory blocks and dedicated
    // allocations over every heap (alignment and block slack included); 0 where unknown (GL). Cheap; no allocation.
    virtual uint64_t allocatorBytes() const { return 0; }

protected:
    MemoryTracker m_memory;
};

} // namespace rhi
