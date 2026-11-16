#pragma once

#include "rhi/Device.h"
#include "rhi/vulkan/VkCommon.h"

#include <memory>
#include <string>
#include <vector>

namespace rhi::vk {

struct BufferRes {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    void* mapped = nullptr;
    uint32_t size = 0;
    bool ownedByRing = false; // transient ring / staging buffers are destroyed by the device itself
    GpuMemoryCategory category = GpuMemoryCategory::Buffers; // M9 plan D6
};

// Array render targets get one single-layer view per layer: a framebuffer attaches exactly one layer (M4 plan D3).
inline constexpr uint32_t kMaxLayerViews = 8;

struct TextureRes {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkImageView view = VK_NULL_HANDLE; // the whole texture, for sampling
    VkImageView layerViews[kMaxLayerViews] = {};
    uint32_t layerViewCount = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED; // layout after all recorded work
    TextureDesc desc;

    VkImageView attachmentView(uint32_t layer) const { return layerViewCount ? layerViews[layer] : view; }
};

struct SamplerRes { VkSampler sampler = VK_NULL_HANDLE; };
struct ShaderRes { VkShaderModule module = VK_NULL_HANDLE; };
struct SetLayoutRes {
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    ResourceSetLayoutDesc desc; // binding kinds decide the descriptor type of each write
};

struct PipelineRes {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE; // owned by the pipeline-layout cache
    PipelineDesc desc;                        // to make it again when one of its shaders is hot-reloaded (M9 plan D9)
};

struct ResourceSetRes {
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
};

struct QueryPoolRes {
    VkQueryPool pools[kFramesInFlight] = {};
    uint32_t count = 0;
    bool submitted[kFramesInFlight] = {}; // results of this slot are valid once its fence signals
    std::vector<uint64_t> scratch;        // value + availability pairs, sized once
};

class VulkanDevice final : public Device {
public:
    explicit VulkanDevice(const DeviceDesc& desc);
    ~VulkanDevice() override;
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

    Extent2D backbufferExtent() const override { return {m_swapExtent.width, m_swapExtent.height}; }
    Format backbufferFormat() const override { return m_backbufferFormat; }
    void readTimestamps(QueryPoolHandle pool, std::span<double> outMilliseconds) override;
    core::Result<void> readbackBackbuffer(std::vector<uint8_t>& outRgba, Extent2D& outExtent) override;
    void waitIdle() override;
    const DeviceStats& stats() const override { return m_stats; }
    void persistCaches() override {
        if (!m_lost) savePipelineCache();
    }
    uint64_t allocatorBytes() const override;

    void onValidationMessage(bool isError, const char* message);

private:
    struct FrameData {
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        uint64_t serial = 0; // submission serial of the last frame recorded in this slot
        std::vector<std::unique_ptr<CommandList>> commandLists;
        uint32_t commandListsUsed = 0;
    };

    struct Deferred {
        enum class Kind : uint8_t {
            Buffer, Image, ImageView, Sampler, ShaderModule, Pipeline, PipelineLayout, SetLayout, DescriptorSet, QueryPool, Framebuffer
        };
        Kind kind;
        uint64_t handle;
        uint64_t extra; // image view / descriptor pool
        VmaAllocation allocation;
        uint64_t serial; // freed once this submission has completed on the GPU
    };

    struct PendingCopy {
        VkImage image;
        VkImageAspectFlags aspect;
        VkImageLayout oldLayout;  // of the copied subresource
        VkBufferImageCopy region;
        // First upload into a fresh image: every subresource moves out of UNDEFINED once, so later copies of other
        // mips/layers start from a known layout (layouts are tracked per image, uploads are per subresource).
        bool initializeImage;
        uint32_t mipLevels;
        uint32_t arrayLayers;
    };

    struct PendingBufferCopy {
        VkBuffer dst;
        VkBufferCopy region;
    };

    bool stageBytes(const void* data, uint32_t bytes, uint32_t& outOffset);
    bool uploadImmediately(const void* data, uint32_t bytes, const PendingCopy* image, const PendingBufferCopy* buffer);
    void recordCopies(VkCommandBuffer cmd, VkBuffer staging, std::span<const PendingCopy> images,
                      std::span<const PendingBufferCopy> buffers);

    struct RenderPassKey {
        VkFormat color[kMaxColorAttachments];
        VkFormat depth;
        uint8_t colorCount;
        uint8_t samples;
        uint8_t colorLoad[kMaxColorAttachments];
        uint8_t colorStore[kMaxColorAttachments];
        uint8_t colorResolve[kMaxColorAttachments]; // 1 = resolved into a single-sample attachment at the end
        uint8_t depthLoad;
        uint8_t depthStore;
        uint8_t toBackbuffer;
        uint8_t depthSampled; // 1 = a sampled depth target (shadow maps): it rests in SHADER_READ_ONLY between passes
        bool operator==(const RenderPassKey&) const = default;
    };

    struct FramebufferEntry {
        VkRenderPass renderPass;
        VkImageView views[kMaxColorAttachments * 2 + 1]; // colors, depth, resolves
        uint32_t viewCount, width, height;
        VkFramebuffer framebuffer;
    };

    struct PipelineLayoutEntry {
        VkDescriptorSetLayout setLayouts[shaderformat::kMaxSets];
        uint32_t setCount;
        VkPipelineLayout layout;
    };

    // init helpers (VulkanDevice.cpp)
    bool createInstance();
    bool pickPhysicalDevice();
    bool createLogicalDevice();
    bool createAllocatorAndPools();
    bool createFrames();
    bool createSwapchain();
    void destroySwapchainResources();
    bool createRings();
    void fillCaps();
    void loadPipelineCache();
    void savePipelineCache();
    void setObjectName(VkObjectType type, uint64_t handle, const char* name);

    // resources (VkResources.cpp)
    VkRenderPass getRenderPass(const RenderPassKey& key);
    RenderPassKey compatibleKey(const AttachmentLayout& layout) const;
    VkFramebuffer getFramebuffer(VkRenderPass pass, const VkImageView* views, uint32_t count, uint32_t w, uint32_t h);
    VkPipelineLayout getPipelineLayout(const PipelineDesc& desc);
    VkPipeline makeVkPipeline(const PipelineDesc& desc, VkPipelineLayout layout); // VK_NULL_HANDLE on failure
    VkDescriptorSet allocateDescriptorSet(VkDescriptorSetLayout layout, VkDescriptorPool& outPool);
    void defer(Deferred::Kind kind, uint64_t handle, uint64_t extra = 0, VmaAllocation allocation = nullptr);
    void runDeferred(bool all);
    void destroyDeferredNow(const Deferred& d);

    // commands (VkCommands.cpp)
    void translate(VkCommandBuffer cmd, const CommandList& list);
    void beginRenderPass(VkCommandBuffer cmd, const RenderPassDesc& desc);
    void flushUploads(VkCommandBuffer cmd);

    DeviceDesc m_desc;
    Caps m_caps;
    DeviceStats m_stats;

    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_messenger = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties m_props{};
    VkPhysicalDeviceFeatures m_enabledFeatures{};
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queueFamily = 0;
    VmaAllocator m_allocator = nullptr;
    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;
    bool m_lost = false;        // M9 plan D7: the device is lost; beginFrame fails and nothing waits on the GPU
    uint32_t m_framesBegun = 0; // for DeviceDesc::simulateLossAtFrame
    std::string m_pipelineCachePath; // <cache directory>vk_pipelines.bin; empty: no cache file (M9 plan D4)
    bool m_debugUtils = false;
    bool m_validationEnabled = false;

    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_swapFormat = VK_FORMAT_UNDEFINED;
    Format m_backbufferFormat = Format::Undefined;
    VkExtent2D m_swapExtent{};
    std::vector<VkImage> m_swapImages;
    std::vector<VkImageView> m_swapViews;
    std::vector<VkSemaphore> m_renderFinished; // one per swapchain image
    uint32_t m_imageIndex = 0;
    bool m_swapchainDirty = false;
    VkResult m_swapchainResult = VK_SUCCESS; // why the last createSwapchain failed (a lost device or surface, or memory)

    FrameData m_frames[kFramesInFlight];
    size_t m_commandBytes = 256 * 1024; // every command list holds at least this (reserveCommandBytes)
    uint32_t m_slot = 0;
    bool m_slotUsed = false;    // a frame was submitted in m_slot; the next beginFrame moves on
    bool m_frameActive = false; // between a successful beginFrame and submit
    uint64_t m_submitSerial = 0;
    uint64_t m_completedSerial = 0;
    VkPipelineLayout m_cmdLayout = VK_NULL_HANDLE; // layout of the last bound pipeline, kept across lists
    bool m_warnedUnboundLayout = false;
    std::vector<Deferred> m_deferred;

    TransientRing m_rings[uint32_t(TransientKind::Count)];
    BufferHandle m_ringBuffers[uint32_t(TransientKind::Count)];
    BufferHandle m_stagingBuffer;
    uint32_t m_stagingCapacity = 0; // per frame slot
    uint32_t m_stagingUsed = 0;
    std::vector<PendingCopy> m_pendingCopies;
    std::vector<PendingBufferCopy> m_pendingBufferCopies;

    std::vector<std::pair<RenderPassKey, VkRenderPass>> m_renderPasses;
    std::vector<FramebufferEntry> m_framebuffers;
    std::vector<PipelineLayoutEntry> m_pipelineLayouts;
    std::vector<VkDescriptorPool> m_descriptorPools;
    VkDescriptorSetLayout m_emptySetLayout = VK_NULL_HANDLE;

    core::HandlePool<BufferRes, BufferTag> m_buffers{4096};
    core::HandlePool<TextureRes, TextureTag> m_textures{4096};
    core::HandlePool<SamplerRes, SamplerTag> m_samplers{256};
    core::HandlePool<ShaderRes, ShaderTag> m_shaders{2048};
    core::HandlePool<PipelineRes, PipelineTag> m_pipelines{2048};
    core::HandlePool<SetLayoutRes, ResourceSetLayoutTag> m_setLayouts{256};
    core::HandlePool<ResourceSetRes, ResourceSetTag> m_resourceSets{8192};
    core::HandlePool<QueryPoolRes, QueryPoolTag> m_queryPools{64};
};

} // namespace rhi::vk
