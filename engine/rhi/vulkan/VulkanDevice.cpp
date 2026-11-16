#include "rhi/vulkan/VulkanDevice.h"

#include "core/Hash.h"
#include "core/Profile.h"
#include "platform/FileSystem.h"
#include "platform/Window.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace rhi::vk {
namespace {

constexpr uint32_t kUniformRingBytes = 4u << 20;
constexpr uint32_t kVertexRingBytes = 8u << 20;
constexpr uint32_t kIndexRingBytes = 1u << 20;
constexpr uint32_t kStagingBytes = 16u << 20;

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
    const bool isError = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
    static_cast<VulkanDevice*>(user)->onValidationMessage(isError, data && data->pMessage ? data->pMessage : "");
    return VK_FALSE;
}

bool hasExtension(const std::vector<VkExtensionProperties>& list, const char* name) {
    return std::any_of(list.begin(), list.end(),
                       [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
}

std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice pd) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> list(count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, list.data());
    return list;
}

} // namespace

core::Result<std::unique_ptr<Device>> createVulkanDevice(const DeviceDesc& desc) {
    auto device = std::make_unique<VulkanDevice>(desc);
    if (auto r = device->init(); !r) return r.error();
    return std::unique_ptr<Device>(std::move(device));
}

VulkanDevice::VulkanDevice(const DeviceDesc& desc) : m_desc(desc) {}

core::Result<void> VulkanDevice::init() {
    FL_ZONE("VulkanDevice::init");
    if (volkInitialize() != VK_SUCCESS) {
        FL_LOG_ERROR("Vulkan loader not found (vulkan-1). Is a Vulkan driver installed?");
        return core::Error::InitFailed;
    }
    if (!createInstance() || !pickPhysicalDevice() || !createLogicalDevice() || !createAllocatorAndPools() ||
        !createFrames() || !createRings())
        return core::Error::InitFailed;
    fillCaps();
    if (!createSwapchain()) return core::Error::InitFailed;
    FL_LOG_INFO("Vulkan device: %s (API %s), backbuffer %s %ux%u", m_caps.deviceName, m_caps.apiVersion,
                formatName(m_backbufferFormat), m_swapExtent.width, m_swapExtent.height);
    return {};
}

bool VulkanDevice::createInstance() {
    uint32_t apiVersion = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&apiVersion);
    if (apiVersion < VK_API_VERSION_1_1) {
        FL_LOG_ERROR("The Vulkan loader only supports %u.%u; 1.1 is required", VK_API_VERSION_MAJOR(apiVersion),
                     VK_API_VERSION_MINOR(apiVersion));
        return false;
    }

    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data());

    std::vector<const char*> extensions;
    for (const char* name : m_desc.window->vulkanInstanceExtensions()) extensions.push_back(name);
    m_debugUtils = hasExtension(available, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (m_debugUtils) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    std::vector<const char*> layers;
    bool layerSettings = false;
    if (m_desc.validation) {
        // Dev builds ship the layer next to the executable. VK_ADD_LAYER_PATH adds that directory to the loader's
        // search without hiding an installed SDK layer (process environment: set once, before enumeration).
        if (m_desc.bundledLayerPath && !std::getenv("VK_ADD_LAYER_PATH")) {
#if defined(_WIN32)
            _putenv_s("VK_ADD_LAYER_PATH", m_desc.bundledLayerPath);
#else
            setenv("VK_ADD_LAYER_PATH", m_desc.bundledLayerPath, 0);
#endif
        }
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> layerProps(count);
        vkEnumerateInstanceLayerProperties(&count, layerProps.data());
        const bool found = std::any_of(layerProps.begin(), layerProps.end(), [](const VkLayerProperties& l) {
            return std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0;
        });
        if (found) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            m_validationEnabled = true;
            uint32_t layerExtCount = 0;
            vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &layerExtCount, nullptr);
            std::vector<VkExtensionProperties> layerExts(layerExtCount);
            vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &layerExtCount, layerExts.data());
            layerSettings = hasExtension(layerExts, VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
            if (layerSettings) extensions.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
            FL_LOG_INFO("Vulkan validation layer enabled%s", std::getenv("VK_ADD_LAYER_PATH") ? " (bundled dev-build copy)" : "");
        } else {
            FL_LOG_WARN("Validation was requested but VK_LAYER_KHRONOS_validation isn't installed (install the "
                        "Vulkan SDK). Running without validation.");
        }
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "FlashLineage3D";
    app.pEngineName = "FlashLineage3D";
    app.apiVersion = VK_API_VERSION_1_1; // the baseline (ARCHITECTURE.md hard constraint)

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = uint32_t(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount = uint32_t(layers.size());
    ci.ppEnabledLayerNames = layers.data();
    // Synchronization validation is off by default in the layer; the acceptance criteria require it
    // (ARCHITECTURE.md §9 P13, §10).
    const VkBool32 enable = VK_TRUE;
    const VkLayerSettingEXT syncSetting{"VK_LAYER_KHRONOS_validation", "validate_sync", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &enable};
    VkLayerSettingsCreateInfoEXT settings{VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT};
    settings.settingCount = 1;
    settings.pSettings = &syncSetting;
    if (layerSettings) ci.pNext = &settings;
    if (!FL_VK_OK(vkCreateInstance(&ci, nullptr, &m_instance))) return false;
    if (m_validationEnabled) FL_LOG_INFO("Vulkan synchronization validation: %s", layerSettings ? "on" : "unavailable");
    volkLoadInstanceOnly(m_instance);

    if (m_debugUtils && m_validationEnabled) {
        VkDebugUtilsMessengerCreateInfoEXT mi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mi.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        mi.pfnUserCallback = debugCallback;
        mi.pUserData = this;
        FL_VK_OK(vkCreateDebugUtilsMessengerEXT(m_instance, &mi, nullptr, &m_messenger));
    }

    uint64_t surface = 0;
    if (!m_desc.window->createVulkanSurface(m_instance, &surface)) return false;
    m_surface = reinterpret_cast<VkSurfaceKHR>(surface);
    return true;
}

bool VulkanDevice::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    int bestScore = -1;
    for (VkPhysicalDevice pd : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.apiVersion < VK_API_VERSION_1_1) continue;
        if (!hasExtension(deviceExtensions(pd), VK_KHR_SWAPCHAIN_EXTENSION_NAME)) continue;

        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &familyCount, families.data());
        for (uint32_t f = 0; f < familyCount; ++f) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, f, m_surface, &present);
            if (!(families[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !present) continue;
            const int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 3
                              : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                                                                                           : 1;
            if (score > bestScore) {
                bestScore = score;
                m_physicalDevice = pd;
                m_queueFamily = f;
                m_props = props;
            }
            break;
        }
    }
    if (!m_physicalDevice) {
        FL_LOG_ERROR("No Vulkan 1.1 GPU with a graphics+present queue and VK_KHR_swapchain was found");
        return false;
    }
    return true;
}

bool VulkanDevice::createLogicalDevice() {
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(m_physicalDevice, &supported);
    m_enabledFeatures.samplerAnisotropy = supported.samplerAnisotropy;
    m_enabledFeatures.fillModeNonSolid = supported.fillModeNonSolid;
    m_enabledFeatures.textureCompressionBC = supported.textureCompressionBC;

    // Optional fast paths are only reported in M0; none is required or enabled (ARCHITECTURE.md §9, P14).
    const auto available = deviceExtensions(m_physicalDevice);
    FL_LOG_INFO("Vulkan optional extensions%s: dynamic_rendering=%d synchronization2=%d depth_stencil_resolve=%d",
                m_desc.vulkanBaseline ? " (baseline mode: none enabled)" : " (available, unused in M0)",
                hasExtension(available, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME),
                hasExtension(available, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME),
                hasExtension(available, VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME));

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = m_queueFamily;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;

    const char* extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qi;
    ci.enabledExtensionCount = 1;
    ci.ppEnabledExtensionNames = extensions;
    ci.pEnabledFeatures = &m_enabledFeatures;
    if (!FL_VK_OK(vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device))) return false;
    volkLoadDevice(m_device);
    vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);
    return true;
}

bool VulkanDevice::createAllocatorAndPools() {
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo ai{};
    ai.physicalDevice = m_physicalDevice;
    ai.device = m_device;
    ai.instance = m_instance;
    ai.vulkanApiVersion = VK_API_VERSION_1_1;
    ai.pVulkanFunctions = &functions;
    if (!FL_VK_OK(vmaCreateAllocator(&ai, &m_allocator))) return false;

    loadPipelineCache();

    VkDescriptorSetLayoutCreateInfo empty{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    return FL_VK_OK(vkCreateDescriptorSetLayout(m_device, &empty, nullptr, &m_emptySetLayout));
}

bool VulkanDevice::createFrames() {
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        FrameData& f = m_frames[i];
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pi.queueFamilyIndex = m_queueFamily;
        if (!FL_VK_OK(vkCreateCommandPool(m_device, &pi, nullptr, &f.commandPool))) return false;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = f.commandPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (!FL_VK_OK(vkAllocateCommandBuffers(m_device, &ai, &f.commandBuffer))) return false;
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (!FL_VK_OK(vkCreateFence(m_device, &fi, nullptr, &f.fence))) return false;
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (!FL_VK_OK(vkCreateSemaphore(m_device, &si, nullptr, &f.imageAvailable))) return false;
    }
    return true;
}

bool VulkanDevice::createRings() {
    const uint32_t uboAlign = uint32_t(std::max<VkDeviceSize>(m_props.limits.minUniformBufferOffsetAlignment, 16));
    struct RingSpec { TransientKind kind; uint32_t bytes; BufferUsage usage; uint32_t align; const char* name; };
    const RingSpec specs[] = {
        {TransientKind::Uniform, kUniformRingBytes, BufferUsage::Uniform, uboAlign, "TransientUniformRing"},
        {TransientKind::Vertex, kVertexRingBytes, BufferUsage::Vertex, 16, "TransientVertexRing"},
        {TransientKind::Index, kIndexRingBytes, BufferUsage::Index, 16, "TransientIndexRing"},
    };
    for (const RingSpec& s : specs) {
        BufferDesc bd;
        bd.debugName = s.name;
        bd.size = s.bytes * kFramesInFlight;
        bd.usage = s.usage;
        bd.memory = MemoryKind::Upload;
        const BufferHandle h = createBuffer(bd);
        if (!h) return false;
        BufferRes* ring = m_buffers.get(h);
        ring->ownedByRing = true;
        m_memory.move(ring->category, GpuMemoryCategory::Buffers, ring->size); // transient, not mesh memory (M9 plan D6)
        ring->category = GpuMemoryCategory::Buffers;
        m_ringBuffers[uint32_t(s.kind)] = h;
        m_rings[uint32_t(s.kind)].init(h, s.bytes, s.align);
    }
    BufferDesc staging;
    staging.debugName = "UploadStaging";
    staging.size = kStagingBytes * kFramesInFlight;
    staging.usage = BufferUsage::TransferSrc;
    staging.memory = MemoryKind::Upload;
    m_stagingBuffer = createBuffer(staging);
    if (!m_stagingBuffer) return false;
    m_buffers.get(m_stagingBuffer)->ownedByRing = true;
    m_stagingCapacity = kStagingBytes;
    return true;
}

void VulkanDevice::fillCaps() {
    Caps& c = m_caps;
    c.backend = Backend::Vulkan;
    c.clipSpace = {true, true};
    std::snprintf(c.deviceName, sizeof(c.deviceName), "%s", m_props.deviceName);
    std::snprintf(c.apiVersion, sizeof(c.apiVersion), "%u.%u.%u (driver %u)", VK_API_VERSION_MAJOR(m_props.apiVersion),
                  VK_API_VERSION_MINOR(m_props.apiVersion), VK_API_VERSION_PATCH(m_props.apiVersion),
                  m_props.driverVersion);
    queryFormatSupport(m_physicalDevice, c);
    c.bc1to5 = c.bptc = m_enabledFeatures.textureCompressionBC == VK_TRUE;
    c.linearDepthCompare = c.formats[uint32_t(Format::D16)].filterable;
    c.timestampQueries = m_props.limits.timestampComputeAndGraphics == VK_TRUE;
    c.wireframe = m_enabledFeatures.fillModeNonSolid == VK_TRUE;
    c.multisampleTextures = true;
    c.depthResolveExtension = false;
    c.programBinaryCache = true; // VkPipelineCache
    c.debugLabels = m_debugUtils;
    c.maxAnisotropy = m_enabledFeatures.samplerAnisotropy ? m_props.limits.maxSamplerAnisotropy : 1.0f;
    const VkSampleCountFlags samples =
        m_props.limits.framebufferColorSampleCounts & m_props.limits.framebufferDepthSampleCounts;
    c.maxMsaaSamples = (samples & VK_SAMPLE_COUNT_8_BIT) ? 8 : (samples & VK_SAMPLE_COUNT_4_BIT) ? 4 : (samples & VK_SAMPLE_COUNT_2_BIT) ? 2 : 1;
    c.uniformOffsetAlignment = uint32_t(m_props.limits.minUniformBufferOffsetAlignment);
    c.maxUniformBlockBytes = m_props.limits.maxUniformBufferRange;
    c.maxFragmentTextureUnits = m_props.limits.maxPerStageDescriptorSamplers;
    c.maxTextureSize = m_props.limits.maxImageDimension2D;
    c.depthFormat = c.formats[uint32_t(Format::D24S8)].depthTarget ? Format::D24S8 : Format::D32F;
}

bool VulkanDevice::createSwapchain() {
    VkSurfaceCapabilitiesKHR sc{};
    m_swapchainResult = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &sc);
    if (!FL_VK_OK(m_swapchainResult)) return false;

    VkExtent2D extent = sc.currentExtent;
    if (extent.width == std::numeric_limits<uint32_t>::max()) {
        extent.width = std::clamp(m_desc.window->drawableWidth(), sc.minImageExtent.width, sc.maxImageExtent.width);
        extent.height = std::clamp(m_desc.window->drawableHeight(), sc.minImageExtent.height, sc.maxImageExtent.height);
    }
    if (extent.width == 0 || extent.height == 0) {
        m_swapExtent = extent; // minimized: keep the old swapchain, render nothing
        m_swapchainDirty = true;
        return true;
    }

    // Fixed arrays, not vectors: the window system can make us rebuild the swapchain in any frame (suboptimal or out of
    // date), and a steady frame must not allocate. A device with more entries reports VK_INCOMPLETE; the first ones do.
    VkSurfaceFormatKHR formats[64];
    uint32_t count = 64;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &count, formats);
    VkSurfaceFormatKHR chosen = count ? formats[0] : VkSurfaceFormatKHR{};
    for (VkFormat preferred : {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB}) {
        const auto it = std::find_if(formats, formats + count, [&](const VkSurfaceFormatKHR& f) {
            return f.format == preferred && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });
        if (it != formats + count) {
            chosen = *it;
            break;
        }
    }

    VkPresentModeKHR modes[16];
    uint32_t modeCount = 16;
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &modeCount, modes);
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR; // always available; vsync
    if (!m_desc.window->vsync()) {
        for (VkPresentModeKHR m : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR})
            if (std::find(modes, modes + modeCount, m) != modes + modeCount) {
                mode = m;
                break;
            }
    }

    uint32_t imageCount = sc.minImageCount + 1;
    if (sc.maxImageCount > 0) imageCount = std::min(imageCount, sc.maxImageCount);

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = m_surface;
    ci.minImageCount = imageCount;
    ci.imageFormat = chosen.format;
    ci.imageColorSpace = chosen.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (sc.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = sc.currentTransform;
    ci.compositeAlpha = (sc.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                            ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                            : VkCompositeAlphaFlagBitsKHR(sc.supportedCompositeAlpha & (~sc.supportedCompositeAlpha + 1u));
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = m_swapchain;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    m_swapchainResult = vkCreateSwapchainKHR(m_device, &ci, nullptr, &swapchain);
    if (!FL_VK_OK(m_swapchainResult)) return false;
    destroySwapchainResources();
    if (m_swapchain) {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        ++m_stats.swapchainRecreations; // a rebuild, not the first creation
    }
    m_swapchain = swapchain;
    m_swapFormat = chosen.format;
    m_backbufferFormat = fromVkFormat(chosen.format);
    m_caps.srgbBackbuffer = formatInfo(m_backbufferFormat).srgb;
    m_swapExtent = extent;

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &count, nullptr);
    m_swapImages.resize(count);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &count, m_swapImages.data());
    m_swapViews.resize(count);
    m_renderFinished.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = m_swapImages[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = m_swapFormat;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!FL_VK_OK(vkCreateImageView(m_device, &vi, nullptr, &m_swapViews[i]))) return false;
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (!FL_VK_OK(vkCreateSemaphore(m_device, &si, nullptr, &m_renderFinished[i]))) return false;
    }
    m_swapchainDirty = false;
    return true;
}

// Only called with the device idle (swapchain recreation, shutdown).
void VulkanDevice::destroySwapchainResources() {
    for (auto it = m_framebuffers.begin(); it != m_framebuffers.end();) {
        const bool usesSwapchain = std::any_of(it->views, it->views + it->viewCount, [&](VkImageView v) {
            return std::find(m_swapViews.begin(), m_swapViews.end(), v) != m_swapViews.end();
        });
        if (usesSwapchain) {
            vkDestroyFramebuffer(m_device, it->framebuffer, nullptr);
            it = m_framebuffers.erase(it);
        } else {
            ++it;
        }
    }
    for (VkImageView v : m_swapViews) vkDestroyImageView(m_device, v, nullptr);
    for (VkSemaphore s : m_renderFinished) vkDestroySemaphore(m_device, s, nullptr);
    m_swapViews.clear();
    m_renderFinished.clear();
    m_swapImages.clear();
}

namespace {

// The pipeline cache file (M9 plan D4): this header, then the driver's data. The header pins the GPU and driver the data
// came from (the driver's own header lacks the driver version) and lets a torn or foreign file be told apart before the
// driver ever sees it.
struct PipelineCacheFileHeader {
    char magic[4];    // "FLPC"
    uint32_t version; // kPipelineCacheFileVersion
    uint32_t vendorID, deviceID, driverVersion;
    uint8_t cacheUUID[VK_UUID_SIZE];
    uint64_t dataSize, dataHash; // FNV-1a 64 of the data
};
constexpr uint32_t kPipelineCacheFileVersion = 1;

} // namespace

void VulkanDevice::loadPipelineCache() {
    std::vector<std::byte> data;
    if (m_desc.cacheDirectory) {
        m_pipelineCachePath = std::string(m_desc.cacheDirectory) + "vk_pipelines.bin";
        if (auto file = platform::readFile(m_pipelineCachePath); file.ok()) {
            const std::vector<std::byte>& bytes = file.value();
            PipelineCacheFileHeader header{};
            const char* rejected = nullptr;
            if (bytes.size() < sizeof(header)) {
                rejected = "too short";
            } else {
                std::memcpy(&header, bytes.data(), sizeof(header));
                const std::span<const std::byte> payload(bytes.data() + sizeof(header), bytes.size() - sizeof(header));
                if (std::memcmp(header.magic, "FLPC", 4) != 0 || header.version != kPipelineCacheFileVersion)
                    rejected = "not a pipeline cache of this version";
                else if (header.vendorID != m_props.vendorID || header.deviceID != m_props.deviceID ||
                         header.driverVersion != m_props.driverVersion ||
                         std::memcmp(header.cacheUUID, m_props.pipelineCacheUUID, VK_UUID_SIZE) != 0)
                    rejected = "written by another GPU or driver";
                else if (header.dataSize != payload.size() || core::hash64(payload) != header.dataHash)
                    rejected = "damaged";
                else
                    data.assign(payload.begin(), payload.end());
            }
            if (rejected) FL_LOG_INFO("Ignoring the Vulkan pipeline cache %s: %s", m_pipelineCachePath.c_str(), rejected);
        }
    }
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size();
    ci.pInitialData = data.empty() ? nullptr : data.data();
    if (!FL_VK_OK(vkCreatePipelineCache(m_device, &ci, nullptr, &m_pipelineCache))) m_pipelineCache = VK_NULL_HANDLE;
    else if (!data.empty()) FL_LOG_INFO("Loaded the Vulkan pipeline cache (%zu bytes)", data.size());
}

void VulkanDevice::savePipelineCache() {
    if (!m_pipelineCache || m_pipelineCachePath.empty()) return;
    size_t size = 0;
    if (vkGetPipelineCacheData(m_device, m_pipelineCache, &size, nullptr) != VK_SUCCESS || size == 0) return;
    constexpr size_t kHeader = sizeof(PipelineCacheFileHeader);
    std::vector<std::byte> file(kHeader + size);
    if (vkGetPipelineCacheData(m_device, m_pipelineCache, &size, file.data() + kHeader) != VK_SUCCESS) return;
    file.resize(kHeader + size);
    PipelineCacheFileHeader header{};
    std::memcpy(header.magic, "FLPC", 4);
    header.version = kPipelineCacheFileVersion;
    header.vendorID = m_props.vendorID;
    header.deviceID = m_props.deviceID;
    header.driverVersion = m_props.driverVersion;
    std::memcpy(header.cacheUUID, m_props.pipelineCacheUUID, VK_UUID_SIZE);
    header.dataSize = size;
    header.dataHash = core::hash64(std::span<const std::byte>(file.data() + kHeader, size));
    std::memcpy(file.data(), &header, kHeader);
    if (!platform::createDirectories(m_desc.cacheDirectory) || !platform::writeFileAtomic(m_pipelineCachePath, file).ok())
        FL_LOG_WARN("Could not write the pipeline cache to %s", m_pipelineCachePath.c_str());
}

void VulkanDevice::setObjectName(VkObjectType type, uint64_t handle, const char* name) {
    if (!m_debugUtils || !name || !*name || !vkSetDebugUtilsObjectNameEXT) return;
    VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name;
    vkSetDebugUtilsObjectNameEXT(m_device, &info);
}

void VulkanDevice::onValidationMessage(bool isError, const char* message) {
    if (isError) {
        ++m_stats.validationErrors;
        FL_LOG_ERROR("[vulkan] %s", message);
    } else {
        ++m_stats.validationWarnings;
        FL_LOG_WARN("[vulkan] %s", message);
    }
}

VulkanDevice::~VulkanDevice() {
    if (!m_device) {
        if (m_surface) vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        if (m_messenger) vkDestroyDebugUtilsMessengerEXT(m_instance, m_messenger, nullptr);
        if (m_instance) vkDestroyInstance(m_instance, nullptr);
        return;
    }
    // M9 plan D7: nothing waits on a lost device. A device the driver declared lost returns at once anyway; one marked lost
    // by the 10 s fence timeout may still be hung (TDR off), and waiting would freeze the shutdown.
    if (!m_lost) vkDeviceWaitIdle(m_device);
    if (!m_lost) savePipelineCache(); // a lost device's cache data isn't worth keeping
    runDeferred(true);

    m_resourceSets.forEach([&](auto, ResourceSetRes& r) { vkFreeDescriptorSets(m_device, r.pool, 1, &r.set); });
    m_pipelines.forEach([&](auto, PipelineRes& r) { vkDestroyPipeline(m_device, r.pipeline, nullptr); });
    m_setLayouts.forEach([&](auto, SetLayoutRes& r) { vkDestroyDescriptorSetLayout(m_device, r.layout, nullptr); });
    m_shaders.forEach([&](auto, ShaderRes& r) { vkDestroyShaderModule(m_device, r.module, nullptr); });
    m_samplers.forEach([&](auto, SamplerRes& r) { vkDestroySampler(m_device, r.sampler, nullptr); });
    m_textures.forEach([&](auto, TextureRes& r) {
        for (uint32_t l = 0; l < r.layerViewCount; ++l) vkDestroyImageView(m_device, r.layerViews[l], nullptr);
        vkDestroyImageView(m_device, r.view, nullptr);
        vmaDestroyImage(m_allocator, r.image, r.allocation);
    });
    m_buffers.forEach([&](auto, BufferRes& r) { vmaDestroyBuffer(m_allocator, r.buffer, r.allocation); });
    m_queryPools.forEach([&](auto, QueryPoolRes& r) {
        for (VkQueryPool p : r.pools) vkDestroyQueryPool(m_device, p, nullptr);
    });

    destroySwapchainResources();
    for (FramebufferEntry& f : m_framebuffers) vkDestroyFramebuffer(m_device, f.framebuffer, nullptr);
    for (auto& [key, pass] : m_renderPasses) vkDestroyRenderPass(m_device, pass, nullptr);
    for (PipelineLayoutEntry& p : m_pipelineLayouts) vkDestroyPipelineLayout(m_device, p.layout, nullptr);
    for (VkDescriptorPool p : m_descriptorPools) vkDestroyDescriptorPool(m_device, p, nullptr);
    if (m_emptySetLayout) vkDestroyDescriptorSetLayout(m_device, m_emptySetLayout, nullptr);
    if (m_swapchain) vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    for (FrameData& f : m_frames) {
        vkDestroyCommandPool(m_device, f.commandPool, nullptr);
        vkDestroyFence(m_device, f.fence, nullptr);
        vkDestroySemaphore(m_device, f.imageAvailable, nullptr);
    }
    if (m_pipelineCache) vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
    if (m_allocator) vmaDestroyAllocator(m_allocator);
    vkDestroyDevice(m_device, nullptr);
    if (m_surface) vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    if (m_messenger) vkDestroyDebugUtilsMessengerEXT(m_instance, m_messenger, nullptr);
    vkDestroyInstance(m_instance, nullptr);
}

core::Result<bool> VulkanDevice::beginFrame() {
    FL_ZONE("VulkanDevice::beginFrame");
    // Minimized: nothing to render to. Return before touching fences or the swapchain (no GPU stalls).
    if (m_desc.window->drawableWidth() == 0 || m_desc.window->drawableHeight() == 0) return false;
    // M9 plan D7: a lost device stays lost; nothing below may wait on it.
    if (m_desc.simulateLossAtFrame && ++m_framesBegun >= m_desc.simulateLossAtFrame && !m_lost) {
        FL_LOG_ERROR("Simulated device loss at frame %u", m_framesBegun);
        m_lost = true;
    }
    if (m_lost) return core::Error::DeviceLost;
    if (m_slotUsed) {
        m_slot = (m_slot + 1) % kFramesInFlight;
        m_slotUsed = false;
    }
    FrameData& f = m_frames[m_slot];
    const VkResult waited = vkWaitForFences(m_device, 1, &f.fence, VK_TRUE, kGpuHangTimeoutNs);
    if (waited != VK_SUCCESS) {
        FL_LOG_ERROR("The GPU device is lost: %s", waited == VK_TIMEOUT ? "a frame didn't finish within 10 s" : resultName(waited));
        m_lost = true;
        return core::Error::DeviceLost;
    }
    for (const FrameData& frame : m_frames)
        if (vkGetFenceStatus(m_device, frame.fence) == VK_SUCCESS)
            m_completedSerial = std::max(m_completedSerial, frame.serial);
    runDeferred(false);

    // M9 plan D7: a lost device or surface ends rendering for good (nothing waits on the GPU from then on); any other
    // failure (out of memory, the window in use elsewhere) fails this frame without counting as a device loss.
    auto failed = [&](VkResult r) -> core::Result<bool> {
        if (r == VK_ERROR_DEVICE_LOST || r == VK_ERROR_SURFACE_LOST_KHR) {
            m_lost = true;
            return core::Error::DeviceLost;
        }
        return r == VK_ERROR_OUT_OF_HOST_MEMORY || r == VK_ERROR_OUT_OF_DEVICE_MEMORY ? core::Error::OutOfMemory : core::Error::InitFailed;
    };
    const bool sizeChanged = m_desc.window->drawableWidth() != m_swapExtent.width ||
                             m_desc.window->drawableHeight() != m_swapExtent.height;
    if (m_swapchainDirty || sizeChanged) {
        vkDeviceWaitIdle(m_device);
        if (!createSwapchain()) return failed(m_swapchainResult);
    }
    if (m_swapExtent.width == 0 || m_swapExtent.height == 0) return false;

    VkResult r = vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX, f.imageAvailable, VK_NULL_HANDLE, &m_imageIndex);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        vkDeviceWaitIdle(m_device);
        if (!createSwapchain()) return failed(m_swapchainResult);
        if (m_swapExtent.width == 0 || m_swapExtent.height == 0) return false;
        r = vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX, f.imageAvailable, VK_NULL_HANDLE, &m_imageIndex);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) { // resized again in between: try again next frame
            m_swapchainDirty = true;
            return false;
        }
    }
    if (r == VK_SUBOPTIMAL_KHR) m_swapchainDirty = true;
    else if (r != VK_SUCCESS) {
        FL_LOG_ERROR("vkAcquireNextImageKHR failed: %s", resultName(r));
        return failed(r);
    }

    vkResetFences(m_device, 1, &f.fence);
    vkResetCommandPool(m_device, f.commandPool, 0);
    for (uint32_t k = 0; k < uint32_t(TransientKind::Count); ++k) {
        BufferRes* b = m_buffers.get(m_ringBuffers[k]);
        const uint32_t base = m_rings[k].capacityBytes() * m_slot;
        m_rings[k].beginFrame(static_cast<std::byte*>(b->mapped) + base, base);
    }
    m_stagingUsed = 0;
    f.commandListsUsed = 0;
    m_frameActive = true;
    return true;
}

void VulkanDevice::reserveCommandBytes(size_t bytes) {
    if (bytes <= m_commandBytes) return;
    m_commandBytes = bytes;
    m_stats.commandListBytes = bytes;
    for (FrameData& frame : m_frames)
        for (auto& list : frame.commandLists) list->reserve(bytes);
}

CommandList* VulkanDevice::allocateCommandList() {
    FrameData& f = m_frames[m_slot];
    if (f.commandListsUsed == f.commandLists.size()) f.commandLists.push_back(std::make_unique<CommandList>(m_commandBytes));
    CommandList* list = f.commandLists[f.commandListsUsed++].get();
    list->reset();
    return list;
}

// Copies bytes into this frame slot's staging partition; outOffset is the absolute staging-buffer offset.
// Only inside beginFrame..submit (the slot's previous GPU use has finished); otherwise, or when the data
// doesn't fit, the caller falls back to uploadImmediately().
bool VulkanDevice::stageBytes(const void* data, uint32_t bytes, uint32_t& outOffset) {
    if (!m_frameActive) return false;
    const uint32_t offset = (m_stagingUsed + 15u) & ~15u;
    if (uint64_t(offset) + bytes > m_stagingCapacity) return false;
    BufferRes* staging = m_buffers.get(m_stagingBuffer);
    const uint32_t base = m_stagingCapacity * m_slot;
    std::memcpy(static_cast<std::byte*>(staging->mapped) + base + offset, data, bytes);
    m_stagingUsed = offset + bytes;
    outOffset = base + offset;
    return true;
}

// Load-time or oversized uploads: a temporary staging buffer and a one-shot command buffer, then a full wait.
// Slow by design; per-frame streaming goes through the staging ring instead.
bool VulkanDevice::uploadImmediately(const void* data, uint32_t bytes, const PendingCopy* image,
                                     const PendingBufferCopy* buffer) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo info{};
    if (!FL_VK_OK(vmaCreateBuffer(m_allocator, &bi, &ai, &staging, &allocation, &info))) return false;
    std::memcpy(info.pMappedData, data, bytes);
    vmaFlushAllocation(m_allocator, allocation, 0, VK_WHOLE_SIZE);

    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pi.queueFamilyIndex = m_queueFamily;
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(m_device, &pi, nullptr, &pool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(m_device, &cai, &cmd);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    // Copies already staged this frame must land first: one of them may be the first upload (and layout
    // initialization) of the image this copy writes into.
    if (!m_pendingCopies.empty() || !m_pendingBufferCopies.empty()) {
        vmaFlushAllocation(m_allocator, m_buffers.get(m_stagingBuffer)->allocation, 0, VK_WHOLE_SIZE);
        flushUploads(cmd);
    }
    recordCopies(cmd, staging, image ? std::span(image, 1) : std::span<const PendingCopy>(),
                 buffer ? std::span(buffer, 1) : std::span<const PendingBufferCopy>());
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    const bool ok = FL_VK_OK(vkQueueSubmit(m_queue, 1, &si, VK_NULL_HANDLE));
    vkQueueWaitIdle(m_queue);
    vkDestroyCommandPool(m_device, pool, nullptr);
    vmaDestroyBuffer(m_allocator, staging, allocation);
    return ok;
}

core::Result<void> VulkanDevice::uploadTexture(TextureHandle handle, const TextureUpload& upload) {
    TextureRes* t = m_textures.get(handle);
    if (!t) return core::Error::NotFound;
    const FormatInfo fi = formatInfo(t->desc.format);
    // Compressed rows are whole blocks: a 2×2 mip of a BC texture still occupies one 4×4 block, and
    // bufferRowLength must be a multiple of the block width.
    const uint32_t texels = upload.rowLength ? upload.rowLength : upload.width;
    const uint32_t rowLength = (texels + fi.blockSize - 1) / fi.blockSize * fi.blockSize;
    const uint64_t bytes = uint64_t(rowLength / fi.blockSize) * ((upload.height + fi.blockSize - 1) / fi.blockSize) * fi.blockBytes;
    if (upload.data.size() < bytes) return core::Error::InvalidData;
    PendingCopy pc{};
    pc.image = t->image;
    pc.aspect = t->aspect;
    pc.initializeImage = t->layout == VK_IMAGE_LAYOUT_UNDEFINED;
    pc.mipLevels = t->desc.mips;
    pc.arrayLayers = t->desc.type == TextureType::Cube ? 6u : t->desc.layers;
    pc.oldLayout = pc.initializeImage ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : t->layout;
    pc.region.bufferRowLength = rowLength;
    pc.region.imageSubresource = {t->aspect, upload.mip, upload.layer, 1};
    pc.region.imageOffset = {int32_t(upload.x), int32_t(upload.y), 0};
    pc.region.imageExtent = {upload.width, upload.height, 1};
    uint32_t stagingOffset = 0;
    if (stageBytes(upload.data.data(), uint32_t(bytes), stagingOffset)) {
        pc.region.bufferOffset = stagingOffset;
        m_pendingCopies.push_back(pc);
    } else if (!uploadImmediately(upload.data.data(), uint32_t(bytes), &pc, nullptr)) {
        return core::Error::OutOfMemory;
    }
    t->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    m_stats.uploadedBytes += bytes;
    return {};
}

void VulkanDevice::submit(std::span<CommandList* const> lists) {
    FL_ZONE("VulkanDevice::submit");
    FrameData& f = m_frames[m_slot];
    m_cmdLayout = VK_NULL_HANDLE;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.commandBuffer, &bi);

    m_queryPools.forEach([&](auto, QueryPoolRes& q) {
        vkCmdResetQueryPool(f.commandBuffer, q.pools[m_slot], 0, q.count);
        q.submitted[m_slot] = true;
    });
    flushUploads(f.commandBuffer);

    m_stats.drawCalls = 0;
    for (CommandList* list : lists) {
        translate(f.commandBuffer, *list);
        m_stats.drawCalls += list->drawCount();
    }
    vkEndCommandBuffer(f.commandBuffer);

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &f.imageAvailable;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.commandBuffer;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &m_renderFinished[m_imageIndex];

    // Host writes must be visible on non-coherent memory types (no-op where memory is coherent).
    for (uint32_t k = 0; k < uint32_t(TransientKind::Count); ++k)
        if (const uint32_t used = std::min(m_rings[k].usedBytes(), m_rings[k].capacityBytes()))
            vmaFlushAllocation(m_allocator, m_buffers.get(m_ringBuffers[k])->allocation, m_rings[k].gpuBaseOffset(), used);
    if (m_stagingUsed)
        vmaFlushAllocation(m_allocator, m_buffers.get(m_stagingBuffer)->allocation, VkDeviceSize(m_stagingCapacity) * m_slot,
                           m_stagingUsed);

    // A failed submit leaves this slot's fence (reset in beginFrame) unsignaled for good: waiting on it would hang, so the
    // device counts as lost (M9 plan D7).
    if (const VkResult submitted = vkQueueSubmit(m_queue, 1, &si, f.fence); submitted != VK_SUCCESS) {
        FL_LOG_ERROR("vkQueueSubmit failed (%s): the GPU device is lost", resultName(submitted));
        m_lost = true;
    }
    f.serial = ++m_submitSerial;
    m_slotUsed = true;
    m_frameActive = false;
}

void VulkanDevice::present() {
    FL_ZONE("VulkanDevice::present");
    if (m_lost) return;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &m_renderFinished[m_imageIndex];
    pi.swapchainCount = 1;
    pi.pSwapchains = &m_swapchain;
    pi.pImageIndices = &m_imageIndex;
    const VkResult r = vkQueuePresentKHR(m_queue, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) m_swapchainDirty = true;
    else if (r != VK_SUCCESS) {
        FL_LOG_ERROR("vkQueuePresentKHR failed: %s", resultName(r));
        m_lost = m_lost || r == VK_ERROR_DEVICE_LOST || r == VK_ERROR_SURFACE_LOST_KHR;
    }
}

void VulkanDevice::readTimestamps(QueryPoolHandle handle, std::span<double> out) {
    std::fill(out.begin(), out.end(), std::nan(""));
    QueryPoolRes* q = m_queryPools.get(handle);
    if (!q || !q->submitted[m_slot]) return;
    const uint32_t n = std::min<uint32_t>(q->count, uint32_t(out.size()));
    const VkResult r = vkGetQueryPoolResults(m_device, q->pools[m_slot], 0, n, n * 2 * sizeof(uint64_t), q->scratch.data(),
                                             2 * sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (r != VK_SUCCESS && r != VK_NOT_READY) return;
    const double msPerTick = double(m_props.limits.timestampPeriod) * 1e-6;
    for (uint32_t i = 0; i < n; ++i)
        if (q->scratch[i * 2 + 1] != 0) out[i] = double(q->scratch[i * 2]) * msPerTick;
}

void VulkanDevice::waitIdle() {
    if (m_device && !m_lost) vkDeviceWaitIdle(m_device); // see the destructor
}

uint64_t VulkanDevice::allocatorBytes() const {
    if (!m_allocator) return 0;
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
    vmaGetHeapBudgets(m_allocator, budgets);
    const VkPhysicalDeviceMemoryProperties* properties = nullptr;
    vmaGetMemoryProperties(m_allocator, &properties);
    uint64_t bytes = 0;
    for (uint32_t h = 0; properties && h < properties->memoryHeapCount; ++h) bytes += budgets[h].statistics.blockBytes;
    return bytes;
}

core::Result<void> VulkanDevice::readbackBackbuffer(std::vector<uint8_t>& out, Extent2D& extent) {
    vkQueueWaitIdle(m_queue); // the frame's render pass left the image in PRESENT_SRC
    const uint32_t w = m_swapExtent.width, h = m_swapExtent.height;
    extent = {w, h};

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = VkDeviceSize(w) * h * 4;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo info{};
    if (!FL_VK_OK(vmaCreateBuffer(m_allocator, &bi, &ai, &buffer, &allocation, &info))) return core::Error::OutOfMemory;

    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pi.queueFamilyIndex = m_queueFamily;
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(m_device, &pi, nullptr, &pool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(m_device, &cai, &cmd);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_swapImages[m_imageIndex];
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(cmd, m_swapImages[m_imageIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(m_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_queue);

    vmaInvalidateAllocation(m_allocator, allocation, 0, VK_WHOLE_SIZE);
    const auto* src = static_cast<const uint8_t*>(info.pMappedData);
    const bool bgra = m_swapFormat == VK_FORMAT_B8G8R8A8_SRGB || m_swapFormat == VK_FORMAT_B8G8R8A8_UNORM;
    out.resize(size_t(w) * h * 4);
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        out[i * 4 + 0] = src[i * 4 + (bgra ? 2 : 0)];
        out[i * 4 + 1] = src[i * 4 + 1];
        out[i * 4 + 2] = src[i * 4 + (bgra ? 0 : 2)];
        out[i * 4 + 3] = src[i * 4 + 3];
    }
    vkDestroyCommandPool(m_device, pool, nullptr);
    vmaDestroyBuffer(m_allocator, buffer, allocation);
    return {};
}

} // namespace rhi::vk
