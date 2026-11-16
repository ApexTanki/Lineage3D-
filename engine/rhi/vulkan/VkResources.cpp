#include "rhi/vulkan/VulkanDevice.h"

#include "platform/Timer.h"

#include <shaderformat/BindingModel.h>

#include <algorithm>

namespace rhi::vk {
namespace {

VkBufferUsageFlags toVkBufferUsage(BufferUsage u) {
    VkBufferUsageFlags f = 0;
    if (any(u & BufferUsage::Vertex)) f |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (any(u & BufferUsage::Index)) f |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (any(u & BufferUsage::Uniform)) f |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (any(u & BufferUsage::TransferSrc)) f |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (any(u & BufferUsage::TransferDst)) f |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    return f;
}

VkShaderStageFlags toVkStages(ShaderStageMask m) {
    VkShaderStageFlags f = 0;
    if (any(m & ShaderStageMask::Vertex)) f |= VK_SHADER_STAGE_VERTEX_BIT;
    if (any(m & ShaderStageMask::Fragment)) f |= VK_SHADER_STAGE_FRAGMENT_BIT;
    return f;
}

VkAttachmentLoadOp toVkLoad(LoadOp op) {
    switch (op) {
    case LoadOp::Load: return VK_ATTACHMENT_LOAD_OP_LOAD;
    case LoadOp::Clear: return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case LoadOp::DontCare: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }
    return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
}

VkAttachmentStoreOp toVkStore(StoreOp op) {
    return op == StoreOp::Store ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
}

constexpr VkShaderStageFlags kPushStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

} // namespace

// ---------------------------------------------------------------- buffers, textures, samplers, shaders

BufferHandle VulkanDevice::createBuffer(const BufferDesc& desc) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = desc.size;
    bi.usage = toVkBufferUsage(desc.usage);
    if (desc.memory == MemoryKind::GpuOnly) bi.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    if (desc.memory == MemoryKind::Upload)
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    else if (desc.memory == MemoryKind::Readback)
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    BufferRes res;
    res.size = desc.size;
    res.category = memoryCategory(desc);
    if (!m_memory.reserve(res.category, desc.size, desc.debugName)) return {}; // over its budget (M9 plan D6)
    MemoryReservation reservation(m_memory, res.category, desc.size);
    VmaAllocationInfo info{};
    if (!FL_VK_OK(vmaCreateBuffer(m_allocator, &bi, &ai, &res.buffer, &res.allocation, &info))) return {};
    res.mapped = info.pMappedData;
    setObjectName(VK_OBJECT_TYPE_BUFFER, uint64_t(res.buffer), desc.debugName);

    bool copyQueued = false;
    if (desc.initialData) {
        if (res.mapped) {
            std::memcpy(res.mapped, desc.initialData, desc.size);
            vmaFlushAllocation(m_allocator, res.allocation, 0, VK_WHOLE_SIZE); // no-op on coherent memory
        } else {
            uint32_t offset = 0;
            const PendingBufferCopy immediate{res.buffer, VkBufferCopy{0, 0, desc.size}};
            if (stageBytes(desc.initialData, desc.size, offset)) {
                m_pendingBufferCopies.push_back({res.buffer, VkBufferCopy{offset, 0, desc.size}});
                copyQueued = true;
            } else if (!uploadImmediately(desc.initialData, desc.size, nullptr, &immediate)) {
                vmaDestroyBuffer(m_allocator, res.buffer, res.allocation);
                return {};
            }
        }
    }
    const BufferHandle h = m_buffers.create(res);
    if (!h) {
        FL_LOG_ERROR("Buffer pool exhausted");
        if (copyQueued) m_pendingBufferCopies.pop_back(); // the next submit must not copy into the buffer destroyed below
        vmaDestroyBuffer(m_allocator, res.buffer, res.allocation);
        return h;
    }
    reservation.keep();
    return h;
}

TextureHandle VulkanDevice::createTexture(const TextureDesc& desc) {
    const GpuMemoryCategory category = memoryCategory(desc);
    const uint64_t memory = memoryBytes(desc);
    if (!m_memory.reserve(category, memory, desc.debugName)) return {}; // over its budget (M9 plan D6)
    MemoryReservation reservation(m_memory, category, memory);
    TextureRes res;
    res.desc = desc;
    res.format = toVkFormat(desc.format);
    res.aspect = formatInfo(desc.format).depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = res.format;
    ii.extent = {desc.width, desc.height, 1};
    ii.mipLevels = desc.mips;
    ii.arrayLayers = desc.type == TextureType::Cube ? 6u : desc.layers;
    ii.samples = VkSampleCountFlagBits(desc.samples);
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    if (desc.type == TextureType::Cube) ii.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    if (any(desc.usage & TextureUsage::Sampled)) ii.usage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (any(desc.usage & TextureUsage::ColorTarget)) ii.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (any(desc.usage & TextureUsage::DepthTarget)) ii.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (any(desc.usage & TextureUsage::TransferSrc)) ii.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (any(desc.usage & TextureUsage::TransferDst)) ii.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    if (!FL_VK_OK(vmaCreateImage(m_allocator, &ii, &ai, &res.image, &res.allocation, nullptr))) return {};

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = res.image;
    vi.viewType = desc.type == TextureType::Cube         ? VK_IMAGE_VIEW_TYPE_CUBE
                  : desc.type == TextureType::Tex2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                         : VK_IMAGE_VIEW_TYPE_2D;
    vi.format = res.format;
    vi.subresourceRange = {res.aspect, 0, desc.mips, 0, ii.arrayLayers};
    if (!FL_VK_OK(vkCreateImageView(m_device, &vi, nullptr, &res.view))) {
        vmaDestroyImage(m_allocator, res.image, res.allocation);
        return {};
    }
    auto destroyViews = [&] {
        for (uint32_t l = 0; l < res.layerViewCount; ++l) vkDestroyImageView(m_device, res.layerViews[l], nullptr);
        vkDestroyImageView(m_device, res.view, nullptr);
        vmaDestroyImage(m_allocator, res.image, res.allocation);
    };
    // Render passes attach one layer of an array target (shadow cascades): one 2D view per layer, mip 0.
    if (desc.type == TextureType::Tex2DArray && any(desc.usage & (TextureUsage::ColorTarget | TextureUsage::DepthTarget))) {
        if (desc.layers > kMaxLayerViews) {
            FL_LOG_ERROR("Texture %s: an array render target can have at most %u layers", desc.debugName, kMaxLayerViews);
            destroyViews();
            return {};
        }
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        for (uint32_t l = 0; l < desc.layers; ++l) {
            vi.subresourceRange = {res.aspect, 0, 1, l, 1};
            if (!FL_VK_OK(vkCreateImageView(m_device, &vi, nullptr, &res.layerViews[l]))) {
                destroyViews();
                return {};
            }
            res.layerViewCount = l + 1;
        }
    }
    setObjectName(VK_OBJECT_TYPE_IMAGE, uint64_t(res.image), desc.debugName);
    const TextureHandle h = m_textures.create(res);
    if (!h) {
        FL_LOG_ERROR("Texture pool exhausted");
        destroyViews();
        return h;
    }
    reservation.keep();
    return h;
}

SamplerHandle VulkanDevice::createSampler(const SamplerDesc& desc) {
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = desc.filter == Filter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = desc.mip == MipMode::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = toVkAddress(desc.address);
    const float aniso = std::min(desc.maxAnisotropy, m_caps.maxAnisotropy);
    si.anisotropyEnable = aniso > 1.0f ? VK_TRUE : VK_FALSE;
    si.maxAnisotropy = std::max(aniso, 1.0f);
    si.compareEnable = desc.compare != CompareOp::Never ? VK_TRUE : VK_FALSE;
    si.compareOp = toVkCompare(desc.compare);
    si.maxLod = VK_LOD_CLAMP_NONE;
    SamplerRes res;
    if (!FL_VK_OK(vkCreateSampler(m_device, &si, nullptr, &res.sampler))) return {};
    return m_samplers.create(res);
}

ShaderHandle VulkanDevice::createShader(const ShaderDesc& desc) {
    if (desc.code.empty() || desc.code.size() % 4 != 0) {
        FL_LOG_ERROR("Shader %s has no valid SPIR-V", desc.debugName);
        return {};
    }
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = desc.code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(desc.code.data());
    ShaderRes res;
    if (!FL_VK_OK(vkCreateShaderModule(m_device, &ci, nullptr, &res.module))) return {};
    setObjectName(VK_OBJECT_TYPE_SHADER_MODULE, uint64_t(res.module), desc.debugName);
    return m_shaders.create(res);
}

// ---------------------------------------------------------------- render passes, framebuffers, layouts

VulkanDevice::RenderPassKey VulkanDevice::compatibleKey(const AttachmentLayout& layout) const {
    RenderPassKey key{};
    key.colorCount = layout.colorCount;
    for (uint32_t i = 0; i < layout.colorCount; ++i) {
        key.color[i] = toVkFormat(layout.color[i]);
        key.colorLoad[i] = uint8_t(LoadOp::DontCare);
        key.colorStore[i] = uint8_t(StoreOp::Store);
    }
    key.depth = toVkFormat(layout.depth);
    key.depthLoad = uint8_t(LoadOp::DontCare);
    key.depthStore = uint8_t(StoreOp::DontCare);
    key.samples = layout.samples;
    return key;
}

VkRenderPass VulkanDevice::getRenderPass(const RenderPassKey& key) {
    for (auto& [k, pass] : m_renderPasses)
        if (k == key) return pass;

    VkAttachmentDescription attachments[kMaxColorAttachments * 2 + 1]{};
    VkAttachmentReference colorRefs[kMaxColorAttachments]{};
    VkAttachmentReference resolveRefs[kMaxColorAttachments]{};
    bool anyResolve = false;
    uint32_t count = 0;
    for (uint32_t i = 0; i < key.colorCount; ++i) {
        VkAttachmentDescription& a = attachments[count];
        a.format = key.color[i];
        a.samples = VkSampleCountFlagBits(key.samples);
        a.loadOp = toVkLoad(LoadOp(key.colorLoad[i]));
        a.storeOp = toVkStore(StoreOp(key.colorStore[i]));
        a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        // Convention: render targets rest in SHADER_READ_ONLY between passes, so the render pass itself
        // performs the transitions; the backbuffer ends in PRESENT_SRC. Multisampled targets are never sampled
        // (they have no SAMPLED usage), so they rest as attachments and their resolve target takes the role.
        const VkImageLayout rest = key.toBackbuffer ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                   : key.samples > 1 ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                                     : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        a.initialLayout = a.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ? rest : VK_IMAGE_LAYOUT_UNDEFINED;
        a.finalLayout = rest;
        colorRefs[i] = {count, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        resolveRefs[i] = {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED};
        anyResolve |= key.colorResolve[i] != 0;
        ++count;
    }
    VkAttachmentReference depthRef{};
    if (key.depth != VK_FORMAT_UNDEFINED) {
        VkAttachmentDescription& a = attachments[count];
        a.format = key.depth;
        a.samples = VkSampleCountFlagBits(key.samples);
        a.loadOp = toVkLoad(LoadOp(key.depthLoad));
        a.storeOp = toVkStore(StoreOp(key.depthStore));
        a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        // Same convention as color targets: a sampled depth target (shadow map) rests in SHADER_READ_ONLY between passes.
        const VkImageLayout rest = key.depthSampled ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        a.initialLayout = a.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ? rest : VK_IMAGE_LAYOUT_UNDEFINED;
        a.finalLayout = rest;
        depthRef = {count, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        ++count;
    }
    // Resolve targets come last; they are written whole, so their old contents never matter.
    for (uint32_t i = 0; i < key.colorCount; ++i) {
        if (!key.colorResolve[i]) continue;
        VkAttachmentDescription& a = attachments[count];
        a.format = key.color[i];
        a.samples = VK_SAMPLE_COUNT_1_BIT;
        a.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        a.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        resolveRefs[i] = {count, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        ++count;
    }

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = key.colorCount;
    subpass.pColorAttachments = colorRefs;
    subpass.pResolveAttachments = anyResolve ? resolveRefs : nullptr;
    subpass.pDepthStencilAttachment = key.depth != VK_FORMAT_UNDEFINED ? &depthRef : nullptr;

    constexpr VkPipelineStageFlags kAttachmentStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    constexpr VkAccessFlags kAttachmentWrites =
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = kAttachmentStages | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask = kAttachmentStages;
    deps[0].srcAccessMask = kAttachmentWrites;
    deps[0].dstAccessMask = kAttachmentWrites | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = kAttachmentStages;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    deps[1].srcAccessMask = kAttachmentWrites;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = count;
    ci.pAttachments = attachments;
    ci.subpassCount = 1;
    ci.pSubpasses = &subpass;
    ci.dependencyCount = 2;
    ci.pDependencies = deps;
    VkRenderPass pass = VK_NULL_HANDLE;
    if (!FL_VK_OK(vkCreateRenderPass(m_device, &ci, nullptr, &pass))) return VK_NULL_HANDLE;
    m_renderPasses.emplace_back(key, pass);
    return pass;
}

VkFramebuffer VulkanDevice::getFramebuffer(VkRenderPass pass, const VkImageView* views, uint32_t count, uint32_t w, uint32_t h) {
    for (const FramebufferEntry& e : m_framebuffers)
        if (e.renderPass == pass && e.viewCount == count && e.width == w && e.height == h &&
            std::equal(views, views + count, e.views))
            return e.framebuffer;
    FramebufferEntry e{};
    e.renderPass = pass;
    e.viewCount = count;
    e.width = w;
    e.height = h;
    std::copy(views, views + count, e.views);
    VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    ci.renderPass = pass;
    ci.attachmentCount = count;
    ci.pAttachments = views;
    ci.width = w;
    ci.height = h;
    ci.layers = 1;
    if (!FL_VK_OK(vkCreateFramebuffer(m_device, &ci, nullptr, &e.framebuffer))) return VK_NULL_HANDLE;
    m_framebuffers.push_back(e);
    return e.framebuffer;
}

VkPipelineLayout VulkanDevice::getPipelineLayout(const PipelineDesc& desc) {
    PipelineLayoutEntry key{};
    for (uint32_t s = 0; s < shaderformat::kMaxSets; ++s) {
        const SetLayoutRes* l = m_setLayouts.get(desc.setLayouts[s]);
        key.setLayouts[s] = l ? l->layout : m_emptySetLayout;
        if (l) key.setCount = s + 1;
    }
    for (const PipelineLayoutEntry& e : m_pipelineLayouts)
        if (e.setCount == key.setCount && std::equal(e.setLayouts, e.setLayouts + e.setCount, key.setLayouts))
            return e.layout;

    // Every layout carries the full 128-byte push range, so all pipeline layouts stay compatible and
    // bound descriptor sets survive pipeline switches.
    const VkPushConstantRange push{kPushStages, 0, shaderformat::kMaxPerDrawBytes};
    VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    ci.setLayoutCount = key.setCount;
    ci.pSetLayouts = key.setLayouts;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &push;
    if (!FL_VK_OK(vkCreatePipelineLayout(m_device, &ci, nullptr, &key.layout))) return VK_NULL_HANDLE;
    m_pipelineLayouts.push_back(key);
    return key.layout;
}

// ---------------------------------------------------------------- pipelines

PipelineHandle VulkanDevice::createPipeline(const PipelineDesc& desc) {
    PipelineRes res;
    res.desc = desc;
    res.layout = getPipelineLayout(desc);
    if (!res.layout) return {};
    const uint64_t start = platform::nowTicks();
    res.pipeline = makeVkPipeline(desc, res.layout);
    if (!res.pipeline) return {};
    const PipelineHandle handle = m_pipelines.create(res);
    if (!handle) { // a full pool must not leak the pipeline it couldn't hold
        FL_LOG_ERROR("Pipeline pool exhausted (%s)", desc.debugName);
        vkDestroyPipeline(m_device, res.pipeline, nullptr);
        return {};
    }
    ++m_stats.pipelinesCreated; // M9 plan D3
    m_stats.pipelineCreateMs += platform::ticksToMilliseconds(platform::nowTicks() - start);
    return handle;
}

// Hot reload (M9 plan D9): both shaders get new modules, and every pipeline made with either a new VkPipeline from its
// stored description; handles stay valid. If anything fails (an edit that changed the shaders' interface), nothing changes.
bool VulkanDevice::replaceShaders(ShaderHandle vertex, const ShaderDesc& vertexDesc, ShaderHandle fragment, const ShaderDesc& fragmentDesc) {
    ShaderRes* shaders[2] = {m_shaders.get(vertex), m_shaders.get(fragment)};
    const ShaderDesc* descs[2] = {&vertexDesc, &fragmentDesc};
    VkShaderModule modules[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkShaderModule old[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    auto undo = [&] {
        for (uint32_t i = 0; i < 2; ++i) {
            if (old[i]) shaders[i]->module = old[i];
            if (modules[i]) vkDestroyShaderModule(m_device, modules[i], nullptr);
        }
        return false;
    };
    for (uint32_t i = 0; i < 2; ++i) {
        if (!shaders[i] || descs[i]->code.empty() || descs[i]->code.size() % 4 != 0) return undo();
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = descs[i]->code.size();
        ci.pCode = reinterpret_cast<const uint32_t*>(descs[i]->code.data());
        if (!FL_VK_OK(vkCreateShaderModule(m_device, &ci, nullptr, &modules[i]))) return undo();
        setObjectName(VK_OBJECT_TYPE_SHADER_MODULE, uint64_t(modules[i]), descs[i]->debugName);
    }
    for (uint32_t i = 0; i < 2; ++i) {
        old[i] = shaders[i]->module;
        shaders[i]->module = modules[i];
    }
    std::vector<std::pair<PipelineRes*, VkPipeline>> rebuilt;
    bool ok = true;
    m_pipelines.forEach([&](PipelineHandle, PipelineRes& p) {
        if (!ok || (p.desc.vertex != vertex && p.desc.fragment != fragment)) return;
        const VkPipeline pipeline = makeVkPipeline(p.desc, p.layout);
        if (pipeline) rebuilt.emplace_back(&p, pipeline);
        else ok = false;
    });
    if (!ok) {
        for (auto& [p, pipeline] : rebuilt) vkDestroyPipeline(m_device, pipeline, nullptr);
        return undo();
    }
    for (auto& [p, pipeline] : rebuilt) {
        defer(Deferred::Kind::Pipeline, uint64_t(p->pipeline)); // frames in flight still use it
        p->pipeline = pipeline;
    }
    for (VkShaderModule m : old) defer(Deferred::Kind::ShaderModule, uint64_t(m));
    m_stats.pipelinesRebuilt += uint32_t(rebuilt.size());
    return true;
}

VkPipeline VulkanDevice::makeVkPipeline(const PipelineDesc& desc, VkPipelineLayout pipelineLayout) {
    const ShaderRes* vs = m_shaders.get(desc.vertex);
    const ShaderRes* fs = m_shaders.get(desc.fragment);
    if (!vs || !fs) {
        FL_LOG_ERROR("Pipeline %s: invalid shader handle", desc.debugName);
        return VK_NULL_HANDLE;
    }
    const VkRenderPass compatPass = getRenderPass(compatibleKey(desc.attachments));
    if (!compatPass) return VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs->module, "main"};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs->module, "main"};

    VkVertexInputBindingDescription bindings[kMaxVertexStreams]{};
    VkVertexInputAttributeDescription attributes[kMaxVertexAttributes]{};
    const VertexLayout& vl = desc.vertexLayout;
    for (uint32_t i = 0; i < vl.streamCount; ++i)
        bindings[i] = {i, vl.streams[i].stride,
                       vl.streams[i].perInstance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
    for (uint32_t i = 0; i < vl.attributeCount; ++i) {
        const VertexAttribute& a = vl.attributes[i];
        attributes[i] = {a.location, a.stream, toVkFormat(a.format), a.offset};
    }
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = vl.streamCount;
    vertexInput.pVertexBindingDescriptions = bindings;
    vertexInput.vertexAttributeDescriptionCount = vl.attributeCount;
    vertexInput.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = desc.topology == Topology::Lines ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = desc.raster.wireframe && m_caps.wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    raster.cullMode = desc.raster.cull == CullMode::Back    ? VK_CULL_MODE_BACK_BIT
                      : desc.raster.cull == CullMode::Front ? VK_CULL_MODE_FRONT_BIT
                                                            : VK_CULL_MODE_NONE;
    // CCW on both backends: the Y flip lives in the projection (§4.2). Mirrored draws ask for CW.
    raster.frontFace = desc.raster.frontFaceClockwise ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.depthBiasEnable = (desc.raster.depthBiasConstant != 0.0f || desc.raster.depthBiasSlope != 0.0f);
    raster.depthBiasConstantFactor = desc.raster.depthBiasConstant;
    raster.depthBiasSlopeFactor = desc.raster.depthBiasSlope;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VkSampleCountFlagBits(std::max<uint8_t>(desc.attachments.samples, 1));
    multisample.alphaToCoverageEnable = desc.raster.alphaToCoverage ? VK_TRUE : VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = desc.depth.test ? VK_TRUE : VK_FALSE;
    depth.depthWriteEnable = desc.depth.write ? VK_TRUE : VK_FALSE;
    depth.depthCompareOp = toVkCompare(desc.depth.compare);

    VkPipelineColorBlendAttachmentState blends[kMaxColorAttachments]{};
    for (uint32_t i = 0; i < desc.attachments.colorCount; ++i) {
        const BlendState& b = desc.blend[i];
        blends[i].blendEnable = b.enable ? VK_TRUE : VK_FALSE;
        blends[i].srcColorBlendFactor = toVkBlendFactor(b.srcColor);
        blends[i].dstColorBlendFactor = toVkBlendFactor(b.dstColor);
        blends[i].colorBlendOp = toVkBlendOp(b.colorOp);
        blends[i].srcAlphaBlendFactor = toVkBlendFactor(b.srcAlpha);
        blends[i].dstAlphaBlendFactor = toVkBlendFactor(b.dstAlpha);
        blends[i].alphaBlendOp = toVkBlendOp(b.alphaOp);
        blends[i].colorWriteMask = b.writeMask;
    }
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = desc.attachments.colorCount;
    blend.pAttachments = blends;

    const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vertexInput;
    ci.pInputAssemblyState = &inputAssembly;
    ci.pViewportState = &viewport;
    ci.pRasterizationState = &raster;
    ci.pMultisampleState = &multisample;
    ci.pDepthStencilState = &depth;
    ci.pColorBlendState = &blend;
    ci.pDynamicState = &dynamic;
    ci.layout = pipelineLayout;
    ci.renderPass = compatPass;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (!FL_VK_OK(vkCreateGraphicsPipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &pipeline))) return VK_NULL_HANDLE;
    setObjectName(VK_OBJECT_TYPE_PIPELINE, uint64_t(pipeline), desc.debugName);
    return pipeline;
}

// ---------------------------------------------------------------- resource sets

ResourceSetLayoutHandle VulkanDevice::createResourceSetLayout(const ResourceSetLayoutDesc& desc) {
    if (desc.count > kMaxSetBindings) {
        FL_LOG_ERROR("Set %u layout has %u bindings; the limit is %u", desc.set, desc.count, kMaxSetBindings);
        return {};
    }
    VkDescriptorSetLayoutBinding bindings[kMaxSetBindings]{};
    for (uint32_t i = 0; i < desc.count; ++i) {
        const ResourceBinding& b = desc.bindings[i];
        for (uint32_t j = 0; j < i; ++j)
            if (desc.bindings[j].binding == b.binding) {
                FL_LOG_ERROR("Set %u layout declares binding %u twice", desc.set, b.binding);
                return {};
            }
        // Enforce the GL slot table on Vulkan too, so a layout that would break on GL fails here first.
        const bool isTexture = b.kind == BindingKind::Texture;
        const uint8_t slot = isTexture ? shaderformat::glTextureUnit(desc.set, b.binding)
                                       : shaderformat::glUniformBufferPoint(desc.set, b.binding);
        if (slot == shaderformat::kInvalidSlot) {
            FL_LOG_ERROR("Set %u binding %u doesn't fit the binding model (ARCHITECTURE.md §4.6)", desc.set, b.binding);
            return {};
        }
        bindings[i].binding = b.binding;
        bindings[i].descriptorType = isTexture ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                     : b.kind == BindingKind::UniformBufferDynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                                                                                   : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = toVkStages(b.stages);
    }
    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = desc.count;
    ci.pBindings = bindings;
    SetLayoutRes res;
    res.desc = desc;
    if (!FL_VK_OK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &res.layout))) return {};
    return m_setLayouts.create(res);
}

VkDescriptorSet VulkanDevice::allocateDescriptorSet(VkDescriptorSetLayout layout, VkDescriptorPool& outPool) {
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!m_descriptorPools.empty()) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = m_descriptorPools.back();
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &layout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            const VkResult r = vkAllocateDescriptorSets(m_device, &ai, &set);
            if (r == VK_SUCCESS) {
                outPool = m_descriptorPools.back();
                return set;
            }
            if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) {
                FL_LOG_ERROR("vkAllocateDescriptorSets failed: %s", resultName(r));
                return VK_NULL_HANDLE;
            }
        }
        const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 512},
                                              {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 512},
                                              {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2048}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pi.maxSets = 1024;
        pi.poolSizeCount = 3;
        pi.pPoolSizes = sizes;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        if (!FL_VK_OK(vkCreateDescriptorPool(m_device, &pi, nullptr, &pool))) return VK_NULL_HANDLE;
        m_descriptorPools.push_back(pool);
    }
    return VK_NULL_HANDLE;
}

ResourceSetHandle VulkanDevice::createResourceSet(const ResourceSetDesc& desc) {
    const SetLayoutRes* layout = m_setLayouts.get(desc.layout);
    if (!layout) return {};
    if (desc.writes.size() > kMaxSetBindings) {
        FL_LOG_ERROR("Resource set %s has %zu writes; the limit is %u", desc.debugName, desc.writes.size(), kMaxSetBindings);
        return {};
    }
    ResourceSetRes res;
    res.set = allocateDescriptorSet(layout->layout, res.pool);
    if (!res.set) return {};

    VkWriteDescriptorSet writes[kMaxSetBindings]{};
    VkDescriptorBufferInfo bufferInfos[kMaxSetBindings]{};
    VkDescriptorImageInfo imageInfos[kMaxSetBindings]{};
    uint32_t count = 0;
    for (const ResourceWrite& w : desc.writes) {
        VkWriteDescriptorSet& wr = writes[count];
        wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet = res.set;
        wr.dstBinding = w.binding;
        wr.descriptorCount = 1;
        if (w.texture) {
            const TextureRes* t = m_textures.get(w.texture);
            const SamplerRes* s = m_samplers.get(w.sampler);
            if (!t || !s) {
                FL_LOG_ERROR("Resource set %s: invalid texture or sampler at binding %u", desc.debugName, w.binding);
                vkFreeDescriptorSets(m_device, res.pool, 1, &res.set);
                return {};
            }
            imageInfos[count] = {s->sampler, t->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            wr.pImageInfo = &imageInfos[count];
        } else {
            const BufferRes* b = m_buffers.get(w.buffer.buffer);
            if (!b) {
                FL_LOG_ERROR("Resource set %s: invalid buffer at binding %u", desc.debugName, w.binding);
                vkFreeDescriptorSets(m_device, res.pool, 1, &res.set);
                return {};
            }
            bufferInfos[count] = {b->buffer, w.buffer.offset, w.buffer.size ? w.buffer.size : VK_WHOLE_SIZE};
            bool dynamic = false;
            for (uint32_t i = 0; i < layout->desc.count; ++i)
                if (layout->desc.bindings[i].binding == w.binding)
                    dynamic = layout->desc.bindings[i].kind == BindingKind::UniformBufferDynamic;
            wr.descriptorType = dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            wr.pBufferInfo = &bufferInfos[count];
        }
        ++count;
    }
    vkUpdateDescriptorSets(m_device, count, writes, 0, nullptr);
    setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET, uint64_t(res.set), desc.debugName);
    return m_resourceSets.create(res);
}

QueryPoolHandle VulkanDevice::createTimestampPool(uint32_t count) {
    QueryPoolRes res;
    res.count = count;
    res.scratch.resize(size_t(count) * 2);
    VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = count;
    for (VkQueryPool& p : res.pools)
        if (!FL_VK_OK(vkCreateQueryPool(m_device, &ci, nullptr, &p))) return {};
    return m_queryPools.create(std::move(res));
}

// ---------------------------------------------------------------- destruction

// The frame being recorded may already reference the object, so it's freed once that frame completes; between
// frames, the last submitted frame is the newest possible user.
void VulkanDevice::defer(Deferred::Kind kind, uint64_t handle, uint64_t extra, VmaAllocation allocation) {
    const uint64_t serial = m_frameActive ? m_submitSerial + 1 : m_submitSerial;
    m_deferred.push_back({kind, handle, extra, allocation, serial});
}

void VulkanDevice::destroyDeferredNow(const Deferred& d) {
    using K = Deferred::Kind;
    switch (d.kind) {
    case K::Buffer: vmaDestroyBuffer(m_allocator, VkBuffer(d.handle), d.allocation); break;
    case K::Image:
        vkDestroyImageView(m_device, VkImageView(d.extra), nullptr);
        vmaDestroyImage(m_allocator, VkImage(d.handle), d.allocation);
        break;
    case K::ImageView: vkDestroyImageView(m_device, VkImageView(d.handle), nullptr); break;
    case K::Sampler: vkDestroySampler(m_device, VkSampler(d.handle), nullptr); break;
    case K::ShaderModule: vkDestroyShaderModule(m_device, VkShaderModule(d.handle), nullptr); break;
    case K::Pipeline: vkDestroyPipeline(m_device, VkPipeline(d.handle), nullptr); break;
    case K::PipelineLayout: vkDestroyPipelineLayout(m_device, VkPipelineLayout(d.handle), nullptr); break;
    case K::SetLayout: vkDestroyDescriptorSetLayout(m_device, VkDescriptorSetLayout(d.handle), nullptr); break;
    case K::DescriptorSet: {
        VkDescriptorSet set = VkDescriptorSet(d.handle);
        vkFreeDescriptorSets(m_device, VkDescriptorPool(d.extra), 1, &set);
        break;
    }
    case K::QueryPool: vkDestroyQueryPool(m_device, VkQueryPool(d.handle), nullptr); break;
    case K::Framebuffer: vkDestroyFramebuffer(m_device, VkFramebuffer(d.handle), nullptr); break;
    }
}

void VulkanDevice::runDeferred(bool all) {
    size_t kept = 0;
    for (size_t i = 0; i < m_deferred.size(); ++i) {
        if (all || m_deferred[i].serial <= m_completedSerial) destroyDeferredNow(m_deferred[i]);
        else m_deferred[kept++] = m_deferred[i];
    }
    m_deferred.resize(kept);
}

void VulkanDevice::destroy(BufferHandle h) {
    if (BufferRes* r = m_buffers.get(h); r && !r->ownedByRing) {
        m_memory.release(r->category, r->size);
        defer(Deferred::Kind::Buffer, uint64_t(r->buffer), 0, r->allocation);
        m_buffers.destroy(h);
    }
}

void VulkanDevice::destroy(TextureHandle h) {
    if (TextureRes* r = m_textures.get(h)) {
        m_memory.release(memoryCategory(r->desc), memoryBytes(r->desc));
        // Framebuffers that reference one of its views must go first.
        auto ownsView = [&](VkImageView v) {
            return v == r->view || std::find(r->layerViews, r->layerViews + r->layerViewCount, v) != r->layerViews + r->layerViewCount;
        };
        for (auto it = m_framebuffers.begin(); it != m_framebuffers.end();) {
            if (std::any_of(it->views, it->views + it->viewCount, ownsView)) {
                defer(Deferred::Kind::Framebuffer, uint64_t(it->framebuffer));
                it = m_framebuffers.erase(it);
            } else {
                ++it;
            }
        }
        for (uint32_t l = 0; l < r->layerViewCount; ++l) defer(Deferred::Kind::ImageView, uint64_t(r->layerViews[l]));
        defer(Deferred::Kind::Image, uint64_t(r->image), uint64_t(r->view), r->allocation);
        m_textures.destroy(h);
    }
}

void VulkanDevice::destroy(SamplerHandle h) {
    if (SamplerRes* r = m_samplers.get(h)) {
        defer(Deferred::Kind::Sampler, uint64_t(r->sampler));
        m_samplers.destroy(h);
    }
}

void VulkanDevice::destroy(ShaderHandle h) {
    if (ShaderRes* r = m_shaders.get(h)) {
        defer(Deferred::Kind::ShaderModule, uint64_t(r->module));
        m_shaders.destroy(h);
    }
}

void VulkanDevice::destroy(PipelineHandle h) {
    if (PipelineRes* r = m_pipelines.get(h)) {
        defer(Deferred::Kind::Pipeline, uint64_t(r->pipeline));
        m_pipelines.destroy(h);
    }
}

void VulkanDevice::destroy(ResourceSetLayoutHandle h) {
    if (SetLayoutRes* r = m_setLayouts.get(h)) {
        // Drop cached pipeline layouts built from it: a future set layout may reuse the same handle value.
        // (Pipelines must be destroyed before their set layouts.)
        for (auto it = m_pipelineLayouts.begin(); it != m_pipelineLayouts.end();) {
            if (std::find(it->setLayouts, it->setLayouts + it->setCount, r->layout) != it->setLayouts + it->setCount) {
                defer(Deferred::Kind::PipelineLayout, uint64_t(it->layout));
                it = m_pipelineLayouts.erase(it);
            } else {
                ++it;
            }
        }
        defer(Deferred::Kind::SetLayout, uint64_t(r->layout));
        m_setLayouts.destroy(h);
    }
}

void VulkanDevice::destroy(ResourceSetHandle h) {
    if (ResourceSetRes* r = m_resourceSets.get(h)) {
        defer(Deferred::Kind::DescriptorSet, uint64_t(r->set), uint64_t(r->pool));
        m_resourceSets.destroy(h);
    }
}

void VulkanDevice::destroy(QueryPoolHandle h) {
    if (QueryPoolRes* r = m_queryPools.get(h)) {
        for (VkQueryPool p : r->pools) defer(Deferred::Kind::QueryPool, uint64_t(p));
        m_queryPools.destroy(h);
    }
}

} // namespace rhi::vk
