#include "rhi/vulkan/VulkanDevice.h"

#include "rhi/CommandStream.h"

namespace rhi::vk {

void VulkanDevice::flushUploads(VkCommandBuffer cmd) {
    if (m_pendingCopies.empty() && m_pendingBufferCopies.empty()) return;
    recordCopies(cmd, m_buffers.get(m_stagingBuffer)->buffer, m_pendingCopies, m_pendingBufferCopies);
    m_pendingCopies.clear();
    m_pendingBufferCopies.clear();
}

void VulkanDevice::recordCopies(VkCommandBuffer cmd, VkBuffer staging, std::span<const PendingCopy> images,
                                std::span<const PendingBufferCopy> buffers) {
    for (const PendingCopy& c : images) {
        if (c.initializeImage) {
            VkImageMemoryBarrier init{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            init.srcAccessMask = 0;
            init.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            init.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            init.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            init.srcQueueFamilyIndex = init.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            init.image = c.image;
            init.subresourceRange = {c.aspect, 0, c.mipLevels, 0, c.arrayLayers};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                                 nullptr, 1, &init);
        }
        VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toTransfer.srcAccessMask = c.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_SHADER_READ_BIT;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toTransfer.oldLayout = c.oldLayout;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = c.image;
        toTransfer.subresourceRange = {c.aspect, c.region.imageSubresource.mipLevel, 1,
                                       c.region.imageSubresource.baseArrayLayer, 1};
        const VkPipelineStageFlags srcStage =
            c.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        vkCmdPipelineBarrier(cmd, srcStage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransfer);

        vkCmdCopyBufferToImage(cmd, staging, c.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c.region);

        VkImageMemoryBarrier toShader = toTransfer;
        toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &toShader);
    }
    if (!buffers.empty()) {
        for (const PendingBufferCopy& c : buffers) vkCmdCopyBuffer(cmd, staging, c.dst, 1, &c.region);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
}

void VulkanDevice::beginRenderPass(VkCommandBuffer cmd, const RenderPassDesc& desc) {
    RenderPassKey key{};
    VkImageView views[kMaxColorAttachments * 2 + 1]{};
    VkClearValue clears[kMaxColorAttachments * 2 + 1]{};
    uint32_t viewCount = 0;
    uint32_t width = 0, height = 0;

    key.colorCount = desc.colorCount;
    key.toBackbuffer = desc.toBackbuffer ? 1 : 0;
    key.samples = 1;
    for (uint32_t i = 0; i < desc.colorCount; ++i) {
        const ColorAttachment& a = desc.color[i];
        if (desc.toBackbuffer && i == 0) {
            key.color[i] = m_swapFormat;
            views[viewCount] = m_swapViews[m_imageIndex];
            width = m_swapExtent.width;
            height = m_swapExtent.height;
        } else {
            TextureRes* t = m_textures.get(a.texture);
            if (!t) {
                FL_LOG_ERROR("Render pass %s: invalid color attachment %u", desc.debugName, i);
                return;
            }
            if (t->layerViewCount && a.layer >= t->layerViewCount) {
                FL_LOG_ERROR("Render pass %s: color attachment %u has no layer %u", desc.debugName, i, a.layer);
                return;
            }
            key.color[i] = t->format;
            key.samples = t->desc.samples;
            views[viewCount] = t->attachmentView(a.layer);
            width = t->desc.width >> a.mip;
            height = t->desc.height >> a.mip;
            t->layout = t->desc.samples > 1 ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        key.colorLoad[i] = uint8_t(a.load);
        key.colorStore[i] = uint8_t(a.store);
        std::memcpy(clears[viewCount].color.float32, a.clear, sizeof(a.clear));
        ++viewCount;
    }
    key.depth = VK_FORMAT_UNDEFINED;
    if (desc.useDepth) {
        TextureRes* t = m_textures.get(desc.depth.texture);
        if (!t) {
            FL_LOG_ERROR("Render pass %s: invalid depth attachment", desc.debugName);
            return;
        }
        if (t->layerViewCount && desc.depth.layer >= t->layerViewCount) {
            FL_LOG_ERROR("Render pass %s: the depth attachment has no layer %u", desc.debugName, desc.depth.layer);
            return;
        }
        const bool sampled = any(t->desc.usage & TextureUsage::Sampled);
        if (desc.colorCount == 0) key.samples = t->desc.samples; // depth-only passes take the count from the depth target
        key.depth = t->format;
        key.depthLoad = uint8_t(desc.depth.load);
        key.depthStore = uint8_t(desc.depth.store);
        key.depthSampled = sampled ? 1 : 0;
        views[viewCount] = t->attachmentView(desc.depth.layer);
        clears[viewCount].depthStencil = {desc.depth.clearDepth, 0};
        width = t->desc.width;
        height = t->desc.height;
        t->layout = sampled ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        ++viewCount;
    }
    for (uint32_t i = 0; i < desc.colorCount; ++i) { // resolve targets follow depth (see getRenderPass)
        if (!desc.color[i].resolveTo || desc.toBackbuffer) continue;
        TextureRes* r = m_textures.get(desc.color[i].resolveTo);
        if (!r || r->desc.samples != 1 || key.samples == 1) {
            FL_LOG_ERROR("Render pass %s: color %u needs a multisampled attachment and a single-sample resolve target",
                         desc.debugName, i);
            return;
        }
        key.colorResolve[i] = 1;
        views[viewCount++] = r->view;
        r->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    const VkRenderPass pass = getRenderPass(key);
    const VkFramebuffer framebuffer = getFramebuffer(pass, views, viewCount, width, height);
    if (!pass || !framebuffer) return;

    VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    bi.renderPass = pass;
    bi.framebuffer = framebuffer;
    bi.renderArea = {{0, 0}, {width, height}};
    bi.clearValueCount = viewCount;
    bi.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);

    // Full-target viewport and scissor by default (attachment texel rows, ARCHITECTURE.md §4.2 C6).
    const VkViewport viewport{0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, {width, height}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void VulkanDevice::translate(VkCommandBuffer cmd, const CommandList& list) {
    using namespace cmd;
    Reader reader(list.data());
    VkPipelineLayout& layout = m_cmdLayout; // carried across the lists of one submit, like GL's bound program
    auto warnNoPipeline = [&] {
        if (!m_warnedUnboundLayout) FL_LOG_ERROR("Resource-set / per-draw binds before any BindPipeline are ignored");
        m_warnedUnboundLayout = true;
    };

    for (const Header* h = reader.next(); h; h = reader.next()) {
        switch (h->op) {
        case Op::BeginRenderPass: beginRenderPass(cmd, Reader::as<BeginRenderPass>(h)->desc); break;
        case Op::EndRenderPass: vkCmdEndRenderPass(cmd); break;
        case Op::SetViewport: {
            const Rect& r = Reader::as<SetViewport>(h)->rect;
            // Positive height: the Y flip is in the projection (ARCHITECTURE.md §4.2).
            const VkViewport vp{float(r.x), float(r.y), float(r.width), float(r.height), 0.0f, 1.0f};
            vkCmdSetViewport(cmd, 0, 1, &vp);
            break;
        }
        case Op::SetScissor: {
            const Rect& r = Reader::as<SetScissor>(h)->rect;
            const VkRect2D s{{r.x, r.y}, {r.width, r.height}};
            vkCmdSetScissor(cmd, 0, 1, &s);
            break;
        }
        case Op::BindPipeline: {
            const PipelineRes* p = m_pipelines.get(Reader::as<BindPipeline>(h)->pipeline);
            if (!p) {
                FL_LOG_ERROR("BindPipeline with an invalid handle");
                break;
            }
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p->pipeline);
            layout = p->layout;
            break;
        }
        case Op::BindResourceSet: {
            const auto* c = Reader::as<BindResourceSet>(h);
            const ResourceSetRes* s = m_resourceSets.get(c->resourceSet);
            if (!s) break;
            if (!layout) {
                warnNoPipeline();
                break;
            }
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, c->set, 1, &s->set, c->dynamicCount,
                                    c->dynamicOffsets);
            break;
        }
        case Op::SetPerDraw: {
            const auto* c = Reader::as<SetPerDraw>(h);
            if (!layout) {
                warnNoPipeline();
                break;
            }
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, c->bytes,
                               payload(c));
            break;
        }
        case Op::BindVertexStreams: {
            const auto* c = Reader::as<BindVertexStreams>(h);
            VkBuffer buffers[kMaxVertexStreams]{};
            VkDeviceSize offsets[kMaxVertexStreams]{};
            for (uint32_t i = 0; i < c->count; ++i) {
                const BufferRes* b = m_buffers.get(c->streams[i].buffer);
                buffers[i] = b ? b->buffer : VK_NULL_HANDLE;
                offsets[i] = c->streams[i].offset;
            }
            vkCmdBindVertexBuffers(cmd, c->first, c->count, buffers, offsets);
            break;
        }
        case Op::BindIndexBuffer: {
            const auto* c = Reader::as<BindIndexBuffer>(h);
            if (const BufferRes* b = m_buffers.get(c->buffer.buffer))
                vkCmdBindIndexBuffer(cmd, b->buffer, c->buffer.offset,
                                     c->type == IndexType::U16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
            break;
        }
        case Op::Draw: {
            const auto* c = Reader::as<Draw>(h);
            vkCmdDraw(cmd, c->vertexCount, c->instanceCount, c->firstVertex, 0);
            break;
        }
        case Op::DrawIndexed: {
            const auto* c = Reader::as<DrawIndexed>(h);
            vkCmdDrawIndexed(cmd, c->indexCount, c->instanceCount, c->firstIndex, 0, 0);
            break;
        }
        case Op::WriteTimestamp: {
            const auto* c = Reader::as<WriteTimestamp>(h);
            if (const QueryPoolRes* q = m_queryPools.get(c->pool); q && c->index < q->count)
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q->pools[m_slot], c->index);
            break;
        }
        case Op::PushDebugGroup:
            if (m_debugUtils && vkCmdBeginDebugUtilsLabelEXT) {
                VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
                label.pLabelName = Reader::as<PushDebugGroup>(h)->name;
                vkCmdBeginDebugUtilsLabelEXT(cmd, &label);
            }
            break;
        case Op::PopDebugGroup:
            if (m_debugUtils && vkCmdEndDebugUtilsLabelEXT) vkCmdEndDebugUtilsLabelEXT(cmd);
            break;
        }
    }
}

} // namespace rhi::vk
