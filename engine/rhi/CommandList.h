#pragma once

#include "rhi/Descs.h"

#include <span>
#include <vector>

namespace rhi {

// Backend-agnostic command stream (ARCHITECTURE.md §5.3). Recording only writes compact POD commands
// into memory: no virtual calls and no graphics-API calls, so it can happen on any thread.
// Device::submit() translates the stream in one switch loop (Vulkan: vkCmd*; GL: replay after one bulk
// uniform upload). This keeps virtual dispatch out of every per-draw path.
//
// The buffer is reserved once and reused every frame, so recording doesn't allocate after warm-up.
class CommandList {
public:
    explicit CommandList(size_t reserveBytes = 256 * 1024);

    void reset();
    // Grows the buffer to at least `bytes` (never shrinks): the renderer sizes it from the scene and settings, so a
    // camera that sees more than before never makes recording allocate (M5).
    void reserve(size_t bytes);
    // Upper bound of one draw's recording when every state changes with it: pipeline, four resource sets, two vertex-stream
    // binds, the index buffer, the largest per-draw block and the draw.
    static size_t maxBytesPerDraw();

    void beginRenderPass(const RenderPassDesc& desc);
    void endRenderPass();
    void setViewport(const Rect& rect);
    void setScissor(const Rect& rect);
    void bindPipeline(PipelineHandle pipeline);
    void bindResourceSet(uint8_t set, ResourceSetHandle resourceSet, std::span<const uint32_t> dynamicOffsets = {});
    void setPerDraw(const void* data, uint8_t bytes); // ≤ 128 B: push constants (VK) / UBO ring range (GL)
    void bindVertexStreams(uint8_t first, std::span<const BufferSlice> streams);
    void bindIndexBuffer(const BufferSlice& buffer, IndexType type);
    void draw(uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0);
    // No base-instance parameter: GL 3.3 core lacks it (ARB_base_instance is GL 4.2).
    void drawIndexed(uint32_t indexCount, uint32_t instanceCount = 1, uint32_t firstIndex = 0);
    void writeTimestamp(QueryPoolHandle pool, uint32_t index);
    void pushDebugGroup(const char* name); // name must outlive the frame (string literals)
    void popDebugGroup();

    std::span<const std::byte> data() const { return {m_data.data(), m_size}; }
    uint32_t drawCount() const { return m_drawCount; }
    uint32_t perDrawCount() const { return m_perDrawCount; }

private:
    void* push(size_t bytes);

    std::vector<std::byte> m_data;
    size_t m_size = 0;
    uint32_t m_drawCount = 0;
    uint32_t m_perDrawCount = 0;
};

} // namespace rhi
