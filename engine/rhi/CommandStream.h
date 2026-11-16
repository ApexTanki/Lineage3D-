#pragma once

// Encoded command layout shared by CommandList (writer) and the backends and tests (readers).
// Every command starts with a Header and is padded to 8 bytes.

#include "rhi/Descs.h"

#include <cstring>

namespace rhi::cmd {

enum class Op : uint8_t {
    BeginRenderPass,
    EndRenderPass,
    SetViewport,
    SetScissor,
    BindPipeline,
    BindResourceSet,
    SetPerDraw,
    BindVertexStreams,
    BindIndexBuffer,
    Draw,
    DrawIndexed,
    WriteTimestamp,
    PushDebugGroup,
    PopDebugGroup,
};

struct Header {
    Op op;
    uint8_t pad;
    uint16_t size; // whole command including the header and any trailing payload
};

struct BeginRenderPass { Header header; RenderPassDesc desc; };
struct EndRenderPass { Header header; };
struct SetViewport { Header header; Rect rect; };
struct SetScissor { Header header; Rect rect; };
struct BindPipeline { Header header; PipelineHandle pipeline; };
struct BindResourceSet {
    Header header;
    uint8_t set;
    uint8_t dynamicCount;
    uint8_t pad[2];
    ResourceSetHandle resourceSet;
    uint32_t dynamicOffsets[kMaxDynamicOffsets];
};
// Followed by `bytes` bytes of payload (rounded up to 8).
struct SetPerDraw { Header header; uint8_t bytes; uint8_t pad[3]; uint32_t sequence; };
struct BindVertexStreams { Header header; uint8_t first; uint8_t count; uint8_t pad[2]; BufferSlice streams[kMaxVertexStreams]; };
struct BindIndexBuffer { Header header; IndexType type; uint8_t pad[3]; BufferSlice buffer; };
struct Draw { Header header; uint32_t vertexCount; uint32_t instanceCount; uint32_t firstVertex; };
struct DrawIndexed { Header header; uint32_t indexCount; uint32_t instanceCount; uint32_t firstIndex; };
struct WriteTimestamp { Header header; QueryPoolHandle pool; uint32_t index; };
struct PushDebugGroup { Header header; const char* name; };
struct PopDebugGroup { Header header; };

inline const void* payload(const SetPerDraw* c) { return c + 1; }

// Walks an encoded stream: `for (const Header* h = r.next(); h; h = r.next())`.
class Reader {
public:
    explicit Reader(std::span<const std::byte> data) : m_data(data) {}
    const Header* next() {
        if (m_offset + sizeof(Header) > m_data.size()) return nullptr;
        const auto* h = reinterpret_cast<const Header*>(m_data.data() + m_offset);
        m_offset += h->size;
        return h;
    }
    template <typename T>
    static const T* as(const Header* h) {
        return reinterpret_cast<const T*>(h);
    }

private:
    std::span<const std::byte> m_data;
    size_t m_offset = 0;
};

} // namespace rhi::cmd
