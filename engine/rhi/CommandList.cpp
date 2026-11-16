#include "rhi/CommandList.h"

#include "core/Assert.h"
#include "rhi/CommandStream.h"

#include <algorithm>

namespace rhi {
namespace {

constexpr size_t align8(size_t v) { return (v + 7) & ~size_t(7); }

template <typename T>
T* emit(void* memory, cmd::Op op, size_t totalBytes) {
    T* c = new (memory) T{};
    c->header.op = op;
    c->header.size = static_cast<uint16_t>(totalBytes);
    return c;
}

} // namespace

CommandList::CommandList(size_t reserveBytes) {
    m_data.resize(reserveBytes);
}

void CommandList::reset() {
    m_size = 0;
    m_drawCount = 0;
    m_perDrawCount = 0;
}

void CommandList::reserve(size_t bytes) {
    if (m_data.size() < bytes) m_data.resize(bytes);
}

size_t CommandList::maxBytesPerDraw() {
    return align8(sizeof(cmd::BindPipeline)) + 4 * align8(sizeof(cmd::BindResourceSet)) + 2 * align8(sizeof(cmd::BindVertexStreams)) +
           align8(sizeof(cmd::BindIndexBuffer)) + align8(sizeof(cmd::SetPerDraw)) + 128 + align8(sizeof(cmd::DrawIndexed));
}

void* CommandList::push(size_t bytes) {
    bytes = align8(bytes);
    if (m_size + bytes > m_data.size()) m_data.resize(std::max(m_data.size() * 2, m_size + bytes)); // warm-up only
    void* p = m_data.data() + m_size;
    m_size += bytes;
    return p;
}

void CommandList::beginRenderPass(const RenderPassDesc& desc) {
    constexpr size_t n = align8(sizeof(cmd::BeginRenderPass));
    emit<cmd::BeginRenderPass>(push(n), cmd::Op::BeginRenderPass, n)->desc = desc;
}

void CommandList::endRenderPass() {
    constexpr size_t n = align8(sizeof(cmd::EndRenderPass));
    emit<cmd::EndRenderPass>(push(n), cmd::Op::EndRenderPass, n);
}

void CommandList::setViewport(const Rect& rect) {
    constexpr size_t n = align8(sizeof(cmd::SetViewport));
    emit<cmd::SetViewport>(push(n), cmd::Op::SetViewport, n)->rect = rect;
}

void CommandList::setScissor(const Rect& rect) {
    constexpr size_t n = align8(sizeof(cmd::SetScissor));
    emit<cmd::SetScissor>(push(n), cmd::Op::SetScissor, n)->rect = rect;
}

void CommandList::bindPipeline(PipelineHandle pipeline) {
    constexpr size_t n = align8(sizeof(cmd::BindPipeline));
    emit<cmd::BindPipeline>(push(n), cmd::Op::BindPipeline, n)->pipeline = pipeline;
}

void CommandList::bindResourceSet(uint8_t set, ResourceSetHandle resourceSet, std::span<const uint32_t> dynamicOffsets) {
    FL_ASSERT(dynamicOffsets.size() <= kMaxDynamicOffsets, "too many dynamic offsets");
    constexpr size_t n = align8(sizeof(cmd::BindResourceSet));
    auto* c = emit<cmd::BindResourceSet>(push(n), cmd::Op::BindResourceSet, n);
    c->set = set;
    c->resourceSet = resourceSet;
    c->dynamicCount = static_cast<uint8_t>(dynamicOffsets.size());
    std::copy(dynamicOffsets.begin(), dynamicOffsets.end(), c->dynamicOffsets);
}

void CommandList::setPerDraw(const void* data, uint8_t bytes) {
    FL_ASSERT(bytes <= shaderformat::kMaxPerDrawBytes, "per-draw constants exceed 128 bytes");
    const size_t n = align8(sizeof(cmd::SetPerDraw) + bytes);
    auto* c = emit<cmd::SetPerDraw>(push(n), cmd::Op::SetPerDraw, n);
    c->bytes = bytes;
    c->sequence = m_perDrawCount++;
    std::memcpy(c + 1, data, bytes);
}

void CommandList::bindVertexStreams(uint8_t first, std::span<const BufferSlice> streams) {
    FL_ASSERT(first + streams.size() <= kMaxVertexStreams, "too many vertex streams");
    constexpr size_t n = align8(sizeof(cmd::BindVertexStreams));
    auto* c = emit<cmd::BindVertexStreams>(push(n), cmd::Op::BindVertexStreams, n);
    c->first = first;
    c->count = static_cast<uint8_t>(streams.size());
    std::copy(streams.begin(), streams.end(), c->streams);
}

void CommandList::bindIndexBuffer(const BufferSlice& buffer, IndexType type) {
    constexpr size_t n = align8(sizeof(cmd::BindIndexBuffer));
    auto* c = emit<cmd::BindIndexBuffer>(push(n), cmd::Op::BindIndexBuffer, n);
    c->buffer = buffer;
    c->type = type;
}

void CommandList::draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex) {
    constexpr size_t n = align8(sizeof(cmd::Draw));
    auto* c = emit<cmd::Draw>(push(n), cmd::Op::Draw, n);
    c->vertexCount = vertexCount;
    c->instanceCount = instanceCount;
    c->firstVertex = firstVertex;
    ++m_drawCount;
}

void CommandList::drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex) {
    constexpr size_t n = align8(sizeof(cmd::DrawIndexed));
    auto* c = emit<cmd::DrawIndexed>(push(n), cmd::Op::DrawIndexed, n);
    c->indexCount = indexCount;
    c->instanceCount = instanceCount;
    c->firstIndex = firstIndex;
    ++m_drawCount;
}

void CommandList::writeTimestamp(QueryPoolHandle pool, uint32_t index) {
    constexpr size_t n = align8(sizeof(cmd::WriteTimestamp));
    auto* c = emit<cmd::WriteTimestamp>(push(n), cmd::Op::WriteTimestamp, n);
    c->pool = pool;
    c->index = index;
}

void CommandList::pushDebugGroup(const char* name) {
    constexpr size_t n = align8(sizeof(cmd::PushDebugGroup));
    emit<cmd::PushDebugGroup>(push(n), cmd::Op::PushDebugGroup, n)->name = name;
}

void CommandList::popDebugGroup() {
    constexpr size_t n = align8(sizeof(cmd::PopDebugGroup));
    emit<cmd::PopDebugGroup>(push(n), cmd::Op::PopDebugGroup, n);
}

} // namespace rhi
