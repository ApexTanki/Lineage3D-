#include "rhi/TransientRing.h"

#include "core/Assert.h"

namespace rhi {

void TransientRing::init(BufferHandle buffer, uint32_t capacityPerFrame, uint32_t alignment) {
    FL_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0, "ring alignment must be a power of two");
    m_buffer = buffer;
    m_capacity = capacityPerFrame;
    m_alignment = alignment;
    m_used.store(0, std::memory_order_relaxed);
}

void TransientRing::beginFrame(std::byte* cpu, uint32_t gpuBaseOffset) {
    m_cpu = cpu;
    m_gpuBase = gpuBaseOffset;
    m_used.store(0, std::memory_order_relaxed);
}

TransientRing::Allocation TransientRing::allocate(uint32_t bytes) {
    const uint32_t size = (bytes + m_alignment - 1) & ~(m_alignment - 1);
    const uint32_t offset = m_used.fetch_add(size, std::memory_order_relaxed);
    if (offset + size > m_capacity) return {};
    Allocation a;
    a.cpu = m_cpu + offset;
    a.gpu = {m_buffer, m_gpuBase + offset, bytes};
    return a;
}

} // namespace rhi
