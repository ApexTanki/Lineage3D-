#pragma once

#include "rhi/Types.h"

#include <atomic>

namespace rhi {

// Per-frame linear sub-allocator over one buffer (ARCHITECTURE.md §4.8). Each frame gets a fresh window:
// Vulkan points it into a persistently mapped buffer partition; GL points it at CPU staging that is
// uploaded once at submit. allocate() is a non-virtual atomic fetch_add, so job threads may allocate
// concurrently, and sizes are rounded up to the offset alignment.
class TransientRing {
public:
    struct Allocation {
        void* cpu = nullptr;
        BufferSlice gpu;
    };

    void init(BufferHandle buffer, uint32_t capacityPerFrame, uint32_t alignment);
    // Starts a frame: CPU writes go to `cpu`; GPU offsets are reported relative to `gpuBaseOffset`.
    void beginFrame(std::byte* cpu, uint32_t gpuBaseOffset);

    // Returns {nullptr, {}} when this frame's budget is exhausted.
    Allocation allocate(uint32_t bytes);

    BufferHandle buffer() const { return m_buffer; }
    uint32_t usedBytes() const { return m_used.load(std::memory_order_relaxed); }
    uint32_t capacityBytes() const { return m_capacity; }
    uint32_t alignment() const { return m_alignment; }
    uint32_t gpuBaseOffset() const { return m_gpuBase; }
    std::byte* cpuBase() const { return m_cpu; }

private:
    BufferHandle m_buffer;
    std::byte* m_cpu = nullptr;
    uint32_t m_gpuBase = 0;
    uint32_t m_capacity = 0;
    uint32_t m_alignment = 16;
    std::atomic<uint32_t> m_used{0};
};

} // namespace rhi
