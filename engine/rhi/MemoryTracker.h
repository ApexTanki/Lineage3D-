#pragma once

#include "rhi/Descs.h"

#include <cstdint>

namespace rhi {

// GPU memory per category (M9 plan D6), counted by both backends from the descriptors, so the renderer's objects count
// the same on Vulkan and GL: Textures (sampled images), RenderTargets (color and depth targets), Meshes (vertex and index
// buffers) and Buffers (uniform, staging and transient buffers). The devices' own Buffers differ: Vulkan keeps a transient
// ring per frame in flight plus a 32 MiB staging buffer (58 MiB), GL one set of rings (13 MiB, twice that when fenced).
enum class GpuMemoryCategory : uint8_t { Textures, RenderTargets, Meshes, Buffers, Count };
inline constexpr uint32_t kGpuMemoryCategories = uint32_t(GpuMemoryCategory::Count);
const char* gpuMemoryCategoryName(GpuMemoryCategory category);

GpuMemoryCategory memoryCategory(const TextureDesc& desc);
GpuMemoryCategory memoryCategory(const BufferDesc& desc);
// Every mip, layer and sample; block-compressed formats count whole blocks.
uint64_t memoryBytes(const TextureDesc& desc);

// Usage is counted when an object is created and when destroy() is called (the memory itself is freed when the deferred
// destruction runs, two frames later). A category with a budget refuses an object that would take it over: the create
// call fails and the caller's out-of-memory path takes over (the renderer keeps its previous settings, M9 plan D6).
// Tests can make the Nth reservation fail, or every one from the Nth on (fl_demo --fail-gpu-alloc=N, --fail-gpu-alloc-from=N),
// to exercise those paths.
class MemoryTracker {
public:
    // False (logged) when the object would exceed its category's budget or a simulated failure is due.
    bool reserve(GpuMemoryCategory category, uint64_t bytes, const char* name);
    void release(GpuMemoryCategory category, uint64_t bytes);
    // Recounts reserved bytes under another category (no budget check): the devices' own transient rings are buffers,
    // whatever usage they were made with.
    void move(GpuMemoryCategory from, GpuMemoryCategory to, uint64_t bytes) {
        release(from, bytes);
        m_usage[uint32_t(to)] += bytes;
    }

    void setBudget(GpuMemoryCategory category, uint64_t bytes) { m_budget[uint32_t(category)] = bytes; } // 0 = none
    // Simulated failures (dev testing): the Nth reservation from now on fails, and with `andLater` every one after it.
    // Counting from the call keeps N the same on both backends, whose devices make different numbers of buffers first.
    void failAt(uint64_t reservation, bool andLater) {
        m_failFrom = reservation ? m_reservations + reservation : 0;
        m_failLater = andLater;
    }
    uint64_t usage(GpuMemoryCategory category) const { return m_usage[uint32_t(category)]; }
    uint64_t budget(GpuMemoryCategory category) const { return m_budget[uint32_t(category)]; }
    uint64_t total() const;
    uint32_t refusals() const { return m_refusals; } // budget refusals and simulated failures
    uint64_t reservations() const { return m_reservations; } // every buffer and texture asked for since start

private:
    uint64_t m_usage[kGpuMemoryCategories] = {};
    uint64_t m_budget[kGpuMemoryCategories] = {};
    uint64_t m_reservations = 0;
    uint64_t m_failFrom = 0; // 0 = never
    bool m_failLater = false;
    uint32_t m_refusals = 0;
};

// A reservation that releases itself unless the object it was for got made (keep()): every failure path of a create call
// gives its bytes back.
class MemoryReservation {
public:
    MemoryReservation(MemoryTracker& tracker, GpuMemoryCategory category, uint64_t bytes)
        : m_tracker(tracker), m_category(category), m_bytes(bytes) {}
    ~MemoryReservation() {
        if (!m_kept) m_tracker.release(m_category, m_bytes);
    }
    MemoryReservation(const MemoryReservation&) = delete;
    MemoryReservation& operator=(const MemoryReservation&) = delete;
    void keep() { m_kept = true; }

private:
    MemoryTracker& m_tracker;
    GpuMemoryCategory m_category;
    uint64_t m_bytes;
    bool m_kept = false;
};

} // namespace rhi
