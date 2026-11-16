#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

namespace renderer {

// One draw (or one instanced run), 32 bytes, POD. Per-draw constants are built at record time from the
// source the item points at.
struct DrawItem {
    enum Flags : uint32_t {
        kInstanced = 1,  // perDrawOrInstance = vertex-ring offset of the first instance row
        kStaticRow = 2,  // perDrawOrInstance = index into the scene's static rows (instancing off)
        kDynamic = 4,    // perDrawOrInstance = dynamic renderable index
        kSkinned = 16,   // with kDynamic: a skinned renderable whose bone palette set 3 carries (M6 plan D9)
        kParticles = 32, // a particle emitter's draw (M7 plan D5): perDrawOrInstance = the particle renderer's draw index
        kReflectionShift = 8, // bits 8-11: the reflection slot bound in set 3 (M3 plan D8)
    };
    uint32_t mesh = 0;               // scene mesh reference
    uint16_t submesh = 0;
    uint16_t lod = 0;
    uint32_t material = 0;           // MaterialHandle bits
    uint32_t pipeline = 0;           // MaterialSystem pipeline slot
    uint32_t perDrawOrInstance = 0;
    uint32_t instanceCount = 1;
    uint32_t objectIndex = 0;
    uint32_t flags = 0;
};
static_assert(sizeof(DrawItem) == 32, "DrawItem is 32 bytes");

// Fixed-capacity queue, reset every frame (ARCHITECTURE.md §5.4). Workers reserve() a contiguous range for a whole
// batch with one atomic add, then fill it with set(); sort() is an LSD radix sort over (key, index) pairs, 8 bits
// per pass, skipping passes where every key has the same byte. Nothing allocates after construction.
class RenderQueue {
public:
    explicit RenderQueue(uint32_t capacity);

    void reset();
    // Thread-safe. Returns the first slot, or kFull when the queue can't hold `count` more items.
    uint32_t reserve(uint32_t count);
    void set(uint32_t slot, uint64_t key, const DrawItem& item) {
        m_keys[slot] = key;
        m_items[slot] = item;
    }
    void sort(); // after every set(); single-threaded

    uint32_t size() const { return m_size; }
    uint32_t capacity() const { return uint32_t(m_items.size()); }
    uint32_t dropped() const { return m_dropped.load(std::memory_order_relaxed); }
    uint64_t sortedKey(uint32_t i) const { return m_sortKeys[m_sorted][i]; }
    const DrawItem& sortedItem(uint32_t i) const { return m_items[m_sortIndex[m_sorted][i]]; }

    static constexpr uint32_t kFull = ~0u;

private:
    std::vector<uint64_t> m_keys;
    std::vector<DrawItem> m_items;
    std::vector<uint64_t> m_sortKeys[2];
    std::vector<uint32_t> m_sortIndex[2];
    std::vector<uint32_t> m_histogram; // 256 buckets
    uint32_t m_sorted = 0;
    uint32_t m_size = 0;
    std::atomic<uint32_t> m_reserved{0};
    std::atomic<uint32_t> m_dropped{0};
    std::atomic<uint32_t> m_validEnd{0}; // start of the first range that didn't fit (capacity when none failed)
};

} // namespace renderer
