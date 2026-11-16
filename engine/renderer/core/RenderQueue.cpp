#include "renderer/core/RenderQueue.h"

#include <algorithm>

namespace renderer {

RenderQueue::RenderQueue(uint32_t capacity) : m_keys(capacity), m_items(capacity), m_histogram(256) {
    for (int i = 0; i < 2; ++i) {
        m_sortKeys[i].resize(capacity);
        m_sortIndex[i].resize(capacity);
    }
    reset();
}

void RenderQueue::reset() {
    m_reserved.store(0, std::memory_order_relaxed);
    m_dropped.store(0, std::memory_order_relaxed);
    m_validEnd.store(capacity(), std::memory_order_relaxed);
    m_size = 0;
    m_sorted = 0;
}

uint32_t RenderQueue::reserve(uint32_t count) {
    const uint32_t first = m_reserved.fetch_add(count, std::memory_order_relaxed);
    if (first > capacity() || count > capacity() - first) {
        m_dropped.fetch_add(count, std::memory_order_relaxed);
        // Ranges are handed out in order, so every slot before the first failed range was filled and none after.
        uint32_t end = m_validEnd.load(std::memory_order_relaxed);
        while (first < end && !m_validEnd.compare_exchange_weak(end, first, std::memory_order_relaxed)) {}
        return kFull;
    }
    return first;
}

void RenderQueue::sort() {
    m_size = std::min(m_reserved.load(std::memory_order_relaxed), m_validEnd.load(std::memory_order_relaxed));
    const uint32_t n = m_size;
    uint32_t src = 0;
    for (uint32_t i = 0; i < n; ++i) {
        m_sortKeys[0][i] = m_keys[i];
        m_sortIndex[0][i] = i;
    }
    for (uint32_t shift = 0; shift < 64; shift += 8) {
        std::fill(m_histogram.begin(), m_histogram.end(), 0u);
        for (uint32_t i = 0; i < n; ++i) ++m_histogram[(m_sortKeys[src][i] >> shift) & 0xFF];
        // A pass where every key has the same byte would only copy.
        if (n == 0 || m_histogram[(m_sortKeys[src][0] >> shift) & 0xFF] == n) continue;
        uint32_t sum = 0;
        for (uint32_t& h : m_histogram) {
            const uint32_t c = h;
            h = sum;
            sum += c;
        }
        const uint32_t dst = src ^ 1;
        for (uint32_t i = 0; i < n; ++i) { // stable: equal digits keep their order
            const uint64_t key = m_sortKeys[src][i];
            const uint32_t pos = m_histogram[(key >> shift) & 0xFF]++;
            m_sortKeys[dst][pos] = key;
            m_sortIndex[dst][pos] = m_sortIndex[src][i];
        }
        src = dst;
    }
    m_sorted = src;
}

} // namespace renderer
