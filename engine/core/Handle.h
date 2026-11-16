#pragma once

#include "core/Assert.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace core {

// 32-bit generational handle: by default a 20-bit slot index + 12-bit generation (ARCHITECTURE.md §5.1). High-churn pools
// with few slots take fewer index bits for more generations (M9 plan D8): 10 bits leave 22, about 4.2 million uses of a
// slot before it retires. Generation 0 is never issued, so a default-constructed handle is always invalid.
template <typename Tag, uint32_t IndexBits = 20>
struct Handle {
    static_assert(IndexBits >= 4 && IndexBits <= 28, "a handle needs index and generation bits");
    static constexpr uint32_t kIndexBits = IndexBits;
    static constexpr uint32_t kIndexMask = (1u << kIndexBits) - 1;
    static constexpr uint32_t kMaxGeneration = (1u << (32 - kIndexBits)) - 1;

    uint32_t bits = 0;

    constexpr uint32_t index() const { return bits & kIndexMask; }
    constexpr uint32_t generation() const { return bits >> kIndexBits; }
    constexpr explicit operator bool() const { return bits != 0; }
    friend constexpr bool operator==(Handle, Handle) = default;

    static constexpr Handle make(uint32_t index, uint32_t generation) {
        return Handle{(generation << kIndexBits) | (index & kIndexMask)};
    }
};

// Fixed-capacity slot pool: storage is allocated once and never grows. A slot whose generation
// reaches MaxGeneration is retired instead of reused, so a stale handle can never alias a newer
// object. (MaxGeneration is a parameter only so tests can reach retirement quickly; IndexBits must match the handle's.)
template <typename T, typename Tag, uint32_t MaxGeneration = Handle<Tag>::kMaxGeneration, uint32_t IndexBits = 20>
class HandlePool {
public:
    using HandleType = Handle<Tag, IndexBits>;
    static_assert(MaxGeneration <= HandleType::kMaxGeneration, "generations beyond what the handle can hold");

    explicit HandlePool(uint32_t capacity) : m_slots(capacity) {
        FL_VERIFY(capacity <= HandleType::kIndexMask + 1, "HandlePool capacity exceeds the handle's index range");
        m_free.reserve(capacity);
        for (uint32_t i = capacity; i-- > 0;) m_free.push_back(i);
    }

    HandlePool(const HandlePool&) = delete;
    HandlePool& operator=(const HandlePool&) = delete;

    // Returns an invalid handle when the pool is full; the caller logs and fails gracefully.
    HandleType create(T value) {
        if (m_free.empty()) return {};
        const uint32_t index = m_free.back();
        m_free.pop_back();
        Slot& slot = m_slots[index];
        slot.value = std::move(value);
        slot.alive = true;
        ++m_liveCount;
        return HandleType::make(index, slot.generation);
    }

    void destroy(HandleType handle) {
        Slot* slot = find(handle);
        FL_ASSERT(slot != nullptr, "destroy() of a stale or invalid handle");
        if (!slot) return;
        slot->value = T{};
        slot->alive = false;
        --m_liveCount;
        if (slot->generation < MaxGeneration) {
            ++slot->generation;
            m_free.push_back(handle.index());
        }
        // else: the slot is retired for good.
    }

    T* get(HandleType handle) {
        Slot* slot = find(handle);
        return slot ? &slot->value : nullptr;
    }
    const T* get(HandleType handle) const {
        const Slot* slot = const_cast<HandlePool*>(this)->find(handle);
        return slot ? &slot->value : nullptr;
    }

    // Visits every live object (used for teardown).
    template <typename Fn>
    void forEach(Fn&& fn) {
        for (uint32_t i = 0; i < m_slots.size(); ++i)
            if (m_slots[i].alive) fn(HandleType::make(i, m_slots[i].generation), m_slots[i].value);
    }

    uint32_t liveCount() const { return m_liveCount; }
    uint32_t capacity() const { return static_cast<uint32_t>(m_slots.size()); }
    uint32_t freeCount() const { return static_cast<uint32_t>(m_free.size()); }

private:
    struct Slot {
        T value{};
        uint32_t generation = 1;
        bool alive = false;
    };

    Slot* find(HandleType handle) {
        if (!handle) return nullptr;
        const uint32_t index = handle.index();
        if (index >= m_slots.size()) return nullptr;
        Slot& slot = m_slots[index];
        return (slot.alive && slot.generation == handle.generation()) ? &slot : nullptr;
    }

    std::vector<Slot> m_slots;
    std::vector<uint32_t> m_free;
    uint32_t m_liveCount = 0;
};

} // namespace core
