#pragma once

#include <cstddef>
#include <memory>
#include <span>

namespace core {

// Bump allocator over one fixed block, reset() at the start of every frame. It never touches the
// heap after construction, which is how per-frame scratch data stays allocation-free.
class LinearAllocator {
public:
    explicit LinearAllocator(size_t capacityBytes);

    // Returns nullptr when exhausted; the caller decides how to degrade.
    void* allocate(size_t bytes, size_t alignment = 16);

    template <typename T>
    std::span<T> allocateArray(size_t count) {
        void* memory = allocate(sizeof(T) * count, alignof(T));
        return memory ? std::span<T>(static_cast<T*>(memory), count) : std::span<T>();
    }

    void reset() { m_used = 0; }
    size_t used() const { return m_used; }
    size_t capacity() const { return m_capacity; }

private:
    std::unique_ptr<std::byte[]> m_memory;
    size_t m_capacity = 0;
    size_t m_used = 0;
};

} // namespace core
