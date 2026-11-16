#include "core/LinearAllocator.h"

#include "core/Assert.h"

#include <cstdint>

namespace core {

LinearAllocator::LinearAllocator(size_t capacityBytes)
    : m_memory(std::make_unique<std::byte[]>(capacityBytes)), m_capacity(capacityBytes) {}

void* LinearAllocator::allocate(size_t bytes, size_t alignment) {
    FL_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0, "alignment must be a power of two");
    const uintptr_t base = reinterpret_cast<uintptr_t>(m_memory.get());
    const uintptr_t aligned = (base + m_used + alignment - 1) & ~(uintptr_t(alignment) - 1);
    const size_t offset = aligned - base;
    if (offset + bytes > m_capacity) return nullptr;
    m_used = offset + bytes;
    return m_memory.get() + offset;
}

} // namespace core
