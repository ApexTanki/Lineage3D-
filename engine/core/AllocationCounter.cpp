#include "core/AllocationCounter.h"

#include <cstdlib>
#include <new>

namespace core {
namespace {

thread_local bool t_counting = false;
thread_local uint64_t t_count = 0;
thread_local uint64_t t_uiCount = 0;
thread_local uint64_t t_sdlCount = 0;

void countSdl() {
    if constexpr (kAllocationCounterEnabled) {
        if (t_counting) ++t_sdlCount;
    }
}

} // namespace

void beginAllocationScope() {
    t_count = 0;
    t_uiCount = 0;
    t_sdlCount = 0;
    t_counting = true;
}

uint64_t lastScopeSdlAllocations() {
    return t_sdlCount;
}

uint64_t endAllocationScope() {
    t_counting = false;
    return t_count;
}

uint64_t lastScopeDebugUiAllocations() {
    return t_uiCount;
}

void* countedDebugUiMalloc(size_t size) {
    if constexpr (kAllocationCounterEnabled) {
        if (t_counting) ++t_uiCount;
    }
    return std::malloc(size);
}

void countAllocation() {
    if constexpr (kAllocationCounterEnabled) {
        if (t_counting) ++t_count;
    }
}

void* countedMalloc(size_t size) {
    countSdl();
    return std::malloc(size);
}

void* countedCalloc(size_t count, size_t size) {
    countSdl();
    return std::calloc(count, size);
}

void* countedRealloc(void* memory, size_t size) {
    countSdl();
    return std::realloc(memory, size);
}

void countedFree(void* memory) {
    std::free(memory);
}

} // namespace core

#if FL_DEV
// Replacement global allocation functions ([new.delete]): the only change is counting. They live in the same
// translation unit as beginAllocationScope(), so every binary that opens a counting scope links them.
namespace {

void* allocateOrThrow(std::size_t size) {
    core::countAllocation();
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc(); // the standard contract of operator new
}

void* allocateAlignedOrThrow(std::size_t size, std::align_val_t alignment) {
    core::countAllocation();
    if (void* p = _aligned_malloc(size ? size : 1, static_cast<std::size_t>(alignment))) return p;
    throw std::bad_alloc();
}

} // namespace

void* operator new(std::size_t size) { return allocateOrThrow(size); }
void* operator new[](std::size_t size) { return allocateOrThrow(size); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    core::countAllocation();
    return std::malloc(size ? size : 1);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    core::countAllocation();
    return std::malloc(size ? size : 1);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }

void* operator new(std::size_t size, std::align_val_t a) { return allocateAlignedOrThrow(size, a); }
void* operator new[](std::size_t size, std::align_val_t a) { return allocateAlignedOrThrow(size, a); }
void* operator new(std::size_t size, std::align_val_t a, const std::nothrow_t&) noexcept {
    core::countAllocation();
    return _aligned_malloc(size ? size : 1, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t size, std::align_val_t a, const std::nothrow_t&) noexcept {
    core::countAllocation();
    return _aligned_malloc(size ? size : 1, static_cast<std::size_t>(a));
}
void operator delete(void* p, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { _aligned_free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { _aligned_free(p); }
#endif
