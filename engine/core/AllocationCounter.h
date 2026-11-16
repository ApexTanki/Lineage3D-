#pragma once

#include <cstddef>
#include <cstdint>

namespace core {

// Dev-build heap-allocation counter (ARCHITECTURE.md §4.12). Global operator new/delete are replaced in
// AllocationCounter.cpp, and SDL's and ImGui's allocator hooks go through the counted* functions below, so
// every heap allocation made by a thread inside a counting scope is counted. The demo opens one scope per
// frame on the main thread; after warm-up the count must stay 0. With FL_DEV=0 nothing is counted.
void beginAllocationScope();    // the calling thread starts counting from 0
uint64_t endAllocationScope();  // stops counting; returns the engine allocations made since beginAllocationScope()
// Allocations by the dev-only debug UI (Dear ImGui) in the last scope. They are reported separately: its draw buffers
// grow (geometrically, rarely) when overlay text gets longer, which isn't part of the game's render loop.
uint64_t lastScopeDebugUiAllocations();
// SDL's allocations in the last scope (event-queue entries, which SDL pools after first use). Reported separately so
// a failure points at the right code.
uint64_t lastScopeSdlAllocations();
void countAllocation();         // operator new calls this
constexpr bool kAllocationCounterEnabled = FL_DEV != 0;

// malloc-family wrappers that count as SDL allocations, for SDL_SetMemoryFunctions.
void* countedMalloc(size_t size);
void* countedCalloc(size_t count, size_t size);
void* countedRealloc(void* memory, size_t size);
void countedFree(void* memory);
// The same for ImGui::SetAllocatorFunctions, counted as debug-UI allocations.
void* countedDebugUiMalloc(size_t size);

} // namespace core
