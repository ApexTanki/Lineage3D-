#pragma once

#include <cstdint>

namespace platform {

// High-resolution monotonic clock.
uint64_t nowTicks();
double ticksToMilliseconds(uint64_t ticks);
void sleepMilliseconds(uint32_t ms);

} // namespace platform
