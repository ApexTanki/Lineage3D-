#include "platform/Timer.h"

#include <SDL3/SDL_timer.h>

namespace platform {

uint64_t nowTicks() {
    return SDL_GetPerformanceCounter();
}

double ticksToMilliseconds(uint64_t ticks) {
    static const double kMsPerTick = 1000.0 / double(SDL_GetPerformanceFrequency());
    return double(ticks) * kMsPerTick;
}

void sleepMilliseconds(uint32_t ms) {
    SDL_Delay(ms);
}

} // namespace platform
