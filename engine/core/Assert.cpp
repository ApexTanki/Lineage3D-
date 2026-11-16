#include "core/Assert.h"

#include "core/Log.h"

#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace core {

void assertFailed(const char* expression, const char* message, const char* file, int line) {
    logMessage(LogLevel::Error, "Assertion failed: %s (%s) at %s:%d", message, expression, file, line);
#if defined(_WIN32)
    if (IsDebuggerPresent()) __debugbreak();
#endif
    std::abort();
}

} // namespace core
