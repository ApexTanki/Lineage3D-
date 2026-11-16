#pragma once

namespace core {
[[noreturn]] void assertFailed(const char* expression, const char* message, const char* file, int line);
} // namespace core

// FL_VERIFY is checked in every build (programmer errors that must never ship silently).
#define FL_VERIFY(condition, message)                                                                                  \
    do {                                                                                                               \
        if (!(condition)) ::core::assertFailed(#condition, message, __FILE__, __LINE__);                               \
    } while (0)

// FL_ASSERT is checked in developer builds only.
#if FL_DEV
#define FL_ASSERT(condition, message) FL_VERIFY(condition, message)
#else
#define FL_ASSERT(condition, message)                                                                                  \
    do {                                                                                                               \
        (void)sizeof(condition);                                                                                       \
    } while (0)
#endif
