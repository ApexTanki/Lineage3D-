#pragma once

#include <cstdarg>
#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#define FL_PRINTF_FORMAT(fmtIndex, argIndex) __attribute__((format(printf, fmtIndex, argIndex)))
#else
#define FL_PRINTF_FORMAT(fmtIndex, argIndex)
#endif

namespace core {

enum class LogLevel : uint8_t { Debug, Info, Warning, Error };

using LogSink = void (*)(LogLevel level, const char* message, void* user);

// One of the two documented process-wide facilities (ARCHITECTURE.md A17). The default sink writes to
// stderr and, on Windows, to the debugger output. Messages are formatted into a stack buffer (no heap).
void setLogSink(LogSink sink, void* user);
void logMessage(LogLevel level, const char* format, ...) FL_PRINTF_FORMAT(2, 3);
void logMessageV(LogLevel level, const char* format, va_list args);

} // namespace core

#define FL_LOG_DEBUG(...) ::core::logMessage(::core::LogLevel::Debug, __VA_ARGS__)
#define FL_LOG_INFO(...) ::core::logMessage(::core::LogLevel::Info, __VA_ARGS__)
#define FL_LOG_WARN(...) ::core::logMessage(::core::LogLevel::Warning, __VA_ARGS__)
#define FL_LOG_ERROR(...) ::core::logMessage(::core::LogLevel::Error, __VA_ARGS__)
