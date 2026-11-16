#include "core/Log.h"

#include <cstdio>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace core {
namespace {

const char* levelPrefix(LogLevel level) {
    switch (level) {
    case LogLevel::Debug: return "[debug] ";
    case LogLevel::Info: return "[info]  ";
    case LogLevel::Warning: return "[warn]  ";
    case LogLevel::Error: return "[error] ";
    }
    return "";
}

void defaultSink(LogLevel level, const char* message, void*) {
    std::fprintf(stderr, "%s%s\n", levelPrefix(level), message);
#if defined(_WIN32)
    OutputDebugStringA(levelPrefix(level));
    OutputDebugStringA(message);
    OutputDebugStringA("\n");
#endif
}

// Loader threads log too, so the sink call is serialized.
std::mutex g_mutex;
LogSink g_sink = defaultSink;
void* g_user = nullptr;

} // namespace

void setLogSink(LogSink sink, void* user) {
    std::lock_guard lock(g_mutex);
    g_sink = sink ? sink : defaultSink;
    g_user = sink ? user : nullptr;
}

void logMessageV(LogLevel level, const char* format, va_list args) {
    char buffer[2048];
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    std::lock_guard lock(g_mutex);
    g_sink(level, buffer, g_user);
}

void logMessage(LogLevel level, const char* format, ...) {
    va_list args;
    va_start(args, format);
    logMessageV(level, format, args);
    va_end(args);
}

} // namespace core
