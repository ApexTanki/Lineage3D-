#include "platform/FileSystem.h"

#include "core/Log.h"

#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_iostream.h>

#include <filesystem>
#include <system_error>

namespace platform {

std::string basePath() {
    const char* base = SDL_GetBasePath(); // owned by SDL
    return base ? std::string(base) : std::string();
}

core::Result<std::vector<std::byte>> readFile(const std::string& path) {
    SDL_IOStream* io = SDL_IOFromFile(path.c_str(), "rb");
    if (!io) return core::Error::NotFound;
    const Sint64 size = SDL_GetIOSize(io);
    if (size < 0) {
        SDL_CloseIO(io);
        return core::Error::IoError;
    }
    std::vector<std::byte> data(static_cast<size_t>(size));
    const size_t read = data.empty() ? 0 : SDL_ReadIO(io, data.data(), data.size());
    SDL_CloseIO(io);
    if (read != data.size()) {
        FL_LOG_ERROR("Short read on %s (%zu of %zu bytes)", path.c_str(), read, data.size());
        return core::Error::IoError;
    }
    return data;
}

core::Result<void> writeFile(const std::string& path, std::span<const std::byte> data) {
    SDL_IOStream* io = SDL_IOFromFile(path.c_str(), "wb");
    if (!io) return core::Error::IoError;
    const size_t written = data.empty() ? 0 : SDL_WriteIO(io, data.data(), data.size());
    const bool closed = SDL_CloseIO(io);
    if (written != data.size() || !closed) return core::Error::IoError;
    return {};
}

namespace {
// Paths are UTF-8 (SDL's convention); char8_t tells std::filesystem so.
std::filesystem::path fsPath(const std::string& utf8) {
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()));
}
} // namespace

core::Result<void> writeFileAtomic(const std::string& path, std::span<const std::byte> data) {
    const std::string temporary = path + ".tmp";
    std::error_code ec;
    if (!writeFile(temporary, data).ok()) {
        std::filesystem::remove(fsPath(temporary), ec);
        return core::Error::IoError;
    }
    std::filesystem::rename(fsPath(temporary), fsPath(path), ec); // replaces an existing file
    if (ec) {
        std::filesystem::remove(fsPath(temporary), ec);
        return core::Error::IoError;
    }
    return {};
}

bool createDirectories(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(fsPath(path), ec);
    return std::filesystem::is_directory(fsPath(path), ec);
}

bool removeAll(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove_all(fsPath(path), ec);
    return !std::filesystem::exists(fsPath(path), ec);
}

} // namespace platform
