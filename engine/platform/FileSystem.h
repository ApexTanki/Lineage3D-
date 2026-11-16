#pragma once

#include "core/Result.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace platform {

// Directory of the running executable, with a trailing separator. Runtime data lives in "<base>data/".
std::string basePath();

// Whole-file I/O for load-time use. (The zero-allocation rule covers the render loop, not loading.)
core::Result<std::vector<std::byte>> readFile(const std::string& path);
core::Result<void> writeFile(const std::string& path, std::span<const std::byte> data);
// Writes "<path>.tmp", then renames it over `path`: a crash or a full disk mid-write leaves the old file or none, never a
// torn one (cache files, M9 plan D4, D5).
core::Result<void> writeFileAtomic(const std::string& path, std::span<const std::byte> data);
// Creates the directory and its parents; true if it exists afterwards.
bool createDirectories(const std::string& path);
// Removes a file or a directory with everything in it; true if nothing is left at `path`.
bool removeAll(const std::string& path);

} // namespace platform
