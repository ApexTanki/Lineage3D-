#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace core {

// FNV-1a 64: cache keys and integrity checks of cache files (M9 plan D4, D5). Not cryptographic. A cache file therefore
// also stores what its key was made from (the Vulkan cache: the GPU, driver and cache UUID; a GL program binary: both
// shader hashes and the driver hash) and is checked against it, so a collision costs a cache miss, never a wrong program.
inline constexpr uint64_t kHashSeed = 14695981039346656037ull;

constexpr uint64_t hash64(std::string_view text, uint64_t h = kHashSeed) {
    for (char c : text) {
        h ^= uint8_t(c);
        h *= 1099511628211ull;
    }
    return h;
}

inline uint64_t hash64(std::span<const std::byte> data, uint64_t h = kHashSeed) {
    for (std::byte b : data) {
        h ^= uint8_t(b);
        h *= 1099511628211ull;
    }
    return h;
}

} // namespace core
