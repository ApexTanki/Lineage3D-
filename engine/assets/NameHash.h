#pragma once

#include <cstdint>
#include <string_view>

namespace assets {

// FNV-1a 32 of a node or clip name (ARCHITECTURE.md §7.2, §7.6): clips bind to nodes by it, so models that share node
// names share clips.
constexpr uint32_t nameHash(std::string_view name) {
    uint32_t h = 2166136261u;
    for (char c : name) {
        h ^= uint8_t(c);
        h *= 16777619u;
    }
    return h;
}

} // namespace assets
