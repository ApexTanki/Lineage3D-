#pragma once

#include "anim/Pose.h"
#include "assets/AnimFile.h"

#include <string_view>
#include <vector>

// Clips (ARCHITECTURE.md §7.6, M6 plan D3, D7): a loaded .fanim library is sampled through its validated view, and
// bound once per model so that sampling writes straight into the model's node array.

namespace anim {

enum class Playback : uint8_t { Loop, Once };

// Channel bits of LibraryBinding::channels.
inline constexpr uint8_t kChannelTranslation = 1, kChannelRotation = 2, kChannelScale = 4;

// Index of the clip with this name, or −1.
int32_t findClip(const assets::AnimView& library, std::string_view name);

// A library's tracks mapped onto one model's nodes by name hash (M6 plan D7). Built once per (library, model) pair.
struct LibraryBinding {
    std::vector<int32_t> trackNode; // per track of the library: the node it drives, or −1 (no such node)
    std::vector<uint8_t> channels;  // [clip × nodeCount + node]: the kChannel* bits that clip animates on that node
    uint32_t nodeCount = 0;
    uint32_t unboundTracks = 0;     // tracks whose node the model lacks (reported once by the caller)
};

LibraryBinding bindLibrary(const assets::AnimView& library, std::span<const uint32_t> nodeHashes);

// Playback time → clip time: wrapped into [0, duration) for loops, clamped to [0, duration] for one-shots.
float clipTime(float time, float duration, Playback playback);

// Writes the channels `clip` animates into `pose` (indexed by node); every other value stays as it is. O(1) per track:
// two keys and a lerp or nlerp.
void sampleClip(const assets::AnimView& library, uint32_t clip, const LibraryBinding& binding, float time,
                std::span<Transform> pose);

} // namespace anim
