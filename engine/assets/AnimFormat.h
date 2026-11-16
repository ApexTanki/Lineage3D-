#pragma once

// .fanim: a clip library (ARCHITECTURE.md §7.6, M6 plan D3): every animation of one source (a glTF file or a builtin
// rig), resampled offline at a fixed rate. Tracks target nodes by name hash (NameHash.h), so any model whose nodes carry
// those names can play the clips. Little-endian; read into memory once and validated by openAnim(); sections are
// 16-byte aligned so the views need no copies.
//
// Layout: AnimHeader | AnimClip[] | AnimTrack[] | float keys[]

#include "assets/ModelFormat.h"

#include <cstdint>

namespace assets {

inline constexpr char kAnimMagic[4] = {'F', 'A', 'N', 'M'};
inline constexpr uint16_t kAnimVersion = 1;
inline constexpr float kAnimSampleRate = 30.0f;     // the cooker's resampling rate (§7.6)
inline constexpr uint32_t kMaxClipFrames = 1u << 20; // per clip: about 9.7 hours at 30 Hz; bounds corrupt headers

enum class AnimChannel : uint8_t { Translation, Rotation, Scale };

// Floats per key of a channel: translation and scale xyz, rotation quaternion xyzw.
constexpr uint32_t animChannelFloats(AnimChannel c) { return c == AnimChannel::Rotation ? 4u : 3u; }

struct AnimHeader {
    char magic[4];
    uint16_t version;
    uint16_t flags;
    uint32_t fileSize;
    uint32_t clipCount, trackCount, floatCount;
    uint32_t clipOffset, trackOffset, floatOffset;
};

struct AnimClip {
    char name[kModelNameLength];
    float duration;      // seconds; the last frame sits at `duration`
    float sampleRate;    // frames per second
    uint32_t frameCount; // keys of a full track: round(duration · sampleRate) + 1
    uint32_t firstTrack, trackCount;
};

struct AnimTrack {
    uint32_t nodeHash;   // nameHash() of the target node
    uint8_t channel;     // AnimChannel
    uint8_t pad[3];
    uint32_t keyCount;   // 1 (constant) or the clip's frameCount
    uint32_t firstFloat; // into the key floats; animChannelFloats() per key
};

} // namespace assets
