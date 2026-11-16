#pragma once

#include "assets/AnimFormat.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace assets {

// Validated views into a loaded .fanim file. openAnim() checks every count, range and key: finite values, normalized
// rotations, and key counts of 1 or the clip's frame count, so sampling needs no further checks.
struct AnimView {
    AnimHeader header{};
    std::span<const AnimClip> clips;
    std::span<const AnimTrack> tracks;
    std::span<const float> keys;
};

bool openAnim(std::span<const std::byte> file, AnimView& out, std::string* error);

// The frame count a clip of this duration needs: round(duration · rate) + 1.
uint32_t animFrameCount(float duration, float sampleRate);
// The rate a clip is resampled at so that its keys span exactly [0, duration]: (frames − 1) / duration with frames =
// animFrameCount(duration, nominalRate), so the last key lands on the clip's end (a loop closes without a pop) and the
// rate stays within half a frame of the nominal one. Zero-length clips keep the nominal rate.
float animEffectiveRate(float duration, float nominalRate = kAnimSampleRate);

// In-memory clip library for the writer (fl_assetc, tests).
struct AnimData {
    struct Track {
        uint32_t nodeHash = 0;
        AnimChannel channel = AnimChannel::Translation;
        std::vector<float> keys; // animChannelFloats(channel) per key; 1 key or the clip's frame count
    };
    struct Clip {
        std::string name;
        float duration = 0.0f;
        float sampleRate = kAnimSampleRate;
        std::vector<Track> tracks;
    };
    std::vector<Clip> clips;
};

// Encodes a library; false (with a message) when a clip or track is inconsistent.
bool writeAnim(const AnimData& data, std::vector<std::byte>& out, std::string& error);

} // namespace assets
