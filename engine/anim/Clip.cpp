#include "anim/Clip.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace anim {

int32_t findClip(const assets::AnimView& library, std::string_view name) {
    for (size_t i = 0; i < library.clips.size(); ++i)
        if (name == std::string_view(library.clips[i].name)) return int32_t(i);
    return -1;
}

LibraryBinding bindLibrary(const assets::AnimView& library, std::span<const uint32_t> nodeHashes) {
    LibraryBinding b;
    b.nodeCount = uint32_t(nodeHashes.size());
    std::unordered_map<uint32_t, int32_t> byHash;
    for (size_t i = 0; i < nodeHashes.size(); ++i) byHash.emplace(nodeHashes[i], int32_t(i)); // the first node wins a collision
    b.trackNode.assign(library.tracks.size(), -1);
    b.channels.assign(library.clips.size() * nodeHashes.size(), 0);
    for (size_t c = 0; c < library.clips.size(); ++c) {
        const assets::AnimClip& clip = library.clips[c];
        for (uint32_t t = clip.firstTrack; t < clip.firstTrack + clip.trackCount; ++t) {
            const assets::AnimTrack& track = library.tracks[t];
            const auto it = byHash.find(track.nodeHash);
            if (it == byHash.end()) {
                ++b.unboundTracks;
                continue;
            }
            b.trackNode[t] = it->second;
            b.channels[c * nodeHashes.size() + size_t(it->second)] |= uint8_t(1u << track.channel);
        }
    }
    return b;
}

float clipTime(float time, float duration, Playback playback) {
    if (!(duration > 0.0f)) return 0.0f;
    if (playback == Playback::Once) return std::clamp(time, 0.0f, duration);
    const float t = std::fmod(time, duration);
    return t < 0.0f ? t + duration : t;
}

void sampleClip(const assets::AnimView& library, uint32_t clip, const LibraryBinding& binding, float time,
                std::span<Transform> pose) {
    if (clip >= library.clips.size()) return;
    const assets::AnimClip& c = library.clips[clip];
    if (!(time >= 0.0f)) time = 0.0f; // also catches NaN, which the integer conversion below must never see
    // Keys are evenly spaced (resampled offline), so the pair around `time` is found by arithmetic.
    const float frame = std::clamp(time * c.sampleRate, 0.0f, float(c.frameCount - 1));
    const uint32_t i0 = std::min(uint32_t(frame), c.frameCount - 1);
    const uint32_t i1 = std::min(i0 + 1, c.frameCount - 1);
    const float alpha = frame - float(i0);
    for (uint32_t t = c.firstTrack; t < c.firstTrack + c.trackCount; ++t) {
        if (t >= binding.trackNode.size()) break; // a binding made for another library (a stale reference)
        const int32_t node = binding.trackNode[t];
        if (node < 0 || size_t(node) >= pose.size()) continue;
        const assets::AnimTrack& track = library.tracks[t];
        const float* keys = library.keys.data() + track.firstFloat;
        const bool constant = track.keyCount == 1;
        Transform& out = pose[size_t(node)];
        switch (assets::AnimChannel(track.channel)) {
        case assets::AnimChannel::Rotation: {
            const float* a = keys + 4 * (constant ? 0 : i0);
            const float* b = keys + 4 * (constant ? 0 : i1);
            const glm::quat qa(a[3], a[0], a[1], a[2]), qb(b[3], b[0], b[1], b[2]);
            out.rotation = constant ? qa : nlerp(qa, qb, alpha);
            break;
        }
        case assets::AnimChannel::Translation:
        case assets::AnimChannel::Scale: {
            const float* a = keys + 3 * (constant ? 0 : i0);
            const float* b = keys + 3 * (constant ? 0 : i1);
            const glm::vec3 va(a[0], a[1], a[2]), vb(b[0], b[1], b[2]);
            const glm::vec3 v = constant ? va : va + (vb - va) * alpha;
            if (assets::AnimChannel(track.channel) == assets::AnimChannel::Translation) out.translation = v;
            else out.scale = v;
            break;
        }
        }
    }
}

} // namespace anim
