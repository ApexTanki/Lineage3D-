#include "assets/AnimFile.h"

#include <cmath>
#include <cstring>

namespace assets {
namespace {

bool fail(std::string* error, const char* message) {
    if (error) *error = message;
    return false;
}

size_t align16(size_t v) {
    return (v + 15) & ~size_t(15);
}

template <typename T>
bool section(std::span<const std::byte> file, uint32_t offset, uint32_t count, std::span<const T>& out) {
    if (count == 0) {
        out = {};
        return true;
    }
    if (offset % 16 != 0) return false;
    if (size_t(offset) + size_t(count) * sizeof(T) > file.size()) return false;
    out = {reinterpret_cast<const T*>(file.data() + offset), count};
    return true;
}

template <typename T>
void put(std::vector<std::byte>& out, size_t offset, const T* data, size_t count) {
    if (count) std::memcpy(out.data() + offset, data, sizeof(T) * count);
}

} // namespace

uint32_t animFrameCount(float duration, float sampleRate) {
    return uint32_t(std::lround(double(duration) * double(sampleRate))) + 1u;
}

float animEffectiveRate(float duration, float nominalRate) {
    const uint32_t frames = animFrameCount(duration, nominalRate);
    return duration > 0.0f && frames > 1 ? float(double(frames - 1) / double(duration)) : nominalRate;
}

bool openAnim(std::span<const std::byte> file, AnimView& out, std::string* error) {
    if (file.size() < sizeof(AnimHeader)) return fail(error, "file smaller than its header");
    if (reinterpret_cast<uintptr_t>(file.data()) % 16 != 0) return fail(error, "file buffer is not 16-byte aligned");
    std::memcpy(&out.header, file.data(), sizeof(AnimHeader));
    const AnimHeader& h = out.header;
    if (std::memcmp(h.magic, kAnimMagic, 4) != 0) return fail(error, "bad magic");
    if (h.version != kAnimVersion) return fail(error, "unsupported animation version");
    if (h.fileSize != file.size()) return fail(error, "size mismatch (truncated file?)");
    if (!section(file, h.clipOffset, h.clipCount, out.clips) || !section(file, h.trackOffset, h.trackCount, out.tracks) ||
        !section(file, h.floatOffset, h.floatCount, out.keys))
        return fail(error, "section out of bounds or misaligned");

    for (const AnimClip& c : out.clips) {
        if (std::memchr(c.name, 0, sizeof(c.name)) == nullptr) return fail(error, "unterminated clip name");
        if (!(c.duration >= 0.0f && c.duration < 1e6f)) return fail(error, "invalid clip duration");
        if (!(c.sampleRate > 0.0f && c.sampleRate <= 1000.0f)) return fail(error, "invalid sample rate");
        if (c.frameCount == 0 || c.frameCount > kMaxClipFrames || c.frameCount != animFrameCount(c.duration, c.sampleRate))
            return fail(error, "clip frame count doesn't match its duration");
        if (c.firstTrack > h.trackCount || c.trackCount > h.trackCount - c.firstTrack) return fail(error, "clip track range out of bounds");
        for (uint32_t t = c.firstTrack; t < c.firstTrack + c.trackCount; ++t) {
            const AnimTrack& track = out.tracks[t];
            if (track.channel > uint8_t(AnimChannel::Scale)) return fail(error, "unknown track channel");
            if (track.keyCount != 1 && track.keyCount != c.frameCount) return fail(error, "track key count must be 1 or the frame count");
            const uint32_t floats = animChannelFloats(AnimChannel(track.channel));
            if (track.firstFloat > h.floatCount || uint64_t(track.keyCount) * floats > h.floatCount - track.firstFloat)
                return fail(error, "track keys out of bounds");
            const float* k = out.keys.data() + track.firstFloat;
            for (uint32_t i = 0; i < track.keyCount * floats; ++i)
                if (!std::isfinite(k[i])) return fail(error, "non-finite key");
            if (AnimChannel(track.channel) == AnimChannel::Rotation)
                for (uint32_t i = 0; i < track.keyCount; ++i) {
                    const float* q = k + 4 * i;
                    const float len2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
                    if (std::fabs(len2 - 1.0f) > 2e-3f) return fail(error, "rotation key is not normalized");
                }
        }
    }
    return true;
}

bool writeAnim(const AnimData& data, std::vector<std::byte>& out, std::string& error) {
    std::vector<AnimClip> clips;
    std::vector<AnimTrack> tracks;
    std::vector<float> keys;
    for (const AnimData::Clip& c : data.clips) {
        if (c.name.empty() || c.name.size() >= kModelNameLength) {
            error = "clip '" + c.name + "': name empty or longer than " + std::to_string(kModelNameLength - 1) + " characters";
            return false;
        }
        AnimClip clip{};
        std::memcpy(clip.name, c.name.data(), c.name.size());
        clip.duration = c.duration;
        clip.sampleRate = c.sampleRate;
        clip.frameCount = animFrameCount(c.duration, c.sampleRate);
        // The reader's limits, enforced when cooking rather than discovered at load time.
        if (!(c.duration >= 0.0f && c.duration < 1e6f) || !(c.sampleRate > 0.0f && c.sampleRate <= 1000.0f) || clip.frameCount > kMaxClipFrames) {
            error = "clip '" + c.name + "': duration, sample rate or frame count out of range";
            return false;
        }
        clip.firstTrack = uint32_t(tracks.size());
        clip.trackCount = uint32_t(c.tracks.size());
        for (const AnimData::Track& t : c.tracks) {
            const uint32_t floats = animChannelFloats(t.channel);
            const size_t count = t.keys.size() / floats;
            if (t.keys.size() % floats != 0 || (count != 1 && count != clip.frameCount)) {
                error = "clip '" + c.name + "': a track has " + std::to_string(count) + " keys, not 1 or " + std::to_string(clip.frameCount);
                return false;
            }
            AnimTrack track{};
            track.nodeHash = t.nodeHash;
            track.channel = uint8_t(t.channel);
            track.keyCount = uint32_t(count);
            track.firstFloat = uint32_t(keys.size());
            keys.insert(keys.end(), t.keys.begin(), t.keys.end());
            tracks.push_back(track);
        }
        clips.push_back(clip);
    }

    AnimHeader h{};
    std::memcpy(h.magic, kAnimMagic, 4);
    h.version = kAnimVersion;
    h.clipCount = uint32_t(clips.size());
    h.trackCount = uint32_t(tracks.size());
    h.floatCount = uint32_t(keys.size());
    size_t offset = align16(sizeof(AnimHeader));
    h.clipOffset = uint32_t(offset);
    offset = align16(offset + clips.size() * sizeof(AnimClip));
    h.trackOffset = uint32_t(offset);
    offset = align16(offset + tracks.size() * sizeof(AnimTrack));
    h.floatOffset = uint32_t(offset);
    h.fileSize = uint32_t(offset + keys.size() * sizeof(float));

    out.assign(h.fileSize, std::byte{0});
    put(out, 0, &h, 1);
    put(out, h.clipOffset, clips.data(), clips.size());
    put(out, h.trackOffset, tracks.data(), tracks.size());
    put(out, h.floatOffset, keys.data(), keys.size());
    return true;
}

} // namespace assets
