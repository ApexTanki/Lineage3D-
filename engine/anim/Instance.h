#pragma once

#include "anim/Clip.h"

#include <vector>

namespace anim {

inline constexpr uint32_t kMaxLayers = 4;

// A clip to play: the library view, the library's binding to this instance's model, and the clip index. The library and
// the binding must outlive their use (the renderer's animation system owns both).
struct ClipRef {
    const assets::AnimView* library = nullptr;
    const LibraryBinding* binding = nullptr;
    uint32_t clip = 0;
    explicit operator bool() const { return library && binding && clip < library->clips.size(); }
};

// The animation state of one model instance (ARCHITECTURE.md §5.5, M6 plan D6): up to kMaxLayers layers evaluated
// bottom to top. Each layer plays a clip and, during a cross-fade, the clip it fades out of; it has a weight (fades the
// whole layer in or out), an optional per-node mask (a turret layer) and may be additive (P1: its clip's motion relative
// to the clip's first frame is added on top). Node overrides (game-driven turret yaw, gun pitch) replace a node's local
// transform after the layers. bind() allocates; nothing else does, so instances evaluate on job threads.
class Instance {
public:
    // Sizes the instance for a model: its parents-first hierarchy and rest pose (the node table's TRS).
    void bind(std::span<const int32_t> parent, std::span<const Transform> rest);
    uint32_t nodeCount() const { return uint32_t(m_rest.size()); }

    // Starts `clip` on a layer. With fadeSeconds > 0 the layer cross-fades from whatever it played (or, if it was silent,
    // fades its weight in); 0 switches at once.
    void play(uint32_t layer, const ClipRef& clip, float fadeSeconds, Playback playback, float speed = 1.0f);
    // Fades the layer's weight to 0 over fadeSeconds, then forgets its clips.
    void stop(uint32_t layer, float fadeSeconds);
    void setWeight(uint32_t layer, float weight, float fadeSeconds = 0.0f);
    // Per-node weights in [0, 1] (caller-owned, one per node; see maskBranch); empty = every node at full weight.
    void setMask(uint32_t layer, std::span<const float> mask);
    void setAdditive(uint32_t layer, bool additive);
    void setSpeed(uint32_t layer, float speed);
    void seek(uint32_t layer, float time); // the current clip's playback time (the fading-out clip keeps its own)

    void setOverride(uint32_t node, const Transform& local);
    void clearOverride(uint32_t node);

    // Advances clip times, cross-fades and layer weights.
    void advance(float dt);

    bool active(uint32_t layer) const;   // plays a clip with a weight above 0 (or fading in)
    bool finished(uint32_t layer) const; // its current clip is a one-shot that reached its end (and holds its last frame)
    float time(uint32_t layer) const;
    float weight(uint32_t layer) const;
    float fade(uint32_t layer) const;    // cross-fade progress, 1 when not fading
    const ClipRef& clip(uint32_t layer) const;

    // The pose in model space (node → model). Writes min(nodeCount, out.size()) matrices; no allocation.
    void evaluate(std::span<core::Affine3x4> outModel);
    std::span<const Transform> localPose() const { return m_pose; } // as of the last evaluate()

private:
    struct Playing {
        ClipRef clip;
        float time = 0.0f;
        float speed = 1.0f;
        Playback playback = Playback::Loop;
    };
    struct Layer {
        Playing current, previous; // previous: the clip a cross-fade leaves (clip empty when not fading)
        float fade = 1.0f;         // weight of current against previous
        float fadeRate = 0.0f;     // per second
        float weight = 1.0f;
        float weightTarget = 1.0f;
        float weightRate = 0.0f;   // per second, toward weightTarget
        std::span<const float> mask;
        bool additive = false;
    };

    void advancePlaying(Playing& p, float dt) const;
    void applyLayer(const Layer& layer);
    void applyAdditive(const Playing& p, std::span<const float> mask, float weight);

    std::vector<int32_t> m_parent;
    std::vector<Transform> m_rest, m_pose, m_a, m_b, m_override;
    std::vector<uint8_t> m_hasOverride;
    Layer m_layers[kMaxLayers];
};

} // namespace anim
