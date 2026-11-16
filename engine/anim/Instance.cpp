#include "anim/Instance.h"

#include <algorithm>
#include <cmath>

namespace anim {
namespace {

float durationOf(const ClipRef& c) {
    return c ? c.library->clips[c.clip].duration : 0.0f;
}

const uint8_t* channelsOf(const ClipRef& c, size_t nodeCount) {
    if (!c || c.binding->nodeCount != nodeCount) return nullptr; // bound to another model: never sampled
    if ((size_t(c.clip) + 1) * nodeCount > c.binding->channels.size()) return nullptr; // a binding of another library
    return c.binding->channels.data() + size_t(c.clip) * nodeCount;
}

float maskWeight(std::span<const float> mask, size_t node) {
    if (mask.empty()) return 1.0f;
    return node < mask.size() ? std::clamp(mask[node], 0.0f, 1.0f) : 0.0f;
}

float approach(float value, float target, float step) {
    return value < target ? std::min(target, value + step) : std::max(target, value - step);
}

// A non-finite time, speed or delta would poison a layer for good (NaN stays NaN): such inputs count as `fallback`.
float finiteOr(float value, float fallback) {
    return std::isfinite(value) ? value : fallback;
}

} // namespace

void Instance::bind(std::span<const int32_t> parent, std::span<const Transform> rest) {
    const size_t n = std::min(parent.size(), rest.size());
    m_parent.assign(parent.begin(), parent.begin() + ptrdiff_t(n));
    m_rest.assign(rest.begin(), rest.begin() + ptrdiff_t(n));
    m_pose = m_a = m_b = m_rest;
    m_override.assign(n, Transform{});
    m_hasOverride.assign(n, 0);
    for (Layer& l : m_layers) {
        l = Layer{};
        l.weight = l.weightTarget = 0.0f; // silent until something plays
    }
}

void Instance::play(uint32_t layer, const ClipRef& clip, float fadeSeconds, Playback playback, float speed) {
    if (layer >= kMaxLayers || !clip) return;
    Layer& l = m_layers[layer];
    const bool audible = bool(l.current.clip) && l.weight > 0.0f;
    if (fadeSeconds > 0.0f && audible) {
        // Cross-fade from what the layer plays now. A fade already in progress loses its outgoing clip here (a small pop
        // of that clip's remaining share).
        l.previous = l.current;
        l.fade = 0.0f;
        l.fadeRate = 1.0f / fadeSeconds;
    } else {
        l.previous = Playing{};
        l.fade = 1.0f;
        l.fadeRate = 0.0f;
    }
    l.current = Playing{clip, 0.0f, finiteOr(speed, 1.0f), playback};
    l.weightTarget = 1.0f;
    if (fadeSeconds > 0.0f) {
        if (!audible) l.weight = 0.0f; // fade the layer in from the pose below it
        l.weightRate = 1.0f / fadeSeconds;
    } else {
        l.weight = 1.0f;
        l.weightRate = 0.0f;
    }
}

void Instance::stop(uint32_t layer, float fadeSeconds) {
    if (layer >= kMaxLayers) return;
    Layer& l = m_layers[layer];
    l.weightTarget = 0.0f;
    if (fadeSeconds > 0.0f) {
        l.weightRate = 1.0f / fadeSeconds;
    } else {
        l.weight = 0.0f;
        l.current = l.previous = Playing{};
        l.fade = 1.0f;
    }
}

void Instance::setWeight(uint32_t layer, float weight, float fadeSeconds) {
    if (layer >= kMaxLayers) return;
    Layer& l = m_layers[layer];
    l.weightTarget = std::clamp(weight, 0.0f, 1.0f);
    if (fadeSeconds > 0.0f) {
        l.weightRate = 1.0f / fadeSeconds;
    } else {
        l.weight = l.weightTarget;
        l.weightRate = 0.0f;
    }
}

void Instance::setMask(uint32_t layer, std::span<const float> mask) {
    if (layer < kMaxLayers) m_layers[layer].mask = mask;
}
void Instance::setAdditive(uint32_t layer, bool additive) {
    if (layer < kMaxLayers) m_layers[layer].additive = additive;
}
void Instance::setSpeed(uint32_t layer, float speed) {
    if (layer < kMaxLayers) m_layers[layer].current.speed = finiteOr(speed, 1.0f);
}
void Instance::seek(uint32_t layer, float time) {
    if (layer < kMaxLayers) m_layers[layer].current.time = finiteOr(time, 0.0f);
}

void Instance::setOverride(uint32_t node, const Transform& local) {
    if (node >= m_override.size()) return;
    m_override[node] = local;
    m_hasOverride[node] = 1;
}
void Instance::clearOverride(uint32_t node) {
    if (node < m_hasOverride.size()) m_hasOverride[node] = 0;
}

void Instance::advancePlaying(Playing& p, float dt) const {
    const float duration = durationOf(p.clip);
    p.time += dt * p.speed;
    // Loops wrap here so the time stays small (float precision over hours of play); one-shots hold at an end.
    p.time = clipTime(p.time, duration, p.playback);
}

void Instance::advance(float dt) {
    dt = std::clamp(finiteOr(dt, 0.0f), 0.0f, 10.0f); // a stalled frame doesn't skip whole clips at once
    for (Layer& l : m_layers) {
        if (!l.current.clip) continue;
        advancePlaying(l.current, dt);
        if (l.previous.clip) advancePlaying(l.previous, dt);
        if (l.fade < 1.0f) {
            l.fade = std::min(1.0f, l.fade + l.fadeRate * dt);
            if (l.fade >= 1.0f) l.previous = Playing{};
        }
        l.weight = l.weightRate > 0.0f ? approach(l.weight, l.weightTarget, l.weightRate * dt) : l.weightTarget;
        if (l.weight <= 0.0f && l.weightTarget <= 0.0f) {
            l.current = l.previous = Playing{};
            l.fade = 1.0f;
        }
    }
}

bool Instance::active(uint32_t layer) const {
    return layer < kMaxLayers && bool(m_layers[layer].current.clip) && (m_layers[layer].weight > 0.0f || m_layers[layer].weightTarget > 0.0f);
}

bool Instance::finished(uint32_t layer) const {
    if (layer >= kMaxLayers) return false;
    const Playing& p = m_layers[layer].current;
    if (!p.clip || p.playback != Playback::Once) return false;
    return p.speed >= 0.0f ? p.time >= durationOf(p.clip) : p.time <= 0.0f;
}

float Instance::time(uint32_t layer) const {
    return layer < kMaxLayers ? m_layers[layer].current.time : 0.0f;
}
float Instance::weight(uint32_t layer) const {
    return layer < kMaxLayers ? m_layers[layer].weight : 0.0f;
}
float Instance::fade(uint32_t layer) const {
    return layer < kMaxLayers ? m_layers[layer].fade : 1.0f;
}
const ClipRef& Instance::clip(uint32_t layer) const {
    static const ClipRef none;
    return layer < kMaxLayers ? m_layers[layer].current.clip : none;
}

// One non-additive layer: per node and channel, the incoming clip A and the outgoing clip B blend by the cross-fade; a
// channel only one of them animates fades against the pose below, so nothing pops when the fade ends. The result then
// blends onto the pose below by layer weight × mask.
void Instance::applyLayer(const Layer& l) {
    const size_t n = m_pose.size();
    const Playing& a = l.current;
    const Playing& b = l.previous;
    const uint8_t* chA = channelsOf(a.clip, n);
    const uint8_t* chB = l.fade < 1.0f ? channelsOf(b.clip, n) : nullptr;
    if (!chA) return;
    sampleClip(*a.clip.library, a.clip.clip, *a.clip.binding, clipTime(a.time, durationOf(a.clip), a.playback), m_a);
    if (chB) sampleClip(*b.clip.library, b.clip.clip, *b.clip.binding, clipTime(b.time, durationOf(b.clip), b.playback), m_b);
    const float fade = chB ? l.fade : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t ca = chA[i], cb = chB ? chB[i] : uint8_t(0);
        if (!(ca | cb)) continue;
        const float k = l.weight * maskWeight(l.mask, i);
        if (k <= 0.0f) continue;
        Transform& p = m_pose[i];
        auto vec = [&](glm::vec3& value, const glm::vec3& va, const glm::vec3& vb, uint8_t bit) {
            const bool ha = ca & bit, hb = cb & bit;
            if (!ha && !hb) return;
            const glm::vec3 target = ha && hb ? glm::mix(vb, va, fade) : ha ? glm::mix(value, va, fade) : glm::mix(value, vb, 1.0f - fade);
            value = glm::mix(value, target, k);
        };
        vec(p.translation, m_a[i].translation, m_b[i].translation, kChannelTranslation);
        vec(p.scale, m_a[i].scale, m_b[i].scale, kChannelScale);
        const bool ha = ca & kChannelRotation, hb = cb & kChannelRotation;
        if (ha || hb) {
            const glm::quat target = ha && hb ? nlerp(m_b[i].rotation, m_a[i].rotation, fade)
                                     : ha     ? nlerp(p.rotation, m_a[i].rotation, fade)
                                              : nlerp(p.rotation, m_b[i].rotation, 1.0f - fade);
            p.rotation = nlerp(p.rotation, target, k);
        }
    }
}

// P1 additive layer: the clip's motion relative to its first frame, scaled by weight × mask, on top of the pose below.
void Instance::applyAdditive(const Playing& p, std::span<const float> mask, float weight) {
    const size_t n = m_pose.size();
    const uint8_t* ch = channelsOf(p.clip, n);
    if (!ch || weight <= 0.0f) return;
    sampleClip(*p.clip.library, p.clip.clip, *p.clip.binding, clipTime(p.time, durationOf(p.clip), p.playback), m_a);
    sampleClip(*p.clip.library, p.clip.clip, *p.clip.binding, 0.0f, m_b); // the reference frame
    const glm::quat identity(1.0f, 0.0f, 0.0f, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        if (!ch[i]) continue;
        const float k = weight * maskWeight(mask, i);
        if (k <= 0.0f) continue;
        Transform& t = m_pose[i];
        if (ch[i] & kChannelTranslation) t.translation += (m_a[i].translation - m_b[i].translation) * k;
        if (ch[i] & kChannelRotation) {
            const glm::quat delta = glm::conjugate(m_b[i].rotation) * m_a[i].rotation; // reference → sample, in the node's frame
            t.rotation = glm::normalize(t.rotation * nlerp(identity, delta, k));
        }
        if (ch[i] & kChannelScale)
            for (int c = 0; c < 3; ++c) {
                const float ratio = std::fabs(m_b[i].scale[c]) > 1e-12f ? m_a[i].scale[c] / m_b[i].scale[c] : 1.0f;
                t.scale[c] *= 1.0f + (ratio - 1.0f) * k;
            }
    }
}

void Instance::evaluate(std::span<core::Affine3x4> outModel) {
    std::copy(m_rest.begin(), m_rest.end(), m_pose.begin());
    for (const Layer& l : m_layers) {
        if (!l.current.clip || l.weight <= 0.0f) continue;
        if (!l.additive) {
            applyLayer(l);
            continue;
        }
        applyAdditive(l.current, l.mask, l.weight * l.fade);
        if (l.previous.clip && l.fade < 1.0f) applyAdditive(l.previous, l.mask, l.weight * (1.0f - l.fade));
    }
    for (size_t i = 0; i < m_pose.size(); ++i)
        if (m_hasOverride[i]) m_pose[i] = m_override[i];
    evaluateHierarchy(m_parent, m_pose, outModel);
}

} // namespace anim
