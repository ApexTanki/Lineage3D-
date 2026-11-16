#include "renderer/particles/ParticleSystem.h"

#include "core/Log.h"
#include "core/Profile.h"
#include "platform/JobSystem.h"
#include "platform/Timer.h"

#include <algorithm>
#include <cmath>

namespace renderer {
namespace {

constexpr float kTwoPi = 6.28318530718f;

// xorshift32: each emitter instance owns one, so results never depend on which worker ran it.
inline float rand01(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return float(s >> 8) * (1.0f / 16777216.0f);
}

inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

uint32_t seedFor(uint32_t effectSeed, uint32_t emitter) {
    uint32_t h = effectSeed * 0x9E3779B9u + emitter * 0x85EBCA6Bu + 0x6A09E667u; // murmur3's finalizer mixes it
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h ? h : 0x2545F491u; // xorshift never leaves 0
}

// Spawn rate multiplier of the distance LOD (M7 plan D10): 1 within lod.x, ¼ at lod.y, nothing beyond.
float lodFactor(const glm::vec2& lod, float distance) {
    if (lod.y <= 0.0f || distance <= lod.x) return 1.0f;
    if (!(distance < lod.y)) return 0.0f; // also a non-finite distance
    return 1.0f - 0.75f * (distance - lod.x) / (lod.y - lod.x);
}

bool finiteTransform(const glm::vec3& p, const glm::quat& q) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && std::isfinite(q.x) && std::isfinite(q.y) &&
           std::isfinite(q.z) && std::isfinite(q.w) && glm::dot(q, q) > 1e-12f;
}

glm::vec3 sampleShape(const assets::ParticleEmitterDesc& d, uint32_t& rng) {
    switch (d.shape) {
    case assets::ParticleShape::Point: break;
    case assets::ParticleShape::Sphere: {
        const float z = 2.0f * rand01(rng) - 1.0f, phi = kTwoPi * rand01(rng);
        const float r = d.radius * std::cbrt(rand01(rng)), xy = std::sqrt(std::max(0.0f, 1.0f - z * z));
        return glm::vec3(xy * std::cos(phi), z, xy * std::sin(phi)) * r;
    }
    case assets::ParticleShape::Disc: {
        const float r = d.radius * std::sqrt(rand01(rng)), phi = kTwoPi * rand01(rng);
        return glm::vec3(r * std::cos(phi), 0.0f, r * std::sin(phi));
    }
    case assets::ParticleShape::Box: {
        const float x = 2.0f * rand01(rng) - 1.0f, y = 2.0f * rand01(rng) - 1.0f, z = 2.0f * rand01(rng) - 1.0f;
        return glm::vec3(x, y, z) * d.extents;
    }
    }
    return glm::vec3(0.0f);
}

// A direction within `cone` radians of `axis`, uniform over the spherical cap.
glm::vec3 sampleCone(const glm::vec3& axis, float cone, uint32_t& rng) {
    const float cosTheta = 1.0f - rand01(rng) * (1.0f - std::cos(cone));
    const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = kTwoPi * rand01(rng);
    const glm::vec3 helper = std::fabs(axis.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 b1 = glm::normalize(glm::cross(helper, axis)), b2 = glm::cross(axis, b1);
    return b1 * (sinTheta * std::cos(phi)) + b2 * (sinTheta * std::sin(phi)) + axis * cosTheta;
}

// One particle born `age` seconds before the end of the step, `motion` of the way along the effect's movement.
void spawnParticle(ParticleSystem::Emitter& e, const assets::ParticleEmitterDesc& d, ParticleArena& a, float age, float motion,
                   const glm::vec3& from, const glm::vec3& velocity) {
    if (e.count >= e.maxAlive) return; // its budget (M7 plan D10)
    const float life = lerp(d.life.x, d.life.y, rand01(e.rng));
    const glm::vec3 local = sampleShape(d, e.rng);
    const glm::vec3 direction = e.rotation * sampleCone(d.direction, d.cone, e.rng);
    const glm::vec3 v = direction * lerp(d.speed.x, d.speed.y, rand01(e.rng)) + velocity * d.inherit;
    // Straight along its velocity for the part of the step it already lived: gravity and drag over a fraction of a
    // frame are invisible, and the next step integrates it fully.
    const glm::vec3 p = glm::mix(from, e.toPosition, motion) + e.rotation * (d.offset + local) + v * age;
    const float sizeScale = lerp(d.sizeRandom.x, d.sizeRandom.y, rand01(e.rng));
    const float rotation = lerp(d.rotation.x, d.rotation.y, rand01(e.rng));
    const float spin = lerp(d.spin.x, d.spin.y, rand01(e.rng));
    const float frameOffset = d.randomStart ? rand01(e.rng) : 0.0f;
    ++e.spawned;
    if (age >= life) return; // it already died inside this step (the random sequence advanced all the same)
    const uint32_t i = e.first + e.count++;
    a.px[i] = p.x, a.py[i] = p.y, a.pz[i] = p.z;
    a.vx[i] = v.x, a.vy[i] = v.y, a.vz[i] = v.z;
    a.age[i] = age;
    a.invLife[i] = 1.0f / life;
    a.sizeScale[i] = sizeScale;
    a.rotation[i] = rotation;
    a.spin[i] = spin;
    a.frameOffset[i] = frameOffset;
}

// One simulation step of `h` seconds: integrate, bounce, age and kill, then spawn what falls inside the step.
void stepEmitter(ParticleSystem::Emitter& e, const assets::ParticleEmitterDesc& d, ParticleArena& a, float h, bool moving) {
    const uint32_t begin = e.first, end = e.first + e.count;
    float* __restrict px = a.px.data();
    float* __restrict py = a.py.data();
    float* __restrict pz = a.pz.data();
    float* __restrict vx = a.vx.data();
    float* __restrict vy = a.vy.data();
    float* __restrict vz = a.vz.data();
    float* __restrict age = a.age.data();
    float* __restrict rotation = a.rotation.data();
    const float* __restrict spin = a.spin.data();
    const float damping = std::exp(-d.drag * h), fall = d.gravity * h;
    for (uint32_t i = begin; i < end; ++i) {
        vy[i] -= fall;
        vx[i] *= damping;
        vy[i] *= damping;
        vz[i] *= damping;
        px[i] += vx[i] * h;
        py[i] += vy[i] * h;
        pz[i] += vz[i] * h;
        rotation[i] += spin[i] * h;
        age[i] += h;
    }
    if (d.collide) {
        const float keep = 1.0f - d.friction;
        for (uint32_t i = begin; i < end; ++i)
            if (py[i] < e.groundY) {
                py[i] = e.groundY;
                if (vy[i] < 0.0f) vy[i] = -vy[i] * d.restitution;
                vx[i] *= keep;
                vz[i] *= keep;
            }
    }
    // Kill: the last live particle takes the dead one's place (the block stays contiguous; draw order is re-sorted).
    for (uint32_t i = begin; i < e.first + e.count;) {
        if (age[i] * a.invLife[i] < 1.0f) {
            ++i;
            continue;
        }
        const uint32_t last = e.first + e.count - 1;
        px[i] = px[last], py[i] = py[last], pz[i] = pz[last];
        vx[i] = vx[last], vy[i] = vy[last], vz[i] = vz[last];
        age[i] = age[last];
        a.invLife[i] = a.invLife[last];
        a.sizeScale[i] = a.sizeScale[last];
        rotation[i] = rotation[last];
        a.spin[i] = a.spin[last];
        a.frameOffset[i] = a.frameOffset[last];
        --e.count;
    }

    // Spawning, in the emitter's local time (after its delay).
    const float t0 = e.time - d.delay, t1 = t0 + h;
    e.time += h;
    if (e.finished) return;
    if (e.stopped) {
        e.finished = true;
        return;
    }
    const glm::vec3 from = moving ? e.fromPosition : e.toPosition;
    const glm::vec3 velocity = moving ? e.velocity : glm::vec3(0.0f);
    // Rate: evenly spread over the part of the step inside its window ([0, duration), or everything after the delay).
    if (d.rate > 0.0f) {
        const float w0 = std::max(t0, 0.0f), w1 = d.loop ? t1 : std::min(t1, d.duration);
        if (w1 > w0) {
            e.spawnCarry += d.rate * e.lodScale * (w1 - w0);
            if (!(e.spawnCarry >= 0.0f && e.spawnCarry < 1e6f)) e.spawnCarry = 0.0f; // NaN must never reach the conversion
            const uint32_t n = uint32_t(e.spawnCarry);
            e.spawnCarry -= float(n);
            for (uint32_t k = 0; k < n; ++k) {
                const float t = w0 + (w1 - w0) * (float(k) + 0.5f) / float(n);
                spawnParticle(e, d, a, t1 - t, (t - t0) / h, from, velocity);
            }
        }
    }
    // Bursts, at their exact times; a looping emitter repeats them every period.
    for (;;) {
        if (e.nextBurst == d.burstCount) {
            if (!d.loop || d.burstCount == 0 || t1 <= float(e.period + 1) * d.duration) break;
            ++e.period;
            e.nextBurst = 0;
            continue;
        }
        const float tb = (d.loop ? float(e.period) * d.duration : 0.0f) + d.bursts[e.nextBurst].time;
        if (tb >= t1) break;
        const float scaled = float(d.bursts[e.nextBurst].count) * e.lodScale;
        const uint32_t n = e.lodScale > 0.0f ? std::max(1u, uint32_t(scaled + 0.5f)) : 0u;
        for (uint32_t k = 0; k < n; ++k) spawnParticle(e, d, a, t1 - std::max(tb, t0), (std::max(tb, t0) - t0) / h, from, velocity);
        ++e.nextBurst;
    }
    if (!d.loop && e.nextBurst == d.burstCount && t1 >= d.duration) e.finished = true;
}

// The world box of every live quad: centers, grown by the largest half diagonal a quad can have.
void computeBounds(ParticleSystem::Emitter& e, const assets::ParticleEmitterDesc& d, const ParticleArena& a) {
    if (e.count == 0) {
        e.boundsMin = e.boundsMax = e.origin;
        return;
    }
    glm::vec3 lo(a.px[e.first], a.py[e.first], a.pz[e.first]), hi = lo;
    float maxSpeed2 = 0.0f;
    for (uint32_t i = e.first; i < e.first + e.count; ++i) {
        lo = glm::min(lo, glm::vec3(a.px[i], a.py[i], a.pz[i]));
        hi = glm::max(hi, glm::vec3(a.px[i], a.py[i], a.pz[i]));
        maxSpeed2 = std::max(maxSpeed2, a.vx[i] * a.vx[i] + a.vy[i] * a.vy[i] + a.vz[i] * a.vz[i]);
    }
    float maxSize = 0.0f;
    for (float s : d.size.samples) maxSize = std::max(maxSize, s);
    const float stretch = d.billboard == assets::ParticleBillboard::Velocity ? std::sqrt(maxSpeed2) * d.stretch : 0.0f;
    const float half = 0.5f * (maxSize * d.sizeRandom.y + stretch) * 1.41422f; // a rotated square's half diagonal
    e.boundsMin = lo - glm::vec3(half);
    e.boundsMax = hi + glm::vec3(half);
}

} // namespace

void ParticleSystem::init(ResourceManager& resources, Scene& scene) {
    m_resources = &resources;
    m_scene = &scene;
    for (std::vector<float>* v : {&m_arena.px, &m_arena.py, &m_arena.pz, &m_arena.vx, &m_arena.vy, &m_arena.vz, &m_arena.age,
                                  &m_arena.invLife, &m_arena.sizeScale, &m_arena.rotation, &m_arena.spin, &m_arena.frameOffset})
        v->assign(kArenaCapacity, 0.0f);
    m_emitters.assign(kMaxEmitters, Emitter{});
    m_freeEmitters.clear();
    m_freeEmitters.reserve(kMaxEmitters);
    for (uint32_t i = kMaxEmitters; i-- > 0;) m_freeEmitters.push_back(uint16_t(i));
    m_freeRanges.clear();
    m_freeRanges.reserve(kMaxEmitters + 2); // free ranges ≤ allocated blocks + 1: inserts never grow it
    m_freeRanges.push_back({0, kArenaCapacity});
    m_simulate.assign(kMaxEmitters, 0);
    m_active.assign(kMaxEmitters, 0);
    m_finished.assign(kMaxEffects, EffectId{});
    m_stats = Stats{};
}

void ParticleSystem::shutdown() {
    if (!m_resources) return;
    clear();
    m_resources = nullptr;
    m_scene = nullptr;
}

EffectId ParticleSystem::spawn(const EffectDesc& desc) {
    if (!finiteTransform(desc.position, desc.rotation)) { // a game-side NaN must not reach the simulation
        if (m_stats.dropped++ % 100 == 0) FL_LOG_WARN("Particle effect with a non-finite position or rotation: dropped");
        return {};
    }
    Effect fx;
    fx.preset = desc.preset;
    fx.position = fx.lastPosition = desc.position;
    fx.rotation = glm::normalize(desc.rotation);
    fx.groundY = std::isfinite(desc.groundHeight) ? desc.groundHeight : desc.position.y;
    fx.seed = desc.seed ? desc.seed : m_nextSeed++;
    fx.prewarm = std::isfinite(desc.prewarm) ? std::clamp(desc.prewarm, 0.0f, 60.0f) : 0.0f;
    const EffectId id = m_effects.create(fx);
    if (!id) {
        if (m_stats.dropped++ % 100 == 0) FL_LOG_WARN("Particle effect pool full (%u effects): effect dropped", kMaxEffects);
        return {};
    }
    m_resources->addRef(desc.preset);
    return id;
}

void ParticleSystem::setTransform(EffectId id, const glm::vec3& position, const glm::quat& rotation) {
    Effect* fx = m_effects.get(id);
    if (!fx || !finiteTransform(position, rotation)) return; // a non-finite transform keeps the last good one
    fx->position = position;
    fx->rotation = glm::normalize(rotation);
    if (!fx->started) fx->lastPosition = position; // waiting for its preset: no motion to inherit yet
}

void ParticleSystem::stop(EffectId id) {
    if (Effect* fx = m_effects.get(id)) fx->stopped = true;
}

void ParticleSystem::kill(EffectId id) {
    Effect* fx = m_effects.get(id);
    if (!fx) return;
    release(*fx);
    m_effects.destroy(id);
}

void ParticleSystem::clear() {
    uint32_t count = 0;
    m_effects.forEach([&](EffectId id, Effect&) { m_finished[count++] = id; });
    for (uint32_t i = 0; i < count; ++i) kill(m_finished[i]);
    m_activeCount = 0;
}

bool ParticleSystem::allocateBlock(uint32_t count, uint32_t& first) {
    for (size_t i = 0; i < m_freeRanges.size(); ++i) {
        Range& r = m_freeRanges[i];
        if (r.count < count) continue;
        first = r.first;
        r.first += count;
        r.count -= count;
        if (r.count == 0) m_freeRanges.erase(m_freeRanges.begin() + ptrdiff_t(i));
        return true;
    }
    return false;
}

void ParticleSystem::freeBlock(uint32_t first, uint32_t count) {
    size_t p = 0;
    while (p < m_freeRanges.size() && m_freeRanges[p].first < first) ++p;
    const bool joinsPrevious = p > 0 && m_freeRanges[p - 1].first + m_freeRanges[p - 1].count == first;
    const bool joinsNext = p < m_freeRanges.size() && first + count == m_freeRanges[p].first;
    if (joinsPrevious && joinsNext) {
        m_freeRanges[p - 1].count += count + m_freeRanges[p].count;
        m_freeRanges.erase(m_freeRanges.begin() + ptrdiff_t(p));
    } else if (joinsPrevious) {
        m_freeRanges[p - 1].count += count;
    } else if (joinsNext) {
        m_freeRanges[p].first = first;
        m_freeRanges[p].count += count;
    } else {
        m_freeRanges.insert(m_freeRanges.begin() + ptrdiff_t(p), Range{first, count});
    }
}

bool ParticleSystem::start(Effect& fx, uint32_t slot) {
    const PresetAsset* preset = m_resources->preset(fx.preset);
    if (!preset || preset->state != LoadState::Ready) return false;
    const std::vector<assets::ParticleEmitterDesc>& descs = preset->file.emitters;
    const uint32_t prewarmSteps = uint32_t(std::lround(fx.prewarm / kPrewarmStep));
    for (uint32_t i = 0; i < descs.size(); ++i) {
        uint32_t first = 0;
        if (m_freeEmitters.empty() || !allocateBlock(descs[i].max, first)) {
            if (m_stats.dropped % 100 == 0)
                FL_LOG_WARN("Particle arena full (%u particles, %u emitters): effect %s not started", kArenaCapacity, kMaxEmitters,
                            preset->path.c_str());
            release(fx);
            return false;
        }
        const uint16_t e = m_freeEmitters.back();
        m_freeEmitters.pop_back();
        Emitter& emitter = m_emitters[e];
        emitter = Emitter{};
        emitter.desc = &descs[i];
        emitter.texture = preset->textures[i];
        emitter.effect = slot;
        emitter.first = first;
        emitter.capacity = descs[i].max;
        emitter.rng = seedFor(fx.seed, i);
        emitter.prewarmSteps = prewarmSteps;
        emitter.origin = emitter.fromPosition = emitter.toPosition = fx.position;
        emitter.boundsMin = emitter.boundsMax = fx.position;
        emitter.rotation = fx.rotation;
        emitter.groundY = fx.groundY;
        fx.emitters[fx.emitterCount++] = e;
    }
    fx.started = true;
    fx.time = float(prewarmSteps) * kPrewarmStep;
    fx.lastPosition = fx.position;
    return true;
}

void ParticleSystem::release(Effect& fx) {
    for (uint32_t k = 0; k < fx.emitterCount; ++k) {
        Emitter& e = m_emitters[fx.emitters[k]];
        freeBlock(e.first, e.capacity);
        e = Emitter{};
        m_freeEmitters.push_back(fx.emitters[k]);
    }
    fx.emitterCount = 0;
    for (LightId& light : fx.lights) {
        if (light && m_scene->light(light)) m_scene->removeLight(light); // the scene may have been cleared meanwhile
        light = {};
    }
    if (fx.preset) m_resources->release(fx.preset);
    fx.preset = {};
}

// Effect lights (M7 plan D11): a point light per emitter that has one, from its start (after its delay) for its life,
// its intensity fading as (1 − age / life)². Driven by the effect's time, so prewarm and frozen time hold them too.
// Disabled particles keep the clock but put the lights out.
void ParticleSystem::updateLights(Effect& fx, bool enabled) {
    for (uint32_t k = 0; k < fx.emitterCount; ++k) {
        const assets::ParticleEmitterDesc& d = *m_emitters[fx.emitters[k]].desc;
        if (!d.hasLight) continue;
        const float age = fx.time - d.delay;
        LightId& id = fx.lights[k];
        if (id && !m_scene->light(id)) id = {}; // the scene was cleared
        if (enabled && age >= 0.0f && age < d.light.life) {
            const float fade = 1.0f - age / d.light.life;
            LightDesc light;
            light.type = LightType::Point;
            light.position = fx.position;
            light.color = d.light.color;
            light.intensity = d.light.intensity * fade * fade;
            light.range = d.light.range;
            if (id) {
                m_scene->updateLight(id, light);
            } else if (!(fx.lightsRefused & (1u << k))) {
                id = m_scene->addLight(light); // a full pool logs once and leaves this light dark for the rest of its life
                if (!id) fx.lightsRefused = uint8_t(fx.lightsRefused | (1u << k));
            }
        } else if (id) {
            m_scene->removeLight(id);
            id = {};
        }
        if (id) ++m_stats.lights;
    }
}

void ParticleSystem::update(platform::JobSystem& jobs, float dt, const glm::vec3& eye, const Settings& settings) {
    FL_ZONE("ParticleSystem::update");
    const uint64_t startTicks = platform::nowTicks();
    dt = std::isfinite(dt) ? std::clamp(dt, 0.0f, kMaxStep) : 0.0f;
    const float budget = std::isfinite(settings.budgetScale) ? std::clamp(settings.budgetScale, 0.0f, 1.0f) : 1.0f;
    // Coverage cap (M7 plan D10): above it, spawning slows in proportion; the scale eases toward its target so it
    // settles instead of oscillating (particles live for seconds, so coverage answers slowly).
    float target = 1.0f;
    if (settings.coverageCap > 0.0f && m_stats.coverage > 0.0f)
        target = std::clamp(m_coverageScale * settings.coverageCap / m_stats.coverage, 0.05f, 1.0f);
    if (dt > 0.0f) m_coverageScale += (target - m_coverageScale) * 0.1f;

    m_stats.spawned = m_stats.lights = 0;
    m_simulateCount = 0;
    uint32_t finished = 0;
    m_effects.forEach([&](EffectId id, Effect& fx) {
        if (!fx.started) {
            const LoadState state = m_resources->state(fx.preset);
            if (state == LoadState::Loading) return; // it waits (its clock hasn't started)
            if (state == LoadState::Failed || !start(fx, id.index())) {
                ++m_stats.dropped;
                m_finished[finished++] = id;
                return;
            }
        }
        fx.time += dt;
        updateLights(fx, settings.enabled);
        bool done = true;
        for (uint32_t k = 0; k < fx.emitterCount; ++k) {
            const Emitter& e = m_emitters[fx.emitters[k]];
            done = done && e.finished && e.count == 0;
        }
        for (uint32_t k = 0; k < fx.emitterCount && done; ++k) {
            const assets::ParticleEmitterDesc& d = *m_emitters[fx.emitters[k]].desc;
            done = !d.hasLight || fx.time - d.delay >= d.light.life;
        }
        if (done) {
            m_finished[finished++] = id;
            return;
        }
        const glm::vec3 velocity = dt > 0.0f ? (fx.position - fx.lastPosition) / dt : glm::vec3(0.0f);
        const float distance = glm::length(fx.position - eye);
        for (uint32_t k = 0; k < fx.emitterCount; ++k) {
            Emitter& e = m_emitters[fx.emitters[k]];
            e.fromPosition = fx.lastPosition;
            e.toPosition = e.origin = fx.position;
            e.rotation = fx.rotation;
            e.velocity = velocity;
            e.groundY = fx.groundY;
            e.stopped = fx.stopped;
            e.lodScale = lodFactor(e.desc->lod, distance) * m_coverageScale;
            e.maxAlive = settings.enabled ? std::min(e.capacity, uint32_t(std::lround(float(e.capacity) * budget))) : 0u;
            m_simulate[m_simulateCount++] = fx.emitters[k];
        }
        fx.lastPosition = fx.position;
    });
    for (uint32_t i = 0; i < finished; ++i) kill(m_finished[i]);

    m_stepDt = dt;
    jobs.parallelFor(m_simulateCount, 4, simulateRange, this);

    m_activeCount = 0;
    m_stats.particles = 0;
    for (uint32_t i = 0; i < m_simulateCount; ++i) {
        const Emitter& e = m_emitters[m_simulate[i]];
        m_stats.particles += e.count;
        m_stats.spawned += e.spawned;
        if (e.count) m_active[m_activeCount++] = m_simulate[i];
    }
    m_stats.effects = m_effects.liveCount();
    m_stats.emitters = m_simulateCount;
    m_stats.activeEmitters = m_activeCount;
    m_stats.coverageScale = m_coverageScale;
    m_stats.simulationMs = platform::ticksToMilliseconds(platform::nowTicks() - startTicks);
}

void ParticleSystem::reportCoverage(float screens) {
    m_stats.coverage = std::isfinite(screens) ? std::max(0.0f, screens) : 0.0f;
}

void ParticleSystem::simulateRange(void* context, uint32_t begin, uint32_t end, uint32_t) {
    ParticleSystem& self = *static_cast<ParticleSystem*>(context);
    for (uint32_t i = begin; i < end; ++i) {
        Emitter& e = self.m_emitters[self.m_simulate[i]];
        const assets::ParticleEmitterDesc& d = *e.desc;
        e.spawned = 0;
        for (uint32_t k = 0; k < e.prewarmSteps; ++k) stepEmitter(e, d, self.m_arena, kPrewarmStep, false);
        e.prewarmSteps = 0;
        if (self.m_stepDt > 0.0f) stepEmitter(e, d, self.m_arena, self.m_stepDt, true);
        computeBounds(e, d, self.m_arena);
    }
}

} // namespace renderer
