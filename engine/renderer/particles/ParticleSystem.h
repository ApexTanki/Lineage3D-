#pragma once

#include "assets/ParticleFormat.h"
#include "core/Handle.h"
#include "renderer/core/Scene.h"
#include "renderer/resources/ResourceManager.h"

#include <limits>
#include <span>
#include <vector>

namespace platform {
class JobSystem;
}

// CPU particles (ARCHITECTURE.md §5.6, M7 plan D1, D3, D4, D10, D11).

namespace renderer {

// M9 plan D8: one-shot effects churn through the 512 slots: 10 index bits leave 22 for generations (about 2.1 billion spawns).
using EffectId = core::Handle<struct EffectTag, 10>;

struct EffectDesc {
    PresetHandle preset;                          // the effect waits while it loads; a failed preset drops the effect
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};   // effect space → world: emission shapes and directions
    float groundHeight = std::numeric_limits<float>::quiet_NaN(); // collision plane (world y); NaN = the spawn height
    uint32_t seed = 0;                            // its random sequence; 0 = the next of the system's counter
    float prewarm = 0.0f;                         // seconds simulated when it starts, in fixed 1/60 s steps
};

// Structure of arrays (M7 plan D3): one contiguous block per emitter instance, [first, first + count) alive.
struct ParticleArena {
    std::vector<float> px, py, pz;  // world position
    std::vector<float> vx, vy, vz;  // m/s
    std::vector<float> age;         // seconds
    std::vector<float> invLife;     // 1 / life: age · invLife is the normalized age the curves read
    std::vector<float> sizeScale;   // per-particle size multiplier
    std::vector<float> rotation;    // radians
    std::vector<float> spin;        // radians per second
    std::vector<float> frameOffset; // random flipbook start, 0–1 of the frame count
};

// Effects are lists of emitter instances simulated on the job workers, one job per emitter: spawning (rate and bursts,
// at their exact times inside the step, along the effect's motion), gravity, drag, spin, ground bounces, ageing and a
// world box for culling. Everything is allocated at init: an arena of kArenaCapacity particles carved into one block
// per emitter instance (its `max`), first fit from a sorted free list, and pools of effects and emitter instances. An
// effect that finds no room doesn't start (counted and logged). Each emitter draws from its own random sequence, so
// the results never depend on the job order, and prewarm runs the very steps of a 60 Hz frame loop.
class ParticleSystem {
public:
    static constexpr uint32_t kArenaCapacity = 65536;
    static constexpr uint32_t kMaxEffects = 512;
    static constexpr uint32_t kMaxEmitters = 2048;
    static constexpr float kPrewarmStep = 1.0f / 60.0f;
    static constexpr float kMaxStep = 0.1f; // a longer frame (a hitch) slows effects down instead of jumping

    struct Settings {
        bool enabled = true;      // off: effects keep their clocks (one-shots finish and free themselves) but spawn
                                  // nothing, and their lights go dark (the renderer draws nothing)
        float speed = 1.0f;       // simulation time scale (0 holds every particle)
        float budgetScale = 1.0f; // scales every emitter's max (tiers; 0–1)
        float coverageCap = 3.0f; // screens of particle area before every spawn rate is scaled down (M7 plan D10)
    };

    // One emitter instance, as the renderer reads it after update().
    struct Emitter {
        const assets::ParticleEmitterDesc* desc = nullptr; // into the preset, which the effect keeps loaded
        TextureAsset texture;     // its atlas (the preset holds the reference); invalid = none
        uint32_t effect = 0;      // pool slot of its effect
        uint32_t first = 0;       // its block in the arena
        uint32_t capacity = 0;
        uint32_t count = 0;       // live particles, [first, first + count)
        glm::vec3 boundsMin{0.0f}, boundsMax{0.0f}; // world box of every live quad (count > 0)
        glm::vec3 origin{0.0f};   // the effect's position: distance LOD and the lit emitters' probe sample
        float lodScale = 1.0f;    // this frame's spawn multiplier: distance LOD × coverage scale
        // Simulation state.
        float time = 0.0f;        // effect time at the end of its last step
        float spawnCarry = 0.0f;  // fraction of a particle the rate owes
        uint32_t nextBurst = 0;   // within the current period
        uint32_t period = 0;      // looping emitters: periods completed
        uint32_t rng = 1;
        uint32_t prewarmSteps = 0;
        uint32_t maxAlive = 0;    // this frame's budget: max × budget scale
        uint32_t spawned = 0;     // this frame
        bool stopped = false;     // its effect was stopped
        bool finished = false;    // will never spawn again
        // The effect's motion over this frame.
        glm::vec3 fromPosition{0.0f}, toPosition{0.0f};
        glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec3 velocity{0.0f};
        float groundY = 0.0f;
    };

    struct Stats {
        uint32_t effects = 0;        // live effects (started or waiting for their preset)
        uint32_t emitters = 0;       // emitter instances of started effects
        uint32_t activeEmitters = 0; // with live particles
        uint32_t particles = 0;
        uint32_t spawned = 0;        // this frame
        uint32_t lights = 0;         // effect lights alive
        uint32_t dropped = 0;        // effects that found no room (arena or pools) or whose preset failed, since start
        float coverageScale = 1.0f;  // spawn rate multiplier from the coverage cap
        float coverage = 0.0f;       // last reported screens of particle area
        double simulationMs = 0.0;   // update(): bookkeeping and the simulation jobs
    };

    ParticleSystem() = default;
    ~ParticleSystem() { shutdown(); }
    ParticleSystem(const ParticleSystem&) = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;

    void init(ResourceManager& resources, Scene& scene);
    void shutdown();

    EffectId spawn(const EffectDesc& desc); // takes a reference on the preset
    void setTransform(EffectId id, const glm::vec3& position, const glm::quat& rotation);
    void stop(EffectId id);  // stops spawning; the effect frees itself once its particles and lights are gone
    void kill(EffectId id);  // removes it at once
    bool alive(EffectId id) const { return m_effects.get(id) != nullptr; }
    void clear();

    // Main thread, once per frame, before the scene's lights are gathered: starts effects whose preset arrived, moves
    // their lights, frees finished ones, and simulates every emitter on the job workers. `dt` is the frame delta
    // already scaled by the particle speed.
    void update(platform::JobSystem& jobs, float dt, const glm::vec3& eye, const Settings& settings);
    // The frame's measured screen coverage (screens of particle area), from the renderer: it scales spawning from the
    // next frame on.
    void reportCoverage(float screens);

    const ParticleArena& arena() const { return m_arena; }
    std::span<const uint32_t> activeEmitters() const { return {m_active.data(), m_activeCount}; } // after update()
    const Emitter& emitter(uint32_t slot) const { return m_emitters[slot]; }
    const Stats& stats() const { return m_stats; }

private:
    struct Effect {
        PresetHandle preset;
        glm::vec3 position{0.0f}, lastPosition{0.0f};
        glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
        float groundY = 0.0f;
        uint32_t seed = 0;
        float prewarm = 0.0f;
        float time = 0.0f;
        bool started = false;
        bool stopped = false;
        uint32_t emitterCount = 0;
        uint16_t emitters[assets::kMaxEffectEmitters] = {};
        LightId lights[assets::kMaxEffectEmitters];
        uint8_t lightsRefused = 0; // bit k: the scene's light pool was full when emitter k's light started (not retried)
    };
    struct Range {
        uint32_t first, count;
    };

    bool start(Effect& effect, uint32_t slot);
    void release(Effect& effect);
    bool allocateBlock(uint32_t count, uint32_t& first);
    void freeBlock(uint32_t first, uint32_t count);
    void updateLights(Effect& effect, bool enabled);
    static void simulateRange(void* context, uint32_t begin, uint32_t end, uint32_t worker);

    ResourceManager* m_resources = nullptr;
    Scene* m_scene = nullptr;
    ParticleArena m_arena;
    core::HandlePool<Effect, EffectTag, EffectId::kMaxGeneration, 10> m_effects{kMaxEffects};
    std::vector<Emitter> m_emitters;          // kMaxEmitters slots
    std::vector<uint16_t> m_freeEmitters;     // a stack of free slots
    std::vector<Range> m_freeRanges;          // sorted by first, never adjacent; capacity kMaxEmitters + 1
    std::vector<uint32_t> m_simulate;         // this frame's emitter slots to simulate
    uint32_t m_simulateCount = 0;
    std::vector<uint32_t> m_active;           // ... and those with live particles after it
    uint32_t m_activeCount = 0;
    std::vector<EffectId> m_finished;         // effects to free after this frame's walk
    uint32_t m_nextSeed = 1;
    float m_coverageScale = 1.0f;
    float m_stepDt = 0.0f;                    // the step simulateRange applies after any prewarm steps
    Stats m_stats;
};

} // namespace renderer
