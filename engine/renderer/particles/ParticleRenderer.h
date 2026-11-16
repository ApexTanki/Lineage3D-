#pragma once

#include "core/Result.h"
#include "renderer/particles/ParticleSystem.h"
#include "rhi/Device.h"

#include <cstddef>
#include <span>
#include <vector>

namespace platform {
class JobSystem;
}

namespace renderer {

class ShaderLibrary;
class Visibility;
class ProbeGrid;
class RenderQueue;

// Per-draw constants of a particle draw (M7 plan D5, D9): the 128-byte PerDraw block of fl_bindings.glsl, read with its
// own meaning by particle.vert/frag.
struct ParticleDrawConstants {
    glm::vec4 params{0.0f};                        // pd.modelRows[0]: billboard, blend, 1 / soft distance (0 = hard), intensity
    glm::vec4 flipbook{1.0f, 1.0f, 1.0f, 0.0f};    // pd.modelRows[1]: columns, rows, frame count, 1 = interpolate
    glm::vec4 normalLit{0.0f, 1.0f, 0.0f, 0.0f};   // pd.modelRows[2]: world normal (world billboards), 1 = lit
    glm::vec4 sun{1.0f, 0.0f, 0.0f, 0.0f};         // pd.lightmapST: x = sun visibility at the emitter (probes)
    glm::uvec4 unused{0u};                         // pd.lights
    glm::vec4 sh[3]{};                             // pd.sh: L1 SH irradiance at the emitter (lit emitters)
};
static_assert(sizeof(ParticleDrawConstants) == 128, "ParticleDrawConstants is the PerDraw block");

// One particle, 32 bytes: instance attributes 8–11 of particle.vert (ARCHITECTURE.md §7.8).
struct ParticleInstance {
    float position[3];        // RGB32F: world center
    uint16_t sizeRotation[4]; // RGBA16F: width, height (m), rotation (rad), frame blend
    uint8_t color[4];         // RGBA8: linear color, opacity
    uint16_t axisFrame[4];    // RGBA16F: velocity (velocity billboards), frame
};
static_assert(sizeof(ParticleInstance) == 32, "ParticleInstance must match particle.vert");

// IEEE half, round to nearest even; beyond the half range clamps to ±65504, NaN becomes 0.
uint16_t packHalf(float value);

// The flipbook frame a particle shows (M7 plan D7): the cell and the blend into the next one.
struct ParticleFrame {
    float frame = 0.0f;
    float blend = 0.0f;
};
ParticleFrame particleFrame(const assets::ParticleEmitterDesc& d, float age, float normalizedAge, float frameOffset);

// Draws the particle system's emitters (M7 plan D5–D10). prepare() culls the active emitters by their boxes, then one job
// per visible emitter sorts its particles back to front (alpha and premultiplied blending; additive needs no order),
// writes their instances into the vertex ring and samples the probes for lit emitters; each visible emitter becomes one
// transparent queue item keyed by its center depth, so emitters interleave with transparent meshes. The `particle`
// pipelines exist per blend mode, with and without the soft fade, plus one for the overdraw view. Nothing allocates per
// frame: the result and sort arrays are sized to the system's capacities, and an atlas's set is made once.
class ParticleRenderer {
public:
    // The command bytes of the most draws the particles can record: part of the renderer's command-list bound (R48),
    // by capacity, so effects coming and going never grow a list.
    static constexpr size_t kMaxCommandBytes = size_t(ParticleSystem::kMaxEmitters) * 320 + 4096;

    struct FrameInput {
        const ParticleSystem* system = nullptr;
        const Visibility* visibility = nullptr;
        const ProbeGrid* probes = nullptr;     // an empty grid lights with `ambientSh`
        const glm::vec4* ambientSh = nullptr;  // [3]: the hemispherical ambient as SH
        glm::vec3 eye{0.0f};
        float invFar = 0.001f;
        float tanHalfFovY = 0.577f;
        float aspect = 1.0f;
        bool soft = true; // soft particles are on this frame (and the device can sample depth)
        bool haze = false; // heat haze is on and its inputs exist: distort emitters draw in pass 8 (M8 plan D11)
    };
    struct Stats {
        uint32_t draws = 0;     // visible emitters
        uint32_t instances = 0; // particles drawn
        uint32_t softDraws = 0; // drawn with the soft fade
        uint32_t dropped = 0;   // visible emitters left out because the vertex ring was full
        float coverage = 0.0f;  // screens of particle area (the coverage cap's input)
        uint32_t hazeDraws = 0;   // distort emitters drawn in pass 8 (M8 plan D11)
        uint32_t hazeSkipped = 0; // distort emitters left out because heat haze is off
        double prepareMs = 0.0;
    };

    core::Result<void> init(rhi::Device& device, ShaderLibrary& shaders, ResourceManager& resources,
                            rhi::ResourceSetLayoutHandle frameLayout, rhi::ResourceSetLayoutHandle softLayout);
    void shutdown();
    // Recreates the pipelines for a scene attachment layout (format or MSAA change; never per frame).
    bool setTarget(const rhi::AttachmentLayout& target);

    // After visibility (and the system's update). Returns the emitters to draw.
    uint32_t prepare(platform::JobSystem& jobs, const FrameInput& in);
    void clearFrame() { // particles are off this frame: nothing to draw, haze included
        m_drawCount = 0;
        m_hazeCount = 0;
        m_stats = Stats{};
    }
    // Their queue items (transparent pass, back to front by center depth), before the queue sorts.
    uint32_t queueDraws(RenderQueue& queue) const; // returns the items queued (0 when the queue is full)
    bool needsSoftDepth() const { return m_stats.softDraws > 0; }
    // One emitter's draw, inside the scene pass. `softPass`: the frame was split and `softSet` holds the scene's linear
    // depth; otherwise soft emitters draw with hard edges.
    void record(rhi::CommandList& cmd, uint32_t draw, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets,
                rhi::ResourceSetHandle softSet, bool softPass) const;
    // The overdraw view: every emitter adds one step per covered pixel of each quad.
    void recordOverdraw(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const;
    // Heat haze (pass 8, M8 plan D11): set 1 of the distort pipeline (binding 1 the linear depth, binding 2 the grab),
    // rebuilt when either texture changes; then the visible distort emitters, back to front, into the resolved scene.
    bool setHazeInputs(rhi::TextureHandle linearDepth, rhi::TextureHandle grab);
    uint32_t hazeCount() const { return m_hazeCount; }
    void recordHaze(rhi::CommandList& cmd, rhi::ResourceSetHandle frameSet, std::span<const uint32_t> frameOffsets) const;
    const Stats& stats() const { return m_stats; }

private:
    struct Draw {
        uint32_t emitter = 0;        // system slot
        rhi::BufferSlice instances;  // in the vertex ring
        uint32_t count = 0;
        float depth = 0.0f;          // center depth / far, the queue key
        uint32_t blend = 0;
        bool soft = false;
        float coverage = 0.0f;
        rhi::ResourceSetHandle atlas;
        ParticleDrawConstants constants;
    };
    struct Atlas {
        TextureAsset asset;
        rhi::TextureHandle texture; // the one its set was made for
        rhi::ResourceSetHandle set;
    };
    struct JobContext {
        ParticleRenderer* self;
        const FrameInput* in;
        std::span<const uint32_t> active;
    };

    static void prepareRange(void* context, uint32_t begin, uint32_t end, uint32_t worker);
    rhi::ResourceSetHandle atlasSet(TextureAsset asset);

    rhi::Device* m_device = nullptr;
    ResourceManager* m_resources = nullptr;
    rhi::ResourceSetLayoutHandle m_frameLayout, m_softLayout, m_atlasLayout;
    rhi::SamplerHandle m_sampler;
    rhi::ShaderHandle m_vertex[2], m_fragment[2]; // [soft]
    rhi::PipelineHandle m_pipelines[3][2];        // [blend][soft]
    rhi::PipelineHandle m_overdraw;
    rhi::ShaderHandle m_hazeVertex, m_hazeFragment;
    rhi::ResourceSetLayoutHandle m_hazeLayout;
    rhi::PipelineHandle m_hazePipeline;           // single-sample scene format, no depth (pass 8)
    rhi::ResourceSetHandle m_hazeSet;
    rhi::TextureHandle m_hazeInputs[2];           // the linear depth and grab m_hazeSet was made with
    std::vector<Draw> m_hazeDraws;                // this frame's visible distort emitters, back to front
    uint32_t m_hazeCount = 0;
    rhi::ResourceSetHandle m_whiteSet;            // emitters without an atlas
    std::vector<Atlas> m_atlases;
    std::vector<Draw> m_results;                  // per active emitter (count 0 = not drawn)
    std::vector<Draw> m_draws;                    // this frame's visible emitters, in active order
    std::vector<uint64_t> m_sortKeys;             // per arena slot: each emitter sorts inside its own block
    uint32_t m_drawCount = 0;
    Stats m_stats;
    bool m_warnedDrops = false;
};

} // namespace renderer
