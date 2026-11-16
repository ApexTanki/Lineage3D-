#pragma once

#include "rhi/Device.h"

#include <span>

namespace renderer {

// Per-pass CPU and GPU timings for the debug overlay (ARCHITECTURE.md §4.13). The GPU side uses timestamp
// pairs; results arrive two frames late, when the frame slot comes around again. Backends without timer
// queries report NaN, which the overlay shows as "n/a".
class GpuTimers {
public:
    static constexpr uint32_t kMaxPasses = 32; // M8: the post chain adds about ten passes

    struct Timing {
        const char* name = "";
        double cpuMs = 0.0;
        double gpuMs = 0.0;
    };

    bool init(rhi::Device& device);
    void shutdown();

    void beginFrame(); // after a successful Device::beginFrame()
    uint32_t beginPass(rhi::CommandList& cmd, const char* name); // name must be a string literal
    void endPass(rhi::CommandList& cmd, uint32_t pass);
    void endFrame();   // after Device::present()

    std::span<const Timing> results() const { return {m_results, m_resultCount}; }

private:
    rhi::Device* m_device = nullptr;
    rhi::QueryPoolHandle m_pool;
    uint32_t m_slot = 0;
    const char* m_names[rhi::kFramesInFlight][kMaxPasses] = {};
    double m_cpuMs[rhi::kFramesInFlight][kMaxPasses] = {};
    uint32_t m_passCount[rhi::kFramesInFlight] = {};
    uint64_t m_cpuStart[kMaxPasses] = {};
    double m_gpu[kMaxPasses * 2] = {};
    Timing m_results[kMaxPasses];
    uint32_t m_resultCount = 0;
};

} // namespace renderer
