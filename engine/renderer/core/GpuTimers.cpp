#include "renderer/core/GpuTimers.h"

#include "platform/Timer.h"

#include <cmath>

namespace renderer {

bool GpuTimers::init(rhi::Device& device) {
    m_device = &device;
    m_pool = device.createTimestampPool(kMaxPasses * 2);
    return bool(m_pool);
}

void GpuTimers::shutdown() {
    if (m_device && m_pool) m_device->destroy(m_pool);
    m_pool = {};
}

void GpuTimers::beginFrame() {
    // The slot we are about to record into last ran two frames ago; its results are ready now.
    m_device->readTimestamps(m_pool, m_gpu);
    const uint32_t count = m_passCount[m_slot];
    m_resultCount = count;
    for (uint32_t i = 0; i < count; ++i) {
        const double start = m_gpu[i * 2], end = m_gpu[i * 2 + 1];
        m_results[i] = {m_names[m_slot][i], m_cpuMs[m_slot][i], std::isnan(start) || std::isnan(end) ? NAN : end - start};
    }
    m_passCount[m_slot] = 0;
}

uint32_t GpuTimers::beginPass(rhi::CommandList& cmd, const char* name) {
    const uint32_t pass = m_passCount[m_slot];
    if (pass >= kMaxPasses) return kMaxPasses;
    m_passCount[m_slot] = pass + 1;
    m_names[m_slot][pass] = name;
    m_cpuStart[pass] = platform::nowTicks();
    cmd.pushDebugGroup(name);
    cmd.writeTimestamp(m_pool, pass * 2);
    return pass;
}

void GpuTimers::endPass(rhi::CommandList& cmd, uint32_t pass) {
    if (pass >= kMaxPasses) return;
    cmd.writeTimestamp(m_pool, pass * 2 + 1);
    cmd.popDebugGroup();
    m_cpuMs[m_slot][pass] = platform::ticksToMilliseconds(platform::nowTicks() - m_cpuStart[pass]);
}

void GpuTimers::endFrame() {
    m_slot = (m_slot + 1) % rhi::kFramesInFlight; // stays in step with the device's frame slots
}

} // namespace renderer
