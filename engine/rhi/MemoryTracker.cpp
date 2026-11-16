#include "rhi/MemoryTracker.h"

#include "core/Log.h"

#include <algorithm>

namespace rhi {

const char* gpuMemoryCategoryName(GpuMemoryCategory category) {
    static constexpr const char* kNames[] = {"textures", "render targets", "meshes", "buffers"};
    return uint32_t(category) < kGpuMemoryCategories ? kNames[uint32_t(category)] : "?";
}

GpuMemoryCategory memoryCategory(const TextureDesc& desc) {
    return any(desc.usage & (TextureUsage::ColorTarget | TextureUsage::DepthTarget)) ? GpuMemoryCategory::RenderTargets
                                                                                      : GpuMemoryCategory::Textures;
}

GpuMemoryCategory memoryCategory(const BufferDesc& desc) {
    return any(desc.usage & (BufferUsage::Vertex | BufferUsage::Index)) ? GpuMemoryCategory::Meshes : GpuMemoryCategory::Buffers;
}

uint64_t memoryBytes(const TextureDesc& desc) {
    const FormatInfo f = formatInfo(desc.format);
    const uint64_t layers = desc.type == TextureType::Cube ? 6u : std::max<uint32_t>(1u, desc.layers);
    const uint32_t block = std::max<uint32_t>(1u, f.blockSize);
    uint64_t bytes = 0;
    for (uint32_t m = 0; m < std::max<uint32_t>(1u, desc.mips); ++m) {
        const uint64_t w = std::max(1u, desc.width >> m), h = std::max(1u, desc.height >> m);
        bytes += ((w + block - 1) / block) * ((h + block - 1) / block) * f.blockBytes;
    }
    return bytes * layers * std::max<uint32_t>(1u, desc.samples);
}

bool MemoryTracker::reserve(GpuMemoryCategory category, uint64_t bytes, const char* name) {
    const uint32_t c = uint32_t(category);
    ++m_reservations;
    if (m_failFrom && (m_failLater ? m_reservations >= m_failFrom : m_reservations == m_failFrom)) {
        ++m_refusals;
        FL_LOG_WARN("Simulated GPU allocation failure #%llu: %s (%s, %.2f MiB)", static_cast<unsigned long long>(m_reservations),
                    name && *name ? name : "unnamed", gpuMemoryCategoryName(category), double(bytes) / (1024.0 * 1024.0));
        return false;
    }
    if (m_budget[c] && m_usage[c] + bytes > m_budget[c]) {
        ++m_refusals;
        FL_LOG_WARN("GPU memory budget: %s (%.2f MiB) would take %s to %.1f of %.1f MiB: refused", name && *name ? name : "unnamed",
                    double(bytes) / (1024.0 * 1024.0), gpuMemoryCategoryName(category), double(m_usage[c] + bytes) / (1024.0 * 1024.0),
                    double(m_budget[c]) / (1024.0 * 1024.0));
        return false;
    }
    m_usage[c] += bytes;
    return true;
}

void MemoryTracker::release(GpuMemoryCategory category, uint64_t bytes) {
    uint64_t& usage = m_usage[uint32_t(category)];
    usage -= std::min(usage, bytes);
}

uint64_t MemoryTracker::total() const {
    uint64_t sum = 0;
    for (uint64_t u : m_usage) sum += u;
    return sum;
}

} // namespace rhi
