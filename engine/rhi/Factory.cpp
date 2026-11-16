#include "rhi/Factory.h"

namespace rhi {
namespace vk {
core::Result<std::unique_ptr<Device>> createVulkanDevice(const DeviceDesc& desc);
}
namespace gl {
core::Result<std::unique_ptr<Device>> createGlDevice(const DeviceDesc& desc);
}

core::Result<std::unique_ptr<Device>> createDevice(const DeviceDesc& desc) {
    if (desc.backend == Backend::Vulkan) return vk::createVulkanDevice(desc);
    return gl::createGlDevice(desc);
}

} // namespace rhi
