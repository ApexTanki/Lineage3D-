#pragma once

#include "rhi/Device.h"

namespace rhi {

// Chooses RHI_Vulkan or RHI_GL at startup (ARCHITECTURE.md §3.1). Linked only by applications.
core::Result<std::unique_ptr<Device>> createDevice(const DeviceDesc& desc);

} // namespace rhi
