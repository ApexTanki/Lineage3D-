#pragma once

// volk loads every Vulkan entry point at runtime, so no prototypes are linked and the SDK isn't needed
// to run. VMA fetches its functions through the pointers we hand it.
#include <volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include "core/Log.h"
#include "rhi/Caps.h"
#include "rhi/Descs.h"

namespace rhi::vk {

VkFormat toVkFormat(Format format);
Format fromVkFormat(VkFormat format);
const char* resultName(VkResult result);
VkCompareOp toVkCompare(CompareOp op);
VkBlendFactor toVkBlendFactor(BlendFactor factor);
VkBlendOp toVkBlendOp(BlendOp op);
VkSamplerAddressMode toVkAddress(AddressMode mode);
void queryFormatSupport(VkPhysicalDevice physicalDevice, Caps& caps);

} // namespace rhi::vk

// Logs failures with the expression and result name. Use where a failure is recoverable by the caller.
#define FL_VK_OK(expr) ::rhi::vk::checkResult((expr), #expr, __FILE__, __LINE__)

namespace rhi::vk {
inline bool checkResult(VkResult result, const char* expr, const char* file, int line) {
    if (result >= VK_SUCCESS) return true;
    FL_LOG_ERROR("Vulkan call failed: %s -> %s (%s:%d)", expr, resultName(result), file, line);
    return false;
}
} // namespace rhi::vk
