#include "rhi/vulkan/VkCommon.h"

namespace rhi::vk {

VkFormat toVkFormat(Format format) {
    switch (format) {
    case Format::Undefined: return VK_FORMAT_UNDEFINED;
    case Format::R8Unorm: return VK_FORMAT_R8_UNORM;
    case Format::RG8Unorm: return VK_FORMAT_R8G8_UNORM;
    case Format::RGBA8Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
    case Format::RGBA8Srgb: return VK_FORMAT_R8G8B8A8_SRGB;
    case Format::BGRA8Unorm: return VK_FORMAT_B8G8R8A8_UNORM;
    case Format::BGRA8Srgb: return VK_FORMAT_B8G8R8A8_SRGB;
    case Format::R16F: return VK_FORMAT_R16_SFLOAT;
    case Format::RG16F: return VK_FORMAT_R16G16_SFLOAT;
    case Format::RGBA16F: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Format::R11G11B10F: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case Format::R32F: return VK_FORMAT_R32_SFLOAT;
    case Format::D16: return VK_FORMAT_D16_UNORM;
    case Format::D24S8: return VK_FORMAT_D24_UNORM_S8_UINT;
    case Format::D32F: return VK_FORMAT_D32_SFLOAT;
    case Format::BC1Srgb: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    case Format::BC3Srgb: return VK_FORMAT_BC3_SRGB_BLOCK;
    case Format::BC3Unorm: return VK_FORMAT_BC3_UNORM_BLOCK;
    case Format::BC4: return VK_FORMAT_BC4_UNORM_BLOCK;
    case Format::BC5: return VK_FORMAT_BC5_UNORM_BLOCK;
    case Format::BC6HUfloat: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
    case Format::BC7Unorm: return VK_FORMAT_BC7_UNORM_BLOCK;
    case Format::BC7Srgb: return VK_FORMAT_BC7_SRGB_BLOCK;
    case Format::RG32F: return VK_FORMAT_R32G32_SFLOAT;
    case Format::RGB32F: return VK_FORMAT_R32G32B32_SFLOAT;
    case Format::RGBA32F: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case Format::RGBA16Sint: return VK_FORMAT_R16G16B16A16_SINT;
    case Format::RGBA8Snorm: return VK_FORMAT_R8G8B8A8_SNORM;
    case Format::RGBA8Uint: return VK_FORMAT_R8G8B8A8_UINT;
    case Format::RG16Unorm: return VK_FORMAT_R16G16_UNORM;
    case Format::RGBA32Uint: return VK_FORMAT_R32G32B32A32_UINT;
    case Format::Count: break;
    }
    return VK_FORMAT_UNDEFINED;
}

Format fromVkFormat(VkFormat format) {
    for (uint32_t i = 0; i < kFormatCount; ++i)
        if (toVkFormat(Format(i)) == format) return Format(i);
    return Format::Undefined;
}

const char* resultName(VkResult result) {
    switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    default: return "VkResult(other)";
    }
}

VkCompareOp toVkCompare(CompareOp op) {
    switch (op) {
    case CompareOp::Never: return VK_COMPARE_OP_NEVER;
    case CompareOp::Less: return VK_COMPARE_OP_LESS;
    case CompareOp::Equal: return VK_COMPARE_OP_EQUAL;
    case CompareOp::LessOrEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case CompareOp::Greater: return VK_COMPARE_OP_GREATER;
    case CompareOp::NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
    case CompareOp::GreaterOrEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case CompareOp::Always: return VK_COMPARE_OP_ALWAYS;
    }
    return VK_COMPARE_OP_ALWAYS;
}

VkBlendFactor toVkBlendFactor(BlendFactor factor) {
    switch (factor) {
    case BlendFactor::Zero: return VK_BLEND_FACTOR_ZERO;
    case BlendFactor::One: return VK_BLEND_FACTOR_ONE;
    case BlendFactor::SrcAlpha: return VK_BLEND_FACTOR_SRC_ALPHA;
    case BlendFactor::OneMinusSrcAlpha: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case BlendFactor::DstAlpha: return VK_BLEND_FACTOR_DST_ALPHA;
    case BlendFactor::OneMinusDstAlpha: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case BlendFactor::SrcColor: return VK_BLEND_FACTOR_SRC_COLOR;
    case BlendFactor::DstColor: return VK_BLEND_FACTOR_DST_COLOR;
    }
    return VK_BLEND_FACTOR_ONE;
}

VkBlendOp toVkBlendOp(BlendOp op) {
    switch (op) {
    case BlendOp::Add: return VK_BLEND_OP_ADD;
    case BlendOp::Subtract: return VK_BLEND_OP_SUBTRACT;
    case BlendOp::ReverseSubtract: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case BlendOp::Min: return VK_BLEND_OP_MIN;
    case BlendOp::Max: return VK_BLEND_OP_MAX;
    }
    return VK_BLEND_OP_ADD;
}

VkSamplerAddressMode toVkAddress(AddressMode mode) {
    switch (mode) {
    case AddressMode::Repeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case AddressMode::ClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case AddressMode::MirroredRepeat: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    }
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

void queryFormatSupport(VkPhysicalDevice physicalDevice, Caps& caps) {
    for (uint32_t i = 1; i < kFormatCount; ++i) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, toVkFormat(Format(i)), &props);
        const VkFormatFeatureFlags f = props.optimalTilingFeatures;
        FormatSupport& s = caps.formats[i];
        s.sampled = (f & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
        s.filterable = (f & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
        s.colorTarget = (f & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0;
        s.blendable = (f & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0;
        s.depthTarget = (f & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;
        s.vertex = (props.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) != 0;
    }
}

} // namespace rhi::vk
