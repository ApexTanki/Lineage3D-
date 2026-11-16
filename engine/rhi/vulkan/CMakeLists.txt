add_library(fl_rhi_vulkan STATIC
    VkConvert.cpp
    VulkanDevice.cpp
    VkResources.cpp
    VkCommands.cpp
    VmaImpl.cpp)
target_link_libraries(fl_rhi_vulkan
    PUBLIC fl_rhi
    PRIVATE volk::volk volk::volk_headers Vulkan::Headers GPUOpen::VulkanMemoryAllocator)
fl_target_defaults(fl_rhi_vulkan)
