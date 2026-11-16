#pragma once

#include "rhi/Types.h"

namespace rhi {

struct FormatSupport {
    bool sampled = false;
    bool filterable = false;
    bool colorTarget = false;
    bool blendable = false;
    bool depthTarget = false;
    bool vertex = false;
};

struct Caps {
    Backend backend = Backend::Vulkan;
    ClipSpaceConvention clipSpace;
    char deviceName[128] = {};
    char apiVersion[64] = {};
    FormatSupport formats[kFormatCount]; // queried per format; drives every fallback chain

    // Feature flags. "floor" = the value under --caps-floor (a minimal GL 3.3 core device, ARCHITECTURE.md A12).
    // Float color targets need no flag: RGBA16F and R11G11B10F are renderable and blendable on GL 3.3 core,
    // and RGBA16F is on Vulkan. Base instance is absent: GL 3.3 core lacks it (ARB_base_instance is GL 4.2).
    bool bc1to5 = false;                // S3TC + RGTC (Vulkan: textureCompressionBC)                floor: true
    bool bptc = false;                  // BC6H + BC7                                                floor: false
    bool linearDepthCompare = false;    // hardware 2x2 PCF; otherwise a 4-tap shader fallback       floor: false
    bool timestampQueries = false;      //                                                           floor: true
    bool wireframe = false;             // fillModeNonSolid / glPolygonMode                          floor: true
    bool multisampleTextures = false;   // sampleable MSAA images                                    floor: true
    bool depthResolveExtension = false; // VK_KHR_depth_stencil_resolve (GL always blits)            floor: n/a
    bool programBinaryCache = false;    // ARB_get_program_binary on GL                              floor: false
    bool srgbBackbuffer = false;        // the backbuffer encodes linear → sRGB on write             floor: false
    bool debugLabels = false;           // object names + debug groups visible in RenderDoc
    bool dynamicRendering = false;      // Vulkan fast path, never required (unused in M0)
    bool synchronization2 = false;      // Vulkan fast path, never required (unused in M0)

    float maxAnisotropy = 1.0f;
    uint32_t maxMsaaSamples = 1;
    uint32_t uniformOffsetAlignment = 256;
    uint32_t maxUniformBlockBytes = 16384;
    uint32_t maxFragmentTextureUnits = 16;
    uint32_t maxTextureSize = 2048;
    Format depthFormat = Format::D24S8; // D24S8, else D32F
};

} // namespace rhi
