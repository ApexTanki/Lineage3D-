#pragma once

#include "rhi/Caps.h"

#include <cmath>

namespace renderer {

// Some backbuffers don't encode linear → sRGB on write (every GL default framebuffer, --caps-floor). Passes
// that write the backbuffer then encode in the shader. This is the single place that decides it
// (ARCHITECTURE.md §4.3).
inline bool shaderEncodesSrgb(const rhi::Caps& caps) {
    return !caps.srgbBackbuffer;
}

// Clears bypass shaders, so a linear clear color must be encoded on the CPU when the backbuffer won't.
inline void backbufferClearColor(const rhi::Caps& caps, const float linear[4], float out[4]) {
    for (int i = 0; i < 3; ++i) {
        const float c = linear[i] < 0.0f ? 0.0f : linear[i];
        out[i] = !shaderEncodesSrgb(caps) ? c
                 : c <= 0.0031308f        ? c * 12.92f
                                          : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
    }
    out[3] = linear[3];
}

} // namespace renderer
