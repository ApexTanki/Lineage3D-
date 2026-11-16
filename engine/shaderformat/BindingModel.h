#pragma once

#include <cstdint>

namespace shaderformat {

// Binding convention (ARCHITECTURE.md §4.6): set 0 per frame, 1 per pass, 2 per material, 3 per-draw
// resources, plus one per-draw constant block (push constants on Vulkan, a UBO range on GL).
inline constexpr uint32_t kMaxSets = 4;
inline constexpr uint32_t kMaxPerDrawBytes = 128;

// GL has no descriptor sets, and GLSL 330 can't declare bindings, so the GL backend maps
// (set, binding) to fixed uniform-buffer binding points and texture units after linking. Inside a set,
// uniform buffers use the lowest binding numbers and textures follow.
struct SetSlots {
    uint8_t uboBase;
    uint8_t uboCount;
    uint8_t firstTextureBinding;
    uint8_t textureBase;
    uint8_t textureCount;
};

inline constexpr SetSlots kSetSlots[kMaxSets] = {
    {0, 2, 2, 0, 5},  // set 0: FrameUniforms, LightArray | CSM, spot atlas, sky, lightmaps, shadowmask
    {2, 1, 1, 5, 4},  // set 1: PassUniforms | 4 pass inputs
    {3, 1, 1, 9, 5},  // set 2: MaterialUniforms | albedo, normal, spec+gloss, emission, detail
    {4, 1, 1, 14, 1}, // set 3: BonePalette | local reflection cube
};
inline constexpr uint8_t kPerDrawUboPoint = 5;
inline constexpr uint8_t kInvalidSlot = 0xFF;

// GL uniform-buffer binding point for (set, binding), or kInvalidSlot if that binding isn't a UBO slot.
constexpr uint8_t glUniformBufferPoint(uint32_t set, uint32_t binding) {
    if (set >= kMaxSets) return kInvalidSlot;
    const SetSlots& s = kSetSlots[set];
    return binding < s.uboCount ? uint8_t(s.uboBase + binding) : kInvalidSlot;
}

// GL texture unit for (set, binding), or kInvalidSlot if that binding isn't a texture slot.
constexpr uint8_t glTextureUnit(uint32_t set, uint32_t binding) {
    if (set >= kMaxSets) return kInvalidSlot;
    const SetSlots& s = kSetSlots[set];
    if (binding < s.firstTextureBinding || binding >= uint32_t(s.firstTextureBinding) + s.textureCount)
        return kInvalidSlot;
    return uint8_t(s.textureBase + (binding - s.firstTextureBinding));
}

// 15 of the 16 fragment texture units that GL 3.3 guarantees.
static_assert(kSetSlots[3].textureBase + kSetSlots[3].textureCount <= 16, "texture units exceed the GL floor");

} // namespace shaderformat
