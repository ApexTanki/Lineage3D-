#pragma once

#include "assets/TierFormat.h"
#include "renderer/core/Renderer.h"

namespace renderer {

// Quality tiers (ARCHITECTURE.md §7.9, M9 plan D1, D2): the tier file's data on the renderer's settings.
//
// applyTier puts a tier's settings into `settings`: MSAA, anisotropy, the local-light limit, soft particles, the shadow
// block and the post toggles (HDR, FXAA, SSAO, bloom, refraction, heat haze, the P2 effects). Everything a tier doesn't
// hold keeps its value: the debug view, culling and LOD switches, the tone curve and the post tuning values, particle
// speeds. A tier switch takes effect on the next frame; MSAA and HDR changes rebuild targets and pipelines (§4.11).
void applyTier(const assets::TierDesc& tier, FrameSettings& settings);

// Whether `settings` holds exactly the tier's values in every field applyTier sets (the demo's Quality panel shows
// "Custom" when no tier matches).
bool matchesTier(const assets::TierDesc& tier, const FrameSettings& settings);

// The tier's GPU memory budgets (M9 plan D6), for Renderer::setMemoryBudgets.
MemoryBudgets tierBudgets(const assets::TierDesc& tier);

} // namespace renderer
