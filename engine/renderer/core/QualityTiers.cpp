#include "renderer/core/QualityTiers.h"

namespace renderer {
namespace {

ShadowSettings tierShadows(const assets::TierDesc& t, ShadowSettings s) {
    s.cascades = uint8_t(t.cascades);
    s.mapSize = uint16_t(t.shadowMapSize);
    s.filter = t.shadowFilter == assets::TierShadowFilter::Poisson ? ShadowFilter::Poisson16 : ShadowFilter::Hardware2x2;
    s.staticCasters = t.staticCasters;
    s.spotShadows = t.spotShadows;
    s.blobShadows = t.blobShadows;
    s.blobStrength = t.blobStrength;
    s.distance = t.shadowDistance;
    return s;
}

} // namespace

void applyTier(const assets::TierDesc& t, FrameSettings& s) {
    s.msaa = uint8_t(t.msaa);
    s.anisotropy = float(t.anisotropy);
    s.maxLocalLights = uint8_t(t.maxLocalLights);
    s.softParticles = t.softParticles;
    s.shadows = tierShadows(t, s.shadows);
    PostSettings& p = s.post;
    p.hdr = t.hdr;
    p.fxaa = t.fxaa;
    p.ssao = t.ssao;
    p.bloom = t.bloom;
    p.refraction = t.refraction;
    p.heatHaze = t.heatHaze;
    p.depthOfField = t.depthOfField;
    p.motionBlur = t.motionBlur;
    p.godRays = t.godRays;
    p.ssr = t.ssr;
}

MemoryBudgets tierBudgets(const assets::TierDesc& t) {
    constexpr uint64_t kMiB = 1ull << 20;
    return {t.textureBudgetMiB * kMiB, t.meshBudgetMiB * kMiB, t.targetBudgetMiB * kMiB, t.bufferBudgetMiB * kMiB};
}

bool matchesTier(const assets::TierDesc& t, const FrameSettings& s) {
    const ShadowSettings want = tierShadows(t, s.shadows);
    const ShadowSettings& have = s.shadows;
    const PostSettings& p = s.post;
    return s.msaa == t.msaa && s.anisotropy == float(t.anisotropy) && s.maxLocalLights == t.maxLocalLights &&
           s.softParticles == t.softParticles && have.cascades == want.cascades && have.mapSize == want.mapSize &&
           have.filter == want.filter && have.staticCasters == want.staticCasters && have.spotShadows == want.spotShadows &&
           have.blobShadows == want.blobShadows && have.blobStrength == want.blobStrength && have.distance == want.distance &&
           p.hdr == t.hdr && p.fxaa == t.fxaa && p.ssao == t.ssao && p.bloom == t.bloom && p.refraction == t.refraction &&
           p.heatHaze == t.heatHaze && p.depthOfField == t.depthOfField && p.motionBlur == t.motionBlur && p.godRays == t.godRays &&
           p.ssr == t.ssr;
}

} // namespace renderer
