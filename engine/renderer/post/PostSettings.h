#pragma once

#include <cstdint>

namespace renderer {

// Post-processing and image quality (M8 plan D2). The presets follow the tier table (ARCHITECTURE.md §7.9); the MSAA
// count stays FrameSettings::msaa, and exposure is a scene value (LightingData::exposure; R8: no eye adaptation).
struct PostSettings {
    enum class Tonemapper : uint8_t {
        Aces,  // Stephen Hill's fit of the ACES RRT + sRGB ODT (default)
        Hable, // John Hable's Uncharted 2 curve (exposure bias 2, white point 11.2)
        Clamp  // no curve: the display before M8 (the sRGB and shadow tests check encoding and shadows with it)
    };

    bool hdr = true; // false: the LDR path (sRGB8 scene target, exposure and the curve inside the forward shaders)
    Tonemapper tonemapper = Tonemapper::Aces;
    bool bloom = true;
    float bloomIntensity = 0.06f; // scales the accumulated bloom chain where the tonemap adds it
    float bloomThreshold = 1.0f;  // brightness (after exposure) where bloom starts
    float bloomKnee = 0.5f;       // width of the soft knee below the threshold, relative to it
    bool fxaa = false;
    bool ssao = false;            // High: also turns the depth prepass on (M8 plan D9)
    float ssaoRadius = 0.6f;      // m
    float ssaoIntensity = 1.5f;
    bool refraction = true;       // water and glass refract the scene behind them; off: they blend
    bool heatHaze = true;         // distort emitters bend the picture; off: they aren't drawn
    // P2 effects (High only, each toggleable, off by default; M8 plan D12).
    bool depthOfField = false;
    float focusDistance = 12.0f;  // m from the camera
    float focusRange = 10.0f;     // m from the focus distance to the full blur
    bool motionBlur = false;
    float motionBlurScale = 0.5f; // shutter: the fraction of the frame interval the blur spans
    bool godRays = false;
    float godRaysIntensity = 0.3f;
    bool ssr = false;             // screen-space reflections on refractive surfaces

    static PostSettings low() {
        PostSettings s;
        s.hdr = false;
        s.bloom = false;
        s.fxaa = true;
        s.refraction = false;
        s.heatHaze = false;
        return s;
    }
    static PostSettings medium() { return PostSettings{}; }
    static PostSettings high() {
        PostSettings s;
        s.ssao = true;
        return s;
    }
};

} // namespace renderer
