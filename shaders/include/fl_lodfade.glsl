#ifndef FL_LODFADE_GLSL
#define FL_LODFADE_GLSL

// LOD cross-fade (M5 plan D7): during a level change both levels draw, the incoming one where the 4×4 Bayer threshold is
// under the progress and the outgoing one elsewhere, so every pixel shows exactly one of them. `word` is the draw's LOD
// word (renderer::packLodWord): bit 9 fading, bit 8 incoming, bits 0–7 progress. The lit fade twins and the depth
// prepass's FADE variant (M8) share this test, so the prepass keeps exactly the pixels the scene pass draws.

// 4×4 Bayer thresholds in (0, 1).
float flBayer4(vec2 pixel) {
    const float kBayer[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
    ivec2 p = ivec2(pixel) & 3;
    return (kBayer[p.y * 4 + p.x] + 0.5) / 16.0;
}

bool flLodFadeDiscards(uint word, vec2 pixel) {
    if ((word & 0x200u) == 0u) return false;
    float progress = float(word & 0xFFu) / 255.0;
    float threshold = flBayer4(pixel);
    return (word & 0x100u) != 0u ? threshold >= progress : threshold < progress;
}

#endif
