#version 450
// Pass 2, the depth prepass (M8 plan D9): depth only. Without FADE: no inputs and no color output, so early depth testing
// stays on. FADE (a draw in a LOD cross-fade): the fade twin's dither, so the depth holds exactly the pixels the scene pass
// draws and fading objects keep their SSAO.
#ifdef FADE
#include "fl_lodfade.glsl"

layout(location = 0) flat in uint v_lodWord;
#endif

void main() {
#ifdef FADE
    if (flLodFadeDiscards(v_lodWord, gl_FragCoord.xy)) discard;
#endif
}
