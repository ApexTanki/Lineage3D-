#version 450
// Pass 9e (M8 plan D7): FXAA 3.11 "PC quality" (Timothy Lottes), 12 search steps, as the frame's last pass. The input
// is an sRGB8 image (ldrColor after the tonemap, or the LDR scene target), so sampling returns linear color; edges are
// found on the square root of luma, which is close to the gamma-encoded luma FXAA was tuned for.
#include "fl_common.glsl"
#include "fl_bindings.glsl"
#include "fl_post.glsl"

layout(set = 1, binding = 1) uniform sampler2D u_input;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

const float kSubpix = 0.75;             // amount of sub-pixel aliasing removal
const float kEdgeThreshold = 0.166;     // minimum local contrast, relative to the local maximum
const float kEdgeThresholdMin = 0.0833; // ... and absolute, so dark regions are left alone
const int kSteps = 12;
const float kStep[12] = float[12](1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0);

float lumaOf(vec3 c) {
    return sqrt(max(flLuma(c), 0.0));
}

float lumaAt(vec2 uv) {
    return lumaOf(textureLod(u_input, uv, 0.0).rgb);
}

void main() {
    vec2 rcpFrame = 1.0 / vec2(textureSize(u_input, 0));
    vec2 posM = v_uv;
    vec3 rgbM = textureLod(u_input, posM, 0.0).rgb;
    float lumaM = lumaOf(rgbM);
    float lumaS = lumaAt(posM + vec2(0.0, 1.0) * rcpFrame);
    float lumaE = lumaAt(posM + vec2(1.0, 0.0) * rcpFrame);
    float lumaN = lumaAt(posM + vec2(0.0, -1.0) * rcpFrame);
    float lumaW = lumaAt(posM + vec2(-1.0, 0.0) * rcpFrame);
    float rangeMax = max(max(lumaN, lumaW), max(lumaE, max(lumaS, lumaM)));
    float rangeMin = min(min(lumaN, lumaW), min(lumaE, min(lumaS, lumaM)));
    float range = rangeMax - rangeMin;
    if (range < max(kEdgeThresholdMin, rangeMax * kEdgeThreshold)) {
        o_color = flEncodeOutput(vec4(rgbM, 1.0));
        return;
    }
    float lumaNW = lumaAt(posM + vec2(-1.0, -1.0) * rcpFrame);
    float lumaSE = lumaAt(posM + vec2(1.0, 1.0) * rcpFrame);
    float lumaNE = lumaAt(posM + vec2(1.0, -1.0) * rcpFrame);
    float lumaSW = lumaAt(posM + vec2(-1.0, 1.0) * rcpFrame);

    float lumaNS = lumaN + lumaS;
    float lumaWE = lumaW + lumaE;
    float subpixRcpRange = 1.0 / range;
    float subpixNSWE = lumaNS + lumaWE;
    float edgeHorz1 = -2.0 * lumaM + lumaNS;
    float edgeVert1 = -2.0 * lumaM + lumaWE;
    float lumaNESE = lumaNE + lumaSE;
    float lumaNWNE = lumaNW + lumaNE;
    float edgeHorz2 = -2.0 * lumaE + lumaNESE;
    float edgeVert2 = -2.0 * lumaN + lumaNWNE;
    float lumaNWSW = lumaNW + lumaSW;
    float lumaSWSE = lumaSW + lumaSE;
    float edgeHorz4 = abs(edgeHorz1) * 2.0 + abs(edgeHorz2);
    float edgeVert4 = abs(edgeVert1) * 2.0 + abs(edgeVert2);
    float edgeHorz3 = -2.0 * lumaW + lumaNWSW;
    float edgeVert3 = -2.0 * lumaS + lumaSWSE;
    float edgeHorz = abs(edgeHorz3) + edgeHorz4;
    float edgeVert = abs(edgeVert3) + edgeVert4;
    float subpixNWSWNESE = lumaNWSW + lumaNESE;
    float lengthSign = rcpFrame.x;
    bool horzSpan = edgeHorz >= edgeVert;
    float subpixA = subpixNSWE * 2.0 + subpixNWSWNESE;
    if (!horzSpan) {
        lumaN = lumaW;
        lumaS = lumaE;
    } else {
        lengthSign = rcpFrame.y;
    }
    float subpixB = subpixA * (1.0 / 12.0) - lumaM;
    float gradientN = lumaN - lumaM;
    float gradientS = lumaS - lumaM;
    float lumaNN = lumaN + lumaM;
    float lumaSS = lumaS + lumaM;
    bool pairN = abs(gradientN) >= abs(gradientS);
    float gradient = max(abs(gradientN), abs(gradientS));
    if (pairN) lengthSign = -lengthSign;
    float subpixC = clamp(abs(subpixB) * subpixRcpRange, 0.0, 1.0);

    vec2 posB = posM;
    vec2 offNP = horzSpan ? vec2(rcpFrame.x, 0.0) : vec2(0.0, rcpFrame.y);
    if (!horzSpan) posB.x += lengthSign * 0.5;
    else posB.y += lengthSign * 0.5;
    vec2 posN = posB - offNP * kStep[0];
    vec2 posP = posB + offNP * kStep[0];
    float subpixD = -2.0 * subpixC + 3.0;
    float lumaEndN = lumaAt(posN);
    float subpixE = subpixC * subpixC;
    float lumaEndP = lumaAt(posP);
    if (!pairN) lumaNN = lumaSS;
    float gradientScaled = gradient * 0.25;
    float lumaMM = lumaM - lumaNN * 0.5;
    float subpixF = subpixD * subpixE;
    bool lumaMLTZero = lumaMM < 0.0;
    lumaEndN -= lumaNN * 0.5;
    lumaEndP -= lumaNN * 0.5;
    bool doneN = abs(lumaEndN) >= gradientScaled;
    bool doneP = abs(lumaEndP) >= gradientScaled;
    if (!doneN) posN -= offNP * kStep[1];
    if (!doneP) posP += offNP * kStep[1];
    for (int s = 2; s < kSteps && !(doneN && doneP); ++s) {
        if (!doneN) lumaEndN = lumaAt(posN) - lumaNN * 0.5;
        if (!doneP) lumaEndP = lumaAt(posP) - lumaNN * 0.5;
        doneN = doneN || abs(lumaEndN) >= gradientScaled;
        doneP = doneP || abs(lumaEndP) >= gradientScaled;
        if (!doneN) posN -= offNP * kStep[s];
        if (!doneP) posP += offNP * kStep[s];
    }

    float dstN = horzSpan ? posM.x - posN.x : posM.y - posN.y;
    float dstP = horzSpan ? posP.x - posM.x : posP.y - posM.y;
    bool goodSpanN = (lumaEndN < 0.0) != lumaMLTZero;
    bool goodSpanP = (lumaEndP < 0.0) != lumaMLTZero;
    float spanLengthRcp = 1.0 / (dstP + dstN);
    bool directionN = dstN < dstP;
    float dst = min(dstN, dstP);
    bool goodSpan = directionN ? goodSpanN : goodSpanP;
    float subpixG = subpixF * subpixF;
    float pixelOffset = dst * (-spanLengthRcp) + 0.5;
    float subpixH = subpixG * kSubpix;
    float pixelOffsetSubpix = max(goodSpan ? pixelOffset : 0.0, subpixH);
    if (!horzSpan) posM.x += pixelOffsetSubpix * lengthSign;
    else posM.y += pixelOffsetSubpix * lengthSign;
    o_color = flEncodeOutput(vec4(textureLod(u_input, posM, 0.0).rgb, 1.0));
}
