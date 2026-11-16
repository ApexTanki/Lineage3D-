#pragma once

#include "core/Math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

// Flipbook animation (ARCHITECTURE.md R33; M6 plan D11): the mapping material flipbooks use (fl_material.glsl,
// flAnimateUv), sprites use, and M7's particles will use: frame = floor(time · fps), cells of a columns × rows atlas read
// row by row from the top-left cell (UV v = 0 is the top image row).

namespace renderer {

struct FlipbookFrame {
    uint32_t frame = 0;    // atlas cell
    bool finished = false; // a one-shot past its last frame
};

// The cell a flipbook shows `time` seconds after it started: frames [firstFrame, firstFrame + frameCount) at `fps`,
// looping or held… then finished for one-shots. fps ≤ 0 shows the first frame.
inline FlipbookFrame flipbookFrame(float time, float fps, uint32_t firstFrame, uint32_t frameCount, bool loop) {
    FlipbookFrame f;
    const uint32_t count = std::max(frameCount, 1u);
    if (!(fps > 0.0f) || time < 0.0f) {
        f.frame = firstFrame;
        return f;
    }
    const double index = std::floor(double(time) * double(fps));
    if (loop) {
        f.frame = firstFrame + uint32_t(std::fmod(index, double(count)));
    } else if (index >= double(count)) {
        f.frame = firstFrame + count - 1;
        f.finished = true;
    } else {
        f.frame = firstFrame + uint32_t(index);
    }
    return f;
}

// UV offset (xy) and scale (zw) of an atlas cell.
inline glm::vec4 flipbookRect(uint32_t frame, uint32_t columns, uint32_t rows) {
    const uint32_t c = std::max(columns, 1u), r = std::max(rows, 1u);
    const uint32_t cell = frame % (c * r);
    return glm::vec4(float(cell % c) / float(c), float(cell / c) / float(r), 1.0f / float(c), 1.0f / float(r));
}

} // namespace renderer
