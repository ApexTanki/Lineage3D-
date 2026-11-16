#pragma once

#include "core/Handle.h"
#include "renderer/resources/ResourceManager.h"

// Sprites (ARCHITECTURE.md §5.6; M6 plan D11): flipbook billboards placed by the game, such as fires, smoke and muzzle
// flashes. M7's particles use the same billboards and flipbook mapping.

namespace renderer {

// M9 plan D8: muzzle flashes and other short sprites churn through the slots: 10 index bits, 22 for generations.
using SpriteId = core::Handle<struct SceneSpriteTag, 10>;

enum class SpriteBlend : uint8_t { Alpha, Additive, Premultiplied };
enum class SpriteFacing : uint8_t {
    Camera, // faces the camera, turned by `rotation` about the view direction
    AxisY   // stands upright and turns about the world Y axis toward the camera (flames, trees in the distance)
};

struct SpriteDesc {
    TextureAsset atlas;           // the flipbook texture; drawn once it is Ready
    uint8_t columns = 1, rows = 1; // the atlas grid
    uint16_t firstFrame = 0, frameCount = 1;
    float fps = 0.0f;             // 0: the first frame only
    bool loop = true;             // false: plays once from startTime, then draws nothing (the game removes it)
    float startTime = 0.0f;       // in the time the world renders with (FrameUniforms.time.x)
    glm::vec3 position{0.0f};     // the sprite's center, world space
    glm::vec2 size{1.0f};         // meters
    float rotation = 0.0f;        // radians (camera-facing sprites)
    glm::vec4 color{1.0f};        // linear RGB multiplier and opacity
    SpriteFacing facing = SpriteFacing::Camera;
    SpriteBlend blend = SpriteBlend::Alpha;
};

} // namespace renderer
