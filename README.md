# FlashLineage3D render engine

FlashLineage3D is a **3D renderer**: the part of a game engine that draws the graphics. It was built for a tank game
in the style of Tanki Online. The goal is good-looking 3D that runs well on weak computers too: baked lighting and
cheap, well-chosen tricks instead of expensive modern effects. It has two backends behind one interface, **Vulkan 1.1**
and **OpenGL 3.3 core**, and three quality tiers, **Low, Medium and High**.

The renderer draws maps with baked lighting, animated tanks with dynamic lights and shadows, particles (smoke, fire,
explosions, sparks, dust) and screen effects (bloom, refraction through water and glass, anti-aliasing). It is **not**
a complete game engine: physics, sound, networking and game logic belong to the game.

This package holds the engine only. The demo program, the tests, the demo content and the full documentation are in
the complete FlashLineage3D project.

## What's in this package

| Folder | Contents |
|---|---|
| `engine/` | The engine libraries: `core`, `platform` (window, input, job system), `rhi` (the Vulkan and GL backends), `renderer`, `anim`, `world` (the API the game uses), `assets` (file formats), `shaderformat` |
| `shaders/` | GLSL sources, compiled into shader packages when you build |
| `tools/` | `fl_shaderc` (the shader compiler) and `fl_assetc` (cooks glTF models, textures and materials, and bakes maps) |
| `assets/source/` | The cook manifest (`cook.json`) and the quality tiers (`settings/tiers.json`) |
| `cmake/`, `CMakeLists.txt`, `CMakePresets.json`, `vcpkg.json` | The build |
| `scripts/bootstrap.ps1` | Fetches vcpkg, which builds the third-party libraries |

## Requirements

- **Windows 10 or 11, x64**, and a GPU with Vulkan 1.1 or OpenGL 3.3 support.
- **Visual Studio 2022** (Community is fine) with **Desktop development with C++** and **C++ CMake tools for
  Windows**; optionally **C++ Clang Compiler for Windows** for the `windows-clangcl-dev` preset.
- **Git** (the bootstrap script clones vcpkg).
- Several GB of disk space: vcpkg and the built libraries stay inside this folder (`external/`, `.cache/`).

## Building

1. Open PowerShell in this folder and run `.\scripts\bootstrap.ps1` (once).
2. Open **Developer PowerShell for VS 2022** and `cd` to this folder.
3. `cmake --preset windows-msvc-dev`. The first time, vcpkg downloads and builds the third-party libraries, which takes
   a while.
4. `cmake --build --preset windows-msvc-dev`

The presets are `windows-msvc-debug`, `windows-msvc-dev` (optimized, with developer features: validation, debug
panels, hot reload), `windows-msvc-release` and `windows-clangcl-dev`.

The build gives, in `build\<preset>\`:
- the engine's static libraries (`fl_world`, `fl_renderer`, `fl_rhi_factory`, `fl_platform` and the ones they use);
- `bin\fl_shaderc.exe` and `bin\fl_assetc.exe`;
- `bin\data\`: the shader packages the renderer loads (`shaders\*.fsp`) and the quality tiers (`settings\tiers.json`).

## Using the engine from a game

Add your game to this CMake project; the end of `CMakeLists.txt` shows how. A game executable links `fl_world`,
`fl_renderer`, `fl_rhi_factory` and `fl_platform`, depends on `fl_shaders` and `fl_cook`, and ships the `data\` folder
next to it.

The game talks to `world::RenderWorld`, a flat API with explicit add, update and remove and no scene graph:

```cpp
#include "renderer/core/Renderer.h"
#include "world/RenderWorld.h"

// Once: an rhi::Device (rhi::createDevice) and a platform::JobSystem, then the renderer and the world.
renderer::Renderer renderer;
if (!renderer.init({&device, &jobs, dataDir})) return;   // dataDir: the data folder, with a trailing separator
world::RenderWorld world(renderer);
world.loadMap("maps/my_map");                            // a map baked by fl_assetc: lightmaps, probes, sky, fog

// A tank: models load asynchronously and appear once they are ready.
world::ModelDesc tank;
tank.model = renderer.resources().loadModel("models/my_tank.fmodel");
tank.transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 10.0f));
tank.blobShadow = true;
const world::ModelId tankId = world.add(tank);

// An explosion: one-shot effects free themselves when they finish.
world::EffectDesc blast;
blast.preset = renderer.resources().loadPreset("particles/explosion.fpart");
blast.position = glm::vec3(0.0f, 0.0f, 12.0f);
world.add(blast);

const world::CameraId camera = world.add(world::CameraDesc{worldFromView, 60.0f, 0.2f, 1000.0f});
renderer::FrameSettings settings;                        // MSAA, shadows, post-processing, culling, LOD...
settings.post = renderer::PostSettings::high();          // or a quality tier: renderer::applyTier(tier, settings)

// Every frame:
world.setTransform(tankId, tankTransform);
world.update(camera, world::CameraDesc{worldFromView, 60.0f, 0.2f, 1000.0f});
world.render(camera, settings, timeSeconds, deltaSeconds);
```

Other calls cover point and spot lights, turret and gun overrides and attach points, skeletal animation clips,
flipbook sprites and the environment (sun, ambient light, fog, sky and exposure).

## Content

`fl_assetc` turns source content into the engine's formats: glTF models into `.fmodel` (with LODs), images into KTX2
textures, material and particle-preset files, animation clips, and map descriptions into baked maps (lightmaps,
light probes, reflection probes). List your content in `assets/source/cook.json`; the build cooks it into `data\`.

## Quality tiers

`assets/source/settings/tiers.json` defines Low, Medium, High and Custom: MSAA, HDR, shadows, every post effect, the
light count, anisotropy and memory budgets per tier. A game parses the file with `assets::parseTiers` and applies a
tier with `renderer::applyTier`; it can switch tiers at runtime.

## License

No license has been chosen yet; all rights are reserved.
