[FlashLineage3D-README.md](https://github.com/user-attachments/files/34921672/FlashLineage3D-README.md)
# FlashLineage3D

## What is it?

FlashLineage3D is a **3D renderer**: the part of a game engine that draws the graphics on screen. It was built for a
tank game in the style of Tanki Online.

The main goal: **good-looking 3D that runs well even on weak computers.** Instead of expensive modern effects, it uses
lighting computed in advance (baked) and cheap, well-chosen tricks.

## What is it for?

The game tells the renderer what is in the scene (tanks, buildings, lights, the camera, explosions), and the renderer
draws it:

- **maps** with baked lighting, textures and a sky;
- **tanks** and other moving objects, with animation, dynamic lights and shadows;
- **effects:** smoke, fire, explosions, sparks and dust;
- **screen effects:** glow (bloom), refraction through water and glass, and anti-aliasing.

It has three quality levels, **Low, Medium and High**, so it runs well on both weak and strong computers.

## What is it not?

It is **not a complete game engine.** Physics, sound, networking and game logic are not part of it; the game adds
those. The renderer only handles the graphics.

## What's included?

- **The renderer library:** the game links it into its own code. It runs on two graphics APIs, **Vulkan** and
  **OpenGL**.
- **The asset tool (`fl_assetc`):** converts 3D models, textures and maps into the formats the renderer loads.
- **The demo program (`fl_demo`):** test scenes that show every feature.

## How do I try it?

In **Developer PowerShell for VS 2022**:

```powershell
cd "D:\FlashLineage3D Engine"
cmake --preset windows-msvc-dev
cmake --build --preset windows-msvc-dev
build\windows-msvc-dev\bin\fl_demo.exe
```

Controls in the demo:
- hold the right mouse button and move the mouse: look around;
- WASD: move;
- Q/E: down/up.

## Status

All milestones are done. It runs on **Windows** only. The full technical description is in the project's `README.md`.
