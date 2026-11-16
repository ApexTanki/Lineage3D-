#pragma once

#include "ImageCook.h"
#include "MeshBuilder.h"

#include <string>

namespace tools {

// Procedural content for demo scenes and tests (ARCHITECTURE.md A13), addressed as "builtin:<spec>".
// Meshes (unit size, centered, CCW outside unless noted):
//   cube | sphere | cylinder | plane:<size m>:<uv repeats>         one material slot
//   box:<x m>:<y m>:<z m>:<tile m>                                  centered box; one texture repeat per <tile> meters
//   wedge:<x m>:<y m>:<z m>:<tile m>                                ramp rising along +Z, centered like a box
//   terrain:<size m>:<cells>:<tile m>:<center x>:<center z>         chunk of the demo arena heightfield (flat 40 m plaza,
//                                                                   hills toward the edges, a pond at (14, −14)), positions
//                                                                   relative to the chunk center, UV0 = world xz / tile
//   crate                                                           cube; slot 0 = sides, slot 1 = top and bottom
//   lamp                                                            lamp post standing on the origin, 2.65 m tall;
//                                                                   slot 0 = pole and base, slot 1 = bulb
// Every mesh gets lightmap UVs from the chart unwrapper (LightmapUv.h).
// Textures (tiling):
//   checker:<RRGGBB>:<RRGGBB>:<squares> | grid:<RRGGBB background>:<RRGGBB lines>:<cells>
//   bricks:<map> | plates:<map> | ground:<map> | concrete:<map>   map = albedo | normal | specgloss
//   water:<map>                      map = albedo | normal (ripples)
//   panel:<map>                      map = albedo | normal | emission
//   fence                            chain-link wires in alpha (albedo_cutout)
//   fan                              four fan blades in a ring, in alpha (albedo_cutout; spun by UV rotation)
//   stripes                          hazard stripes (a conveyor belt; scrolled by UV animation)
//   foam                             white streaks in alpha (a water detail layer)
//   radar                            4 × 4 flipbook of a radar sweep (emission)
//   fire | smoke | explosion           4 × 4 sprite flipbooks, straight alpha (albedo_alpha): a looping flame, a looping puff,
//                                    a one-shot fireball (M6 plan D11)
//   glow:<RRGGBB>                    radial falloff in alpha (albedo_alpha, for additive or blended sprites)
//   ramp:<steps>                     gray steps round(255·k/(steps−1)) left to right (raw_srgb, the sRGB test)
bool makeBuiltinMesh(const std::string& spec, RawMesh& out, std::string& error);
bool makeBuiltinImage(const std::string& spec, Image& out, std::string& error);

} // namespace tools
