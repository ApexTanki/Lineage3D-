#pragma once

#include "MeshBuilder.h"

#include <cstdint>

namespace tools {

// UV1 layouts are made at this density; ModelMesh::lightmapWidth/Height are in texels at it (M3 plan D3).
inline constexpr float kLightmapReferenceDensity = 4.0f; // texels per meter

// Lightmap UVs for a mesh without its own: triangles are flood-filled into charts whose normals stay within 30° of the
// chart's seed (the largest unassigned triangle), each chart is projected onto its seed plane in meters, and the
// charts are shelf-packed at kLightmapReferenceDensity with one texel of padding on every side, so bilinear filtering
// never reads a neighbour chart. Vertices on chart borders are split; index order (and so submesh ranges) is kept.
// Writes every vertex's uv1 and the layout size in texels.
void generateLightmapUvs(RawMesh& mesh, uint16_t& width, uint16_t& height);

// Reference size for a mesh that brings its own UV1 (glTF TEXCOORD_1), estimated from its surface area.
void lightmapSizeFromArea(const RawMesh& mesh, uint16_t& width, uint16_t& height);

} // namespace tools
