#pragma once

#include "HdrCodec.h"
#include "RayTracer.h"
#include "Sky.h"
#include "assets/MapFormat.h"

#include <cstdint>
#include <vector>

namespace tools {

// A material map decoded for bake-time sampling (a small mip), linear RGBA, rows top to bottom.
struct BakeTexture {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<glm::vec4> texels;
    glm::vec4 sample(const glm::vec2& uv) const; // bilinear, repeating
};

struct BakeMaterial {
    glm::vec4 baseColor{1.0f};
    glm::vec3 emission{0.0f}; // color × intensity
    int32_t albedo = -1;      // Baker texture index, −1 = none
    int32_t emissionMap = -1;
    bool alphaTest = false;
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
    bool castShadows = true; // false: transparent to sun and baked-light shadow rays, like the shadow maps (M4)
};

// World-space triangles of one static mesh node. A mesh without a lightmap rect still casts shadows and bounces light.
struct BakeMesh {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uv0;
    std::vector<glm::vec2> uv1;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> materials; // per triangle
    int32_t lightmapRect = -1;
};

struct LightmapRect {
    uint32_t page = 0;
    uint32_t x = 0, y = 0;
    uint32_t width = 0, height = 0;
};

struct BakeLight {
    bool spot = false;
    glm::vec3 position{0.0f};
    glm::vec3 direction{0.0f, -1.0f, 0.0f}; // spots: where the cone points
    glm::vec3 color{1.0f};                  // color × intensity (π folded in, like runtime lights)
    float range = 10.0f;
    float cosInner = 1.0f;
    float cosOuter = 0.0f;
    float radius = 0.1f; // source size, for soft shadows
};

struct BakeSettings {
    uint32_t samples = 256;       // cosine-weighted hemisphere rays per lightmap texel
    uint32_t bounces = 2;
    uint32_t sunSamples = 64;     // shadowmask rays per texel (and probe sun occlusion rays)
    uint32_t lightSamples = 64;   // shadow rays per baked light per texel (soft shadows of the light's sphere)
    uint32_t probeRays = 2048;
    float sunAngularRadius = 0.012f; // radians; the shadowmask penumbra
};

// The offline light baker of `fl_assetc map` (M3 plan D1, D4, D6): a CPU path tracer over the map's static meshes.
//   Lightmap texel: E/π from the sky, light bounced off surfaces (the sun's bounce included), emissive surfaces and baked
//                   local lights — never the sun's direct term. Shadowmask texel: the visible fraction of the sun disc.
//   Probe: L1 SH of the light arriving at a point (surfaces shaded with the baked lightmaps plus sun direct, the sky,
//          baked lights analytically) in irradiance form, its sun visibility, and whether it lies inside geometry.
//   Cube: the radiance seen from a point (reflection probes), surfaces shaded the same way.
// Alpha-tested materials are cut out by their albedo alpha. Texels and probes whose rays mostly reach back faces lie
// inside geometry; texels are then rebuilt from valid neighbours, and probes are flagged invalid. Deterministic: every
// texel and probe seeds its own random sequence.
class Baker {
public:
    Baker(const BakeSettings& settings, const SkySettings& sky, const glm::vec3& sunDirection, const glm::vec3& sunColor);

    uint32_t addTexture(BakeTexture texture);
    uint32_t addMaterial(const BakeMaterial& material);
    void addMesh(BakeMesh mesh);
    void addLight(const BakeLight& light);
    void build();

    void bakeLightmaps(const std::vector<LightmapRect>& rects, uint32_t pageSize, uint32_t pages);
    const std::vector<HdrImage>& lightmapPages() const { return m_light; }
    const std::vector<std::vector<float>>& shadowmaskPages() const { return m_shadow; }
    uint32_t bakedTexels() const { return m_bakedTexels; }
    uint32_t invalidTexels() const { return m_invalidTexels; }

    void bakeProbes(assets::ProbeGridData& grid) const; // uses grid.dims, origin and spacing
    CubeFaces renderCube(const glm::vec3& position, uint32_t faceSize) const;

    uint32_t triangleCount() const { return m_scene.triangleCount(); }

private:
    struct Triangle {
        uint32_t mesh;
        uint32_t i0, i1, i2;
        uint32_t material;
        glm::vec3 normal; // geometric, unit length
    };
    struct Surface {
        glm::vec3 position, normal, geometricNormal;
        glm::vec2 uv0, uv1;
        const BakeMaterial* material;
        int32_t rect;
    };
    struct Rng {
        uint64_t state;
        explicit Rng(uint64_t seed);
        uint32_t next();
        float uniform();
    };

    static bool alphaFilter(const void* context, uint32_t triangle, float u, float v);
    // Sun shadow rays: like alphaFilter, and materials with castShadows false let them through (the shadow maps skip them too).
    static bool sunShadowFilter(const void* context, uint32_t triangle, float u, float v);
    // Shadow rays toward baked lights also pass through emissive surfaces: a lamp's bulb must not shadow its own light.
    static bool lightShadowFilter(const void* context, uint32_t triangle, float u, float v);
    Surface surface(uint32_t triangle, float u, float v) const;
    glm::vec3 albedo(const Surface& s) const;
    glm::vec3 emission(const Surface& s) const;
    glm::vec3 sky(const glm::vec3& direction) const;
    float sunVisibility(const glm::vec3& origin, uint32_t samples, float offset0, float offset1) const;
    glm::vec3 lightDirect(const BakeLight& light, const glm::vec3& origin, const glm::vec3& normal, uint32_t samples, Rng& rng) const;
    glm::vec3 directAt(const Surface& s, const glm::vec3& origin, Rng& rng) const; // sun + baked lights, one sample each
    glm::vec3 trace(const Ray& ray, uint32_t depth, Rng& rng, bool& backFace) const;
    glm::vec3 lightmapAt(const Surface& s) const;
    glm::vec3 shadeWithLightmap(const Ray& ray, Rng& rng, bool& backFace) const; // probes and cubes
    void filterIndirect(const std::vector<LightmapRect>& rects, const std::vector<std::vector<uint8_t>>& valid,
                        const std::vector<std::vector<glm::vec3>>& indirect, const std::vector<std::vector<glm::vec3>>& normals);
    void dilate(const std::vector<LightmapRect>& rects, std::vector<std::vector<uint8_t>>& valid);

    BakeSettings m_settings;
    SkySettings m_skySettings;
    CubeFaces m_skyCube; // the sky at 64² per face: bilinear lookups instead of evaluating clouds per ray
    glm::vec3 m_sunDirection;
    glm::vec3 m_sunColor;
    glm::vec3 m_ambientUp, m_ambientDown; // E/π from the sky, for surfaces without a lightmap
    std::vector<BakeTexture> m_textures;
    std::vector<BakeMaterial> m_materials;
    std::vector<BakeMesh> m_meshes;
    std::vector<BakeLight> m_lights;
    std::vector<Triangle> m_triangles;
    RayScene m_scene;
    std::vector<LightmapRect> m_rects;
    uint32_t m_pageSize = 0;
    std::vector<HdrImage> m_light;
    std::vector<std::vector<float>> m_shadow;
    uint32_t m_bakedTexels = 0;
    uint32_t m_invalidTexels = 0;
};

} // namespace tools
