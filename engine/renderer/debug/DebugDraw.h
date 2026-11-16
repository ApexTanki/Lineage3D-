#pragma once

#include "core/Math.h"
#include "core/Result.h"
#include "rhi/Device.h"

namespace renderer {

class MaterialSystem;
class RenderQueue;
class ResourceManager;
class Scene;
class ShaderLibrary;

// The wireframe and overdraw debug views (M5 plan D10): every draw of the sorted scene queue again, positions only,
// with the `debug` template. Wireframe: lines over the lit frame (depth-tested, pulled toward the camera). Overdraw: a
// constant added per covering draw into the cleared scene target, depth test off; the present pass turns the count
// into a heat ramp. Both draw each item at its own LOD (fading items show both levels).
class DebugDraw {
public:
    enum class Mode : uint8_t { Wireframe, Overdraw };
    static constexpr float kOverdrawStep = 1.0f / 32.0f; // added to red per draw; the present pass multiplies by 32

    // paletteLayout / paletteSet: set 3 of skinned draws (the bone palette; owned by Environment, M6 plan D9).
    core::Result<void> init(rhi::Device& device, ShaderLibrary& shaders, rhi::ResourceSetLayoutHandle paletteLayout, rhi::ResourceSetHandle paletteSet);
    void shutdown();
    // Recreates the pipelines for a scene attachment layout (format or MSAA change; never per frame).
    bool setTarget(const rhi::AttachmentLayout& target);
    // Inside the scene render pass. `viewProj`: the scene camera in backend clip space.
    // Faces are culled as their materials cull them (double-sided, mirrored), so overdraw counts the faces that shade.
    void record(rhi::CommandList& cmd, Mode mode, const RenderQueue& queue, const Scene& scene, const ResourceManager& resources,
                const MaterialSystem& materials, rhi::BufferHandle vertexRing, const glm::mat4& viewProj);

private:
    rhi::Device* m_device = nullptr;
    ShaderLibrary* m_shaders = nullptr;
    rhi::ResourceSetLayoutHandle m_passLayout; // set 1: DebugPassUniforms (dynamic offset)
    rhi::ResourceSetHandle m_passSet;
    rhi::ResourceSetLayoutHandle m_paletteLayout;
    rhi::ResourceSetHandle m_paletteSet;
    rhi::PipelineHandle m_pipelines[2][3][3]; // [mode][variant: dynamic, instanced, skinned][cull: none (double-sided), back, back mirrored]
};

} // namespace renderer
