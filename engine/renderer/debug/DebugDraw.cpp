#include "renderer/debug/DebugDraw.h"

#include "core/Log.h"
#include "core/Profile.h"
#include "renderer/core/RenderQueue.h"
#include "renderer/core/Scene.h"
#include "renderer/core/ShaderLibrary.h"
#include "renderer/core/ShaderTypes.h"
#include "renderer/materials/MaterialSystem.h"
#include "renderer/resources/ResourceManager.h"
#include "renderer/visibility/Lod.h"

#include <algorithm>
#include <cstring>

namespace renderer {

core::Result<void> DebugDraw::init(rhi::Device& device, ShaderLibrary& shaders, rhi::ResourceSetLayoutHandle paletteLayout,
                                    rhi::ResourceSetHandle paletteSet) {
    m_device = &device;
    m_shaders = &shaders;
    m_paletteLayout = paletteLayout;
    m_paletteSet = paletteSet;
    rhi::ResourceSetLayoutDesc ld;
    ld.set = 1;
    ld.count = 1;
    ld.bindings[0] = {0, rhi::BindingKind::UniformBufferDynamic, rhi::ShaderStageMask::All};
    m_passLayout = device.createResourceSetLayout(ld);
    if (!m_passLayout) return core::Error::InitFailed;
    const rhi::ResourceWrite write{0, rhi::BufferSlice{device.transient(rhi::TransientKind::Uniform).buffer(), 0,
                                                       uint32_t(sizeof(DebugPassUniforms))},
                                   {}, {}};
    rhi::ResourceSetDesc sd;
    sd.debugName = "debug pass";
    sd.layout = m_passLayout;
    sd.writes = {&write, 1};
    m_passSet = device.createResourceSet(sd);
    if (!m_passSet) return core::Error::InitFailed;
    return {};
}

void DebugDraw::shutdown() {
    if (!m_device) return;
    for (auto& mode : m_pipelines)
        for (auto& variant : mode)
            for (rhi::PipelineHandle& p : variant) m_device->destroy(p);
    m_device->destroy(m_passSet);
    m_device->destroy(m_passLayout);
    m_device = nullptr;
}

bool DebugDraw::setTarget(const rhi::AttachmentLayout& target) {
    auto object = m_shaders->axis("debug", "OBJECT");
    if (!object) return false;
    bool ok = true;
    for (uint32_t mode = 0; mode < 2; ++mode)
        for (uint32_t variant = 0; variant < 3; ++variant) // the OBJECT value: dynamic, instanced, skinned
            for (uint32_t cull = 0; cull < 3; ++cull) {
                auto program = m_shaders->program("debug", uint64_t(variant) << object.value().bitOffset);
                if (!program) return false;
                rhi::PipelineDesc pd;
                pd.debugName = mode == 0 ? "debug wireframe" : "debug overdraw";
                pd.vertex = program.value().vertex;
                pd.fragment = program.value().fragment;
                if (!fixedVertexLayout(*program.value().reflection, pd.vertexLayout)) return false;
                pd.raster.cull = cull == 0 ? rhi::CullMode::None : rhi::CullMode::Back;
                pd.raster.frontFaceClockwise = cull == 2;
                if (mode == 0) {
                    // Lines over the lit frame: depth-tested, pulled toward the camera so they win against their own faces.
                    pd.raster.wireframe = true;
                    pd.raster.depthBiasConstant = -2.0f;
                    pd.raster.depthBiasSlope = -1.0f;
                    pd.depth = {true, false, rhi::CompareOp::LessOrEqual};
                } else {
                    // Every covering draw adds its constant: no depth test, additive.
                    pd.depth = {false, false, rhi::CompareOp::Always};
                    pd.blend[0] = {true, rhi::BlendFactor::One, rhi::BlendFactor::One, rhi::BlendOp::Add,
                                   rhi::BlendFactor::Zero, rhi::BlendFactor::One, rhi::BlendOp::Add, 0xF};
                }
                pd.attachments = target;
                pd.setLayouts[1] = m_passLayout;
                if (variant == 2) pd.setLayouts[3] = m_paletteLayout;
                pd.perDrawBytes = uint8_t(sizeof(InstanceRows));
                rhi::PipelineHandle& slot = m_pipelines[mode][variant][cull];
                m_device->destroy(slot);
                slot = m_device->createPipeline(pd);
                if (!slot) {
                    FL_LOG_ERROR("The %s pipeline could not be created", pd.debugName);
                    ok = false;
                }
            }
    if (!m_device->caps().wireframe) FL_LOG_WARN("This device has no wireframe fill mode: the wireframe view draws filled triangles");
    return ok;
}

void DebugDraw::record(rhi::CommandList& cmd, Mode mode, const RenderQueue& queue, const Scene& scene, const ResourceManager& resources,
                       const MaterialSystem& materials, rhi::BufferHandle vertexRing, const glm::mat4& viewProj) {
    FL_ZONE("DebugDraw::record");
    DebugPassUniforms pass;
    pass.viewProj = viewProj;
    pass.color = mode == Mode::Wireframe ? glm::vec4(0.05f, 0.85f, 0.25f, 1.0f) : glm::vec4(kOverdrawStep, 0.0f, 0.0f, 0.0f);
    const rhi::TransientRing::Allocation a = m_device->transient(rhi::TransientKind::Uniform).allocate(sizeof(pass));
    if (!a.cpu) return;
    std::memcpy(a.cpu, &pass, sizeof(pass)); // ring memory is write-only (ARCHITECTURE.md §4.8)
    const uint32_t offset = a.gpu.offset;
    const uint32_t m = mode == Mode::Wireframe ? 0u : 1u;
    rhi::PipelineHandle bound;
    bool setBound = false;
    uint32_t boundMesh = ~0u;
    bool skinBound = false; // stream 2 holds boundMesh's skin stream (instanced draws put instance data there)
    const GpuMesh* mesh = nullptr;
    const ModelAsset* model = nullptr;
    for (uint32_t i = 0; i < queue.size(); ++i) {
        const DrawItem& item = queue.sortedItem(i);
        if (item.flags == 0 || (item.flags & DrawItem::kParticles)) continue; // particles draw themselves (M7 plan D13)
        if (item.mesh != boundMesh) {
            const Scene::MeshRef& ref = scene.meshRefs()[item.mesh];
            const ModelAsset* candidate = resources.model(ref.model);
            if (!candidate || candidate->state != LoadState::Ready) continue;
            model = candidate;
            mesh = &candidate->meshes[ref.mesh];
        }
        // The material's cull mode: double-sided faces both ways, mirrored transforms turn front faces clockwise.
        const uint32_t instanced = (item.flags & DrawItem::kInstanced) ? 1u : 0u;
        const bool skinned = (item.flags & DrawItem::kSkinned) != 0;
        const uint32_t variant = skinned ? 2u : instanced;
        bool mirrored = false;
        if (item.flags & DrawItem::kDynamic) mirrored = scene.dynamics()[item.perDrawOrInstance].mirrored;
        else if (item.flags & DrawItem::kStaticRow) mirrored = scene.staticGroups()[scene.staticRowGroups()[item.perDrawOrInstance]].mirrored;
        else mirrored = scene.staticGroups()[item.objectIndex].mirrored; // instanced: objectIndex = the group
        const assets::MaterialFile* file = materials.file(MaterialHandle{item.material});
        const uint32_t cull = file && file->doubleSided ? 0u : mirrored ? 2u : 1u;
        const rhi::PipelineHandle pipeline = m_pipelines[m][variant][cull];
        if (!pipeline) continue;
        if (!(pipeline == bound)) {
            cmd.bindPipeline(pipeline);
            if (!setBound) cmd.bindResourceSet(1, m_passSet, {&offset, 1}); // every debug layout is the same
            setBound = true;
            bound = pipeline;
        }
        if (skinned) { // the renderable's bone palette for this frame
            const uint32_t palette = scene.dynamics()[item.perDrawOrInstance].paletteOffset;
            if (palette == Scene::kNoPalette) continue;
            cmd.bindResourceSet(3, m_paletteSet, {&palette, 1});
        }
        if (item.mesh != boundMesh || (skinned && !skinBound)) {
            const rhi::BufferSlice streams[3] = {mesh->positions, mesh->surface, mesh->skin};
            cmd.bindVertexStreams(0, {streams, size_t(skinned ? 3 : 2)});
            cmd.bindIndexBuffer(mesh->indices, mesh->indexType);
            skinBound = skinned;
            boundMesh = item.mesh;
        }
        const GpuSubmesh& sub = model->submeshes[mesh->firstSubmesh + item.submesh];
        const uint32_t level = std::min<uint32_t>(lodWordLevel(item.lod), mesh->lodCount - 1);
        if (instanced) {
            const rhi::BufferSlice data{vertexRing, item.perDrawOrInstance, item.instanceCount * uint32_t(sizeof(InstanceData))};
            cmd.bindVertexStreams(2, {&data, 1});
            skinBound = false;
            cmd.drawIndexed(sub.lodCount[level], item.instanceCount, sub.lodFirst[level]);
        } else {
            const InstanceRows& rows = (item.flags & DrawItem::kDynamic) ? scene.dynamics()[item.perDrawOrInstance].rows
                                                                          : scene.staticRows()[item.perDrawOrInstance];
            cmd.setPerDraw(&rows, uint8_t(sizeof(rows)));
            cmd.drawIndexed(sub.lodCount[level], 1, sub.lodFirst[level]);
        }
    }
}

} // namespace renderer
