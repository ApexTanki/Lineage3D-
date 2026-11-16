#include "renderer/debug/ImGuiRenderer.h"

#include "core/Log.h"
#include "core/Math.h"
#include "renderer/core/ClipSpace.h"
#include "renderer/core/ShaderLibrary.h"

#include <imgui.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace renderer {
namespace {

struct TextureEntry {
    rhi::TextureHandle texture;
    rhi::ResourceSetHandle set;
};

struct ImGuiPerDraw {
    glm::mat4 projection;
    glm::vec4 params;
};

} // namespace

core::Result<void> ImGuiRenderer::init(rhi::Device& device, ShaderLibrary& shaders, const rhi::AttachmentLayout& target) {
    m_device = &device;
    auto program = shaders.program("imgui", 0);
    if (!program) return program.error();

    rhi::ResourceSetLayoutDesc layout;
    layout.set = 2;
    layout.count = 1;
    layout.bindings[0] = {1, rhi::BindingKind::Texture, rhi::ShaderStageMask::Fragment, rhi::TextureType::Tex2D, false};
    m_setLayout = device.createResourceSetLayout(layout);

    rhi::SamplerDesc sampler;
    sampler.filter = rhi::Filter::Linear;
    sampler.mip = rhi::MipMode::Nearest;
    sampler.address = rhi::AddressMode::ClampToEdge;
    m_sampler = device.createSampler(sampler);

    rhi::PipelineDesc pd;
    pd.debugName = "ImGui";
    pd.vertex = program.value().vertex;
    pd.fragment = program.value().fragment;
    pd.vertexLayout.streamCount = 1;
    pd.vertexLayout.streams[0] = {uint16_t(sizeof(ImDrawVert)), false};
    pd.vertexLayout.attributeCount = 3;
    pd.vertexLayout.attributes[0] = {0, 0, rhi::Format::RG32F, uint16_t(offsetof(ImDrawVert, pos))};
    pd.vertexLayout.attributes[1] = {3, 0, rhi::Format::RG32F, uint16_t(offsetof(ImDrawVert, uv))};
    pd.vertexLayout.attributes[2] = {7, 0, rhi::Format::RGBA8Unorm, uint16_t(offsetof(ImDrawVert, col))};
    pd.raster.cull = rhi::CullMode::None; // UI triangles have mixed winding
    pd.blend[0] = {true,
                   rhi::BlendFactor::SrcAlpha, rhi::BlendFactor::OneMinusSrcAlpha, rhi::BlendOp::Add,
                   rhi::BlendFactor::One, rhi::BlendFactor::OneMinusSrcAlpha, rhi::BlendOp::Add, 0xF};
    pd.attachments = target;
    pd.setLayouts[2] = m_setLayout;
    pd.perDrawBytes = sizeof(ImGuiPerDraw);
    m_pipeline = device.createPipeline(pd);
    if (!m_setLayout || !m_sampler || !m_pipeline) return core::Error::InitFailed;

    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = "flashlineage3d_rhi";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    // A 4096² RGBA atlas (64 MiB) is far beyond any UI need; larger uploads would take the slow immediate path.
    ImGui::GetPlatformIO().Renderer_TextureMaxWidth = ImGui::GetPlatformIO().Renderer_TextureMaxHeight =
        int(std::min<uint32_t>(device.caps().maxTextureSize, 4096));
    return {};
}

void ImGuiRenderer::shutdown() {
    if (!m_device) return;
    for (ImTextureData* tex : ImGui::GetPlatformIO().Textures)
        if (tex->BackendUserData) {
            tex->SetStatus(ImTextureStatus_WantDestroy);
            tex->UnusedFrames = 1;
            updateTexture(tex);
        }
    m_device->destroy(m_pipeline);
    m_device->destroy(m_sampler);
    m_device->destroy(m_setLayout);
    ImGui::GetIO().BackendFlags &= ~ImGuiBackendFlags_RendererHasTextures;
    ImGui::GetIO().BackendRendererName = nullptr;
    m_device = nullptr;
}

void ImGuiRenderer::updateTexture(ImTextureData* tex) {
    if (tex->Status == ImTextureStatus_WantCreate) {
        rhi::TextureDesc desc;
        desc.debugName = "ImGuiTexture";
        desc.format = rhi::Format::RGBA8Unorm;
        desc.width = uint32_t(tex->Width);
        desc.height = uint32_t(tex->Height);
        auto* entry = new TextureEntry();
        entry->texture = m_device->createTexture(desc);
        rhi::TextureUpload upload;
        upload.width = desc.width;
        upload.height = desc.height;
        upload.data = {reinterpret_cast<const std::byte*>(tex->GetPixels()), size_t(tex->GetSizeInBytes())};
        if (!entry->texture || !m_device->uploadTexture(entry->texture, upload).ok()) {
            FL_LOG_ERROR("ImGui texture creation failed");
            m_device->destroy(entry->texture);
            delete entry;
            return;
        }
        const rhi::ResourceWrite write{1, {}, entry->texture, m_sampler};
        rhi::ResourceSetDesc setDesc;
        setDesc.debugName = "ImGuiTextureSet";
        setDesc.layout = m_setLayout;
        setDesc.writes = {&write, 1};
        entry->set = m_device->createResourceSet(setDesc);
        tex->SetTexID(ImTextureID(entry->set.bits));
        tex->BackendUserData = entry;
        tex->SetStatus(ImTextureStatus_OK);
    } else if (tex->Status == ImTextureStatus_WantUpdates) {
        auto* entry = static_cast<TextureEntry*>(tex->BackendUserData);
        std::vector<std::byte> rows; // rare (atlas growth), so a packed copy beats a stride-aware upload path
        for (const ImTextureRect& r : tex->Updates) {
            rows.resize(size_t(r.w) * r.h * 4);
            for (int y = 0; y < r.h; ++y)
                std::memcpy(rows.data() + size_t(y) * r.w * 4, tex->GetPixelsAt(r.x, r.y + y), size_t(r.w) * 4);
            rhi::TextureUpload upload;
            upload.x = r.x;
            upload.y = r.y;
            upload.width = r.w;
            upload.height = r.h;
            upload.data = rows;
            if (entry)
                if (auto result = m_device->uploadTexture(entry->texture, upload); !result)
                    FL_LOG_ERROR("ImGui texture update failed: %s", core::errorName(result.error()));
        }
        tex->SetStatus(ImTextureStatus_OK);
    } else if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0) {
        // Device destruction is deferred until in-flight frames retire, so this is safe immediately.
        if (auto* entry = static_cast<TextureEntry*>(tex->BackendUserData)) {
            m_device->destroy(entry->set);
            m_device->destroy(entry->texture);
            delete entry;
        }
        tex->SetTexID(ImTextureID_Invalid);
        tex->BackendUserData = nullptr;
        tex->SetStatus(ImTextureStatus_Destroyed);
    }
}

void ImGuiRenderer::render(rhi::CommandList& cmd, const ClipSpace& clipSpace, bool targetEncodesSrgb) {
    ImDrawData* dd = ImGui::GetDrawData();
    if (!dd || !m_device) return;
    if (dd->Textures)
        for (ImTextureData* tex : *dd->Textures)
            if (tex->Status != ImTextureStatus_OK) updateTexture(tex);

    const float fbWidth = dd->DisplaySize.x * dd->FramebufferScale.x;
    const float fbHeight = dd->DisplaySize.y * dd->FramebufferScale.y;
    if (fbWidth <= 0.0f || fbHeight <= 0.0f || dd->TotalVtxCount == 0) return;

    // ImGui works in top-left pixel coordinates; the canonical ortho maps the top edge to NDC +1 and
    // ClipSpace makes that right for the backend.
    const float left = dd->DisplayPos.x, right = left + dd->DisplaySize.x;
    const float top = dd->DisplayPos.y, bottom = top + dd->DisplaySize.y;
    const ImGuiPerDraw perDraw{clipSpace.orthographic(left, right, bottom, top, -1.0f, 1.0f),
                               glm::vec4(targetEncodesSrgb ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f)};
    auto bindState = [&] {
        cmd.bindPipeline(m_pipeline);
        cmd.setPerDraw(&perDraw, sizeof(perDraw));
    };
    bindState();

    rhi::TransientRing& vertices = m_device->transient(rhi::TransientKind::Vertex);
    rhi::TransientRing& indices = m_device->transient(rhi::TransientKind::Index);
    const ImVec2 clipOffset = dd->DisplayPos;
    const ImVec2 clipScale = dd->FramebufferScale;

    for (const ImDrawList* list : dd->CmdLists) {
        const auto v = vertices.allocate(uint32_t(list->VtxBuffer.Size * sizeof(ImDrawVert)));
        const auto i = indices.allocate(uint32_t(list->IdxBuffer.Size * sizeof(ImDrawIdx)));
        if (!v.cpu || !i.cpu) {
            if (!m_warnedRingFull) FL_LOG_ERROR("ImGui geometry doesn't fit the transient rings; UI truncated");
            m_warnedRingFull = true;
            break;
        }
        std::memcpy(v.cpu, list->VtxBuffer.Data, list->VtxBuffer.Size * sizeof(ImDrawVert));
        std::memcpy(i.cpu, list->IdxBuffer.Data, list->IdxBuffer.Size * sizeof(ImDrawIdx));
        cmd.bindVertexStreams(0, {&v.gpu, 1});
        cmd.bindIndexBuffer(i.gpu, sizeof(ImDrawIdx) == 2 ? rhi::IndexType::U16 : rhi::IndexType::U32);

        for (const ImDrawCmd& c : list->CmdBuffer) {
            if (c.UserCallback) {
                if (c.UserCallback == ImDrawCallback_ResetRenderState) bindState();
                else c.UserCallback(list, &c);
                continue;
            }
            // A texture whose creation failed has no ID; skip it rather than tripping ImGui's assert.
            const ImTextureID texId = c.TexRef._TexData ? c.TexRef._TexData->TexID : c.TexRef._TexID;
            if (texId == ImTextureID_Invalid) continue;
            const float x0 = std::max((c.ClipRect.x - clipOffset.x) * clipScale.x, 0.0f);
            const float y0 = std::max((c.ClipRect.y - clipOffset.y) * clipScale.y, 0.0f);
            const float x1 = std::min((c.ClipRect.z - clipOffset.x) * clipScale.x, fbWidth);
            const float y1 = std::min((c.ClipRect.w - clipOffset.y) * clipScale.y, fbHeight);
            if (x1 <= x0 || y1 <= y0) continue;
            const rhi::Rect topLeft{int32_t(x0), int32_t(y0), uint32_t(x1 - x0), uint32_t(y1 - y0)};
            cmd.setScissor(clipSpace.fromTopLeft(topLeft, uint32_t(fbHeight)));
            cmd.bindResourceSet(2, rhi::ResourceSetHandle{uint32_t(texId)});
            cmd.drawIndexed(c.ElemCount, 1, c.IdxOffset);
        }
    }
}

} // namespace renderer
