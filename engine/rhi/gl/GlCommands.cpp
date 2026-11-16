#include "rhi/gl/GlDevice.h"

#include "core/Log.h"
#include "rhi/CommandStream.h"

#include <shaderformat/BindingModel.h>

#include <algorithm>

namespace rhi::gl {
namespace {

GLenum toGlBlendFactor(BlendFactor f) {
    switch (f) {
    case BlendFactor::Zero: return GL_ZERO;
    case BlendFactor::One: return GL_ONE;
    case BlendFactor::SrcAlpha: return GL_SRC_ALPHA;
    case BlendFactor::OneMinusSrcAlpha: return GL_ONE_MINUS_SRC_ALPHA;
    case BlendFactor::DstAlpha: return GL_DST_ALPHA;
    case BlendFactor::OneMinusDstAlpha: return GL_ONE_MINUS_DST_ALPHA;
    case BlendFactor::SrcColor: return GL_SRC_COLOR;
    case BlendFactor::DstColor: return GL_DST_COLOR;
    }
    return GL_ONE;
}

GLenum toGlBlendOp(BlendOp op) {
    switch (op) {
    case BlendOp::Add: return GL_FUNC_ADD;
    case BlendOp::Subtract: return GL_FUNC_SUBTRACT;
    case BlendOp::ReverseSubtract: return GL_FUNC_REVERSE_SUBTRACT;
    case BlendOp::Min: return GL_MIN;
    case BlendOp::Max: return GL_MAX;
    }
    return GL_FUNC_ADD;
}

GLenum toGlCompareOp(CompareOp op) {
    constexpr GLenum table[] = {GL_NEVER, GL_LESS, GL_EQUAL, GL_LEQUAL, GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS};
    return table[uint32_t(op)];
}

} // namespace

void GlDevice::setCap(GLenum cap, int& cached, bool enable) {
    if (cached == int(enable)) return;
    if (enable) glEnable(cap);
    else glDisable(cap);
    cached = int(enable);
}

void GlDevice::bindUniformRange(uint8_t point, GLuint buffer, uint32_t offset, uint32_t size) {
    if (m_state.ubo[point] == buffer && m_state.uboOffset[point] == offset && m_state.uboSize[point] == size) return;
    glBindBufferRange(GL_UNIFORM_BUFFER, point, buffer, GLintptr(offset), GLsizeiptr(size));
    m_state.ubo[point] = buffer;
    m_state.uboOffset[point] = offset;
    m_state.uboSize[point] = size;
}

namespace {

void attachTexture(GLenum attachment, const GlTexture& t, uint16_t mip, uint16_t layer) {
    if (t.target == GL_TEXTURE_2D_ARRAY) glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, attachment, t.id, mip, layer);
    else if (t.target == GL_TEXTURE_CUBE_MAP)
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, attachment, GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer, t.id, mip);
    else glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, attachment, t.target, t.id, mip);
}

} // namespace

GLuint GlDevice::framebufferFor(const FboKey& key, const GlTexture* const* colors, const GlTexture* depth) {
    for (const FboEntry& e : m_fbos)
        if (e.key == key) return e.fbo;
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    GLenum drawBuffers[kMaxColorAttachments];
    for (uint32_t i = 0; i < key.colorCount; ++i) {
        attachTexture(GL_COLOR_ATTACHMENT0 + i, *colors[i], key.mip[i], key.layer[i]);
        drawBuffers[i] = GL_COLOR_ATTACHMENT0 + i;
    }
    if (depth)
        attachTexture(depth->desc.format == Format::D24S8 ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT, *depth, 0,
                      key.depthLayer);
    if (key.colorCount) {
        glDrawBuffers(GLsizei(key.colorCount), drawBuffers);
    } else {
        // Depth only (shadow maps): GL 3.3 also wants the read buffer NONE, or the framebuffer is incomplete
        // (INCOMPLETE_READ_BUFFER; relaxed in GL 4.1). glReadBuffer sets it on the read binding. Callers rebind both.
        glDrawBuffer(GL_NONE);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        glReadBuffer(GL_NONE);
    }
    const GLenum status = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        FL_LOG_ERROR("GL framebuffer incomplete (0x%04x)", status);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
        return 0;
    }
    m_fbos.push_back({key, fbo});
    return fbo;
}

void GlDevice::endRenderPass() {
    // Multisample resolve: blits bypass the fragment pipeline except for the scissor test.
    if (m_pendingResolveCount) setCap(GL_SCISSOR_TEST, m_state.scissor, false);
    for (uint32_t i = 0; i < m_pendingResolveCount; ++i) {
        const PendingResolve& r = m_pendingResolves[i];
        FboKey key;
        key.colorCount = 1;
        key.color[0] = r.target->id;
        const GLuint dst = framebufferFor(key, &r.target, nullptr);
        if (!dst) continue;
        glBindFramebuffer(GL_READ_FRAMEBUFFER, r.sourceFbo);
        glReadBuffer(GL_COLOR_ATTACHMENT0 + r.attachment);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst);
        glBlitFramebuffer(0, 0, GLint(r.width), GLint(r.height), 0, 0, GLint(r.width), GLint(r.height), GL_COLOR_BUFFER_BIT,
                          GL_NEAREST);
    }
    m_pendingResolveCount = 0;
}

void GlDevice::beginRenderPass(const RenderPassDesc& desc) {
    m_pendingResolveCount = 0;
    if (!desc.toBackbuffer) {
        FboKey key;
        const GlTexture* colors[kMaxColorAttachments] = {};
        const GlTexture* depth = nullptr;
        uint32_t width = 0, height = 0;
        key.colorCount = desc.colorCount;
        for (uint32_t i = 0; i < desc.colorCount; ++i) {
            colors[i] = m_textures.get(desc.color[i].texture);
            if (!colors[i]) {
                FL_LOG_ERROR("Render pass %s: invalid color attachment %u", desc.debugName, i);
                return;
            }
            key.color[i] = colors[i]->id;
            key.mip[i] = desc.color[i].mip;
            key.layer[i] = desc.color[i].layer;
            width = std::max(1u, colors[i]->desc.width >> desc.color[i].mip);
            height = std::max(1u, colors[i]->desc.height >> desc.color[i].mip);
        }
        if (desc.useDepth) {
            depth = m_textures.get(desc.depth.texture);
            if (!depth) {
                FL_LOG_ERROR("Render pass %s: invalid depth attachment", desc.debugName);
                return;
            }
            key.depth = depth->id;
            key.depthLayer = desc.depth.layer;
            width = depth->desc.width;
            height = depth->desc.height;
        }
        const GLuint fbo = framebufferFor(key, colors, depth);
        if (!fbo) return;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        // sRGB targets (the LDR scene, ldrColor; M8 plan D8) encode writes and blend in linear space only while
        // GL_FRAMEBUFFER_SRGB is on; it stays off for every other target and for the default framebuffer (§4.3).
        bool srgb = false;
        for (uint32_t i = 0; i < desc.colorCount; ++i)
            srgb = srgb || colors[i]->desc.format == Format::RGBA8Srgb || colors[i]->desc.format == Format::BGRA8Srgb;
        setCap(GL_FRAMEBUFFER_SRGB, m_state.framebufferSrgb, srgb);
        glViewport(0, 0, GLsizei(width), GLsizei(height));
        setCap(GL_SCISSOR_TEST, m_state.scissor, false); // scissor also clips clears
        for (uint32_t i = 0; i < desc.colorCount; ++i) {
            if (desc.color[i].load == LoadOp::Clear) {
                if (m_state.colorMask != 0xF) {
                    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                    m_state.colorMask = 0xF;
                }
                glClearBufferfv(GL_COLOR, GLint(i), desc.color[i].clear);
            }
            if (const GlTexture* r = m_textures.get(desc.color[i].resolveTo)) {
                if (colors[i]->desc.samples > 1 && r->desc.samples == 1)
                    m_pendingResolves[m_pendingResolveCount++] = {fbo, i, r, width, height};
                else
                    FL_LOG_ERROR("Render pass %s: color %u resolve needs a multisampled source and a single-sample target",
                                 desc.debugName, i);
            }
        }
        if (depth && desc.depth.load == LoadOp::Clear) {
            if (m_state.depthWrite != 1) {
                glDepthMask(GL_TRUE);
                m_state.depthWrite = 1;
            }
            if (depth->desc.format == Format::D24S8) glClearBufferfi(GL_DEPTH_STENCIL, 0, desc.depth.clearDepth, 0);
            else glClearBufferfv(GL_DEPTH, 0, &desc.depth.clearDepth);
        }
        return;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    setCap(GL_FRAMEBUFFER_SRGB, m_state.framebufferSrgb, false); // shaders encode for the default framebuffer
    glViewport(0, 0, GLsizei(m_backbuffer.width), GLsizei(m_backbuffer.height));
    setCap(GL_SCISSOR_TEST, m_state.scissor, false); // scissor also clips glClear

    GLbitfield mask = 0;
    if (desc.colorCount > 0 && desc.color[0].load == LoadOp::Clear) {
        const float* c = desc.color[0].clear;
        glClearColor(c[0], c[1], c[2], c[3]);
        if (m_state.colorMask != 0xF) {
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            m_state.colorMask = 0xF;
        }
        mask |= GL_COLOR_BUFFER_BIT;
    }
    if (desc.useDepth && desc.depth.load == LoadOp::Clear) {
        glClearDepth(desc.depth.clearDepth);
        if (m_state.depthWrite != 1) {
            glDepthMask(GL_TRUE);
            m_state.depthWrite = 1;
        }
        mask |= GL_DEPTH_BUFFER_BIT;
    }
    if (mask) glClear(mask);
}

void GlDevice::applyPipelineState(const GlPipeline& p) {
    const PipelineDesc& d = p.desc;
    if (m_state.program != p.program) {
        glUseProgram(p.program);
        m_state.program = p.program;
    }
    // Front face is CCW on both backends because the Y flip lives in the projection (ARCHITECTURE.md §4.2);
    // mirrored draws ask for CW. Set even without culling: gl_FrontFacing follows it (double-sided materials light
    // back faces from their own side), and Vulkan pipelines always carry it.
    setCap(GL_CULL_FACE, m_state.cull, d.raster.cull != CullMode::None);
    const GLenum frontFace = d.raster.frontFaceClockwise ? GL_CW : GL_CCW;
    if (m_state.frontFace != frontFace) {
        glFrontFace(frontFace);
        m_state.frontFace = frontFace;
    }
    const GLenum cullFace = d.raster.cull == CullMode::Front ? GL_FRONT : GL_BACK;
    if (d.raster.cull != CullMode::None && m_state.cullFace != cullFace) {
        glCullFace(cullFace);
        m_state.cullFace = cullFace;
    }
    setCap(GL_DEPTH_TEST, m_state.depthTest, d.depth.test);
    if (d.depth.test && m_state.depthFunc != toGlCompareOp(d.depth.compare)) {
        m_state.depthFunc = toGlCompareOp(d.depth.compare);
        glDepthFunc(m_state.depthFunc);
    }
    if (m_state.depthWrite != int(d.depth.write)) {
        glDepthMask(d.depth.write ? GL_TRUE : GL_FALSE);
        m_state.depthWrite = int(d.depth.write);
    }
    const BlendState& b = d.blend[0];
    setCap(GL_BLEND, m_state.blend, b.enable);
    if (b.enable) {
        const GLenum s = toGlBlendFactor(b.srcColor), dst = toGlBlendFactor(b.dstColor);
        const GLenum sa = toGlBlendFactor(b.srcAlpha), da = toGlBlendFactor(b.dstAlpha);
        if (s != m_state.blendSrc || dst != m_state.blendDst || sa != m_state.blendSrcA || da != m_state.blendDstA) {
            glBlendFuncSeparate(s, dst, sa, da);
            m_state.blendSrc = s, m_state.blendDst = dst, m_state.blendSrcA = sa, m_state.blendDstA = da;
        }
        const GLenum eq = toGlBlendOp(b.colorOp), eqa = toGlBlendOp(b.alphaOp);
        if (eq != m_state.blendEq || eqa != m_state.blendEqA) {
            glBlendEquationSeparate(eq, eqa);
            m_state.blendEq = eq, m_state.blendEqA = eqa;
        }
    }
    if (m_state.colorMask != b.writeMask) {
        glColorMask(b.writeMask & 1, (b.writeMask >> 1) & 1, (b.writeMask >> 2) & 1, (b.writeMask >> 3) & 1);
        m_state.colorMask = b.writeMask;
    }
    setCap(GL_SAMPLE_ALPHA_TO_COVERAGE, m_state.alphaToCoverage, d.raster.alphaToCoverage);
    const bool bias = d.raster.depthBiasConstant != 0.0f || d.raster.depthBiasSlope != 0.0f;
    setCap(GL_POLYGON_OFFSET_FILL, m_state.polygonOffset, bias);
    // GL applies polygon offset per rasterization mode: lines (wireframe pipelines) need their own switch (Vulkan biases
    // every mode).
    setCap(GL_POLYGON_OFFSET_LINE, m_state.polygonOffsetLine, bias && d.raster.wireframe);
    if (bias) glPolygonOffset(d.raster.depthBiasSlope, d.raster.depthBiasConstant);
    if (m_caps.wireframe && m_state.wireframe != int(d.raster.wireframe)) {
        glPolygonMode(GL_FRONT_AND_BACK, d.raster.wireframe ? GL_LINE : GL_FILL);
        m_state.wireframe = int(d.raster.wireframe);
    }
    m_vertexDirty = true;
}

// GL 3.3 has no separate vertex-binding state, so attribute pointers are re-specified when the pipeline
// or the bound streams change (ARCHITECTURE.md §9, P11). Absolute indices mean no base vertex is needed.
void GlDevice::applyVertexInput() {
    if (!m_vertexDirty || !m_pipeline) return;
    const VertexLayout& layout = m_pipeline->desc.vertexLayout;
    uint32_t enabled = 0;
    for (uint32_t i = 0; i < layout.attributeCount; ++i) {
        const VertexAttribute& a = layout.attributes[i];
        const BufferSlice& stream = m_streams[a.stream];
        const GlBuffer* buffer = m_buffers.get(stream.buffer);
        if (!buffer) continue;
        const GlVertexFormat f = glVertexFormat(a.format);
        const GLsizei stride = layout.streams[a.stream].stride;
        const void* pointer = reinterpret_cast<const void*>(uintptr_t(stream.offset + a.offset));
        glBindBuffer(GL_ARRAY_BUFFER, buffer->id);
        if (f.integer) glVertexAttribIPointer(a.location, f.components, f.type, stride, pointer);
        else glVertexAttribPointer(a.location, f.components, f.type, f.normalized, stride, pointer);
        glVertexAttribDivisor(a.location, layout.streams[a.stream].perInstance ? 1 : 0);
        enabled |= 1u << a.location;
    }
    const uint32_t changed = enabled ^ m_state.enabledAttributes;
    for (uint32_t loc = 0; loc < 16; ++loc) {
        if (!(changed & (1u << loc))) continue;
        if (enabled & (1u << loc)) glEnableVertexAttribArray(loc);
        else glDisableVertexAttribArray(loc);
    }
    m_state.enabledAttributes = enabled;
    m_vertexDirty = false;
}

void GlDevice::replay(const CommandList& list, uint32_t& perDrawIndex) {
    using namespace cmd;
    Reader reader(list.data());
    const GlBuffer* uniformRing = m_buffers.get(m_ringBuffers[uint32_t(TransientKind::Uniform)]);

    for (const Header* h = reader.next(); h; h = reader.next()) {
        switch (h->op) {
        case Op::BeginRenderPass: beginRenderPass(Reader::as<BeginRenderPass>(h)->desc); break;
        case Op::EndRenderPass: endRenderPass(); break;
        case Op::SetViewport: {
            // Attachment texel rows on both backends; no conversion here (ARCHITECTURE.md §4.2, C6).
            const Rect& r = Reader::as<SetViewport>(h)->rect;
            glViewport(r.x, r.y, GLsizei(r.width), GLsizei(r.height));
            break;
        }
        case Op::SetScissor: {
            const Rect& r = Reader::as<SetScissor>(h)->rect;
            setCap(GL_SCISSOR_TEST, m_state.scissor, true);
            glScissor(r.x, r.y, GLsizei(r.width), GLsizei(r.height));
            break;
        }
        case Op::BindPipeline: {
            const GlPipeline* p = m_pipelines.get(Reader::as<BindPipeline>(h)->pipeline);
            if (!p) {
                FL_LOG_ERROR("BindPipeline with an invalid handle");
                m_pipeline = nullptr;
                break;
            }
            m_pipeline = p;
            m_skipDraws = false;
            applyPipelineState(*p);
            break;
        }
        case Op::BindResourceSet: {
            const auto* c = Reader::as<BindResourceSet>(h);
            const GlResourceSet* s = m_resourceSets.get(c->resourceSet);
            if (!s) break;
            uint32_t dynamicIndex = 0;
            for (uint32_t i = 0; i < s->uboCount; ++i) {
                const GlResourceSet::Ubo& u = s->ubos[i];
                const uint32_t extra = u.dynamic && dynamicIndex < c->dynamicCount ? c->dynamicOffsets[dynamicIndex++] : 0;
                bindUniformRange(u.point, u.buffer, u.offset + extra, u.size);
            }
            for (uint32_t i = 0; i < s->textureCount; ++i) {
                const GlResourceSet::Tex& t = s->textures[i];
                if (m_state.texture[t.unit] != t.texture) {
                    glActiveTexture(GL_TEXTURE0 + t.unit);
                    glBindTexture(t.target, t.texture);
                    m_state.texture[t.unit] = t.texture;
                }
                if (m_state.sampler[t.unit] != t.sampler) {
                    glBindSampler(t.unit, t.sampler);
                    m_state.sampler[t.unit] = t.sampler;
                }
            }
            break;
        }
        case Op::SetPerDraw: {
            // A per-draw block that didn't fit into the uniform ring must not draw with someone else's data. Blocks before
            // m_perDrawHead and from m_perDrawTailStart on have entries, in order; the ones between were dropped (submit).
            const uint32_t i = perDrawIndex++;
            const uint32_t slot = i < m_perDrawHead ? i : i >= m_perDrawTailStart ? m_perDrawHead + (i - m_perDrawTailStart) : kNoPerDraw;
            const uint32_t offset = slot < m_perDrawOffsets.size() ? m_perDrawOffsets[slot] : kNoPerDraw;
            m_skipDraws = offset == kNoPerDraw;
            if (!m_skipDraws) bindUniformRange(shaderformat::kPerDrawUboPoint, uniformRing->id, offset, shaderformat::kMaxPerDrawBytes);
            break;
        }
        case Op::BindVertexStreams: {
            const auto* c = Reader::as<BindVertexStreams>(h);
            for (uint32_t i = 0; i < c->count; ++i) m_streams[c->first + i] = c->streams[i];
            m_vertexDirty = true;
            break;
        }
        case Op::BindIndexBuffer: {
            const auto* c = Reader::as<BindIndexBuffer>(h);
            m_indexBuffer = c->buffer;
            m_indexType = c->type;
            if (const GlBuffer* b = m_buffers.get(c->buffer.buffer)) glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b->id);
            break;
        }
        case Op::Draw: {
            if (!m_pipeline || m_skipDraws) break;
            const auto* c = Reader::as<Draw>(h);
            applyVertexInput();
            const GLenum mode = m_pipeline->desc.topology == Topology::Lines ? GL_LINES : GL_TRIANGLES;
            if (c->instanceCount == 1) glDrawArrays(mode, GLint(c->firstVertex), GLsizei(c->vertexCount));
            else glDrawArraysInstanced(mode, GLint(c->firstVertex), GLsizei(c->vertexCount), GLsizei(c->instanceCount));
            break;
        }
        case Op::DrawIndexed: {
            if (!m_pipeline || m_skipDraws) break;
            const auto* c = Reader::as<DrawIndexed>(h);
            applyVertexInput();
            const GLenum mode = m_pipeline->desc.topology == Topology::Lines ? GL_LINES : GL_TRIANGLES;
            const bool u16 = m_indexType == IndexType::U16;
            const GLenum type = u16 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT;
            const void* offset = reinterpret_cast<const void*>(uintptr_t(m_indexBuffer.offset + c->firstIndex * (u16 ? 2u : 4u)));
            if (c->instanceCount == 1) glDrawElements(mode, GLsizei(c->indexCount), type, offset);
            else glDrawElementsInstanced(mode, GLsizei(c->indexCount), type, offset, GLsizei(c->instanceCount));
            break;
        }
        case Op::WriteTimestamp: {
            const auto* c = Reader::as<WriteTimestamp>(h);
            GlQueryPool* q = m_queryPools.get(c->pool);
            if (!q || !m_timerQueries || c->index >= q->count) break;
            glQueryCounter(q->ids[m_slot][c->index], GL_TIMESTAMP);
            q->written[m_slot][c->index] = 1;
            break;
        }
        case Op::PushDebugGroup:
            if (m_khrDebug) glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, Reader::as<PushDebugGroup>(h)->name);
            break;
        case Op::PopDebugGroup:
            if (m_khrDebug) glPopDebugGroup();
            break;
        }
    }
}

} // namespace rhi::gl
