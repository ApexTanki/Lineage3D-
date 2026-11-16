#include "rhi/gl/GlDevice.h"

#include "core/Log.h"
#include "core/Profile.h"
#include "platform/Window.h"
#include "rhi/CommandStream.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace rhi::gl {
namespace {

constexpr uint32_t kUniformRingBytes = 4u << 20;
constexpr uint32_t kVertexRingBytes = 8u << 20;
constexpr uint32_t kIndexRingBytes = 1u << 20;

// glad1's loader callback carries no user pointer, so the window is parked here during loading only.
const platform::Window* g_loadingWindow = nullptr;
void* loadProc(const char* name) {
    return g_loadingWindow->glGetProcAddress(name);
}

void GLAPIENTRY debugCallback(GLenum, GLenum type, GLuint, GLenum severity, GLsizei, const GLchar* message,
                              const void* user) {
    static_cast<GlDevice*>(const_cast<void*>(user))->onDebugMessage(type, severity, message);
}

} // namespace

core::Result<std::unique_ptr<Device>> createGlDevice(const DeviceDesc& desc) {
    auto device = std::make_unique<GlDevice>(desc);
    if (auto r = device->init(); !r) return r.error();
    return std::unique_ptr<Device>(std::move(device));
}

GlDevice::GlDevice(const DeviceDesc& desc) : m_desc(desc) {}

core::Result<void> GlDevice::init() {
    FL_ZONE("GlDevice::init");
    g_loadingWindow = m_desc.window;
    const int loaded = gladLoadGLLoader(loadProc);
    g_loadingWindow = nullptr;
    if (!loaded) {
        FL_LOG_ERROR("glad failed to load OpenGL entry points");
        return core::Error::InitFailed;
    }
    if (!GLAD_GL_VERSION_3_3) {
        FL_LOG_ERROR("The context is older than OpenGL 3.3");
        return core::Error::InitFailed;
    }

    // Never trust an extension flag without its function pointers.
    m_khrDebug = GLAD_GL_KHR_debug && glDebugMessageCallback && glObjectLabel && glPushDebugGroup && glPopDebugGroup;
    if (m_desc.validation && m_khrDebug) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(debugCallback, this);
    } else if (m_desc.validation) {
        FL_LOG_WARN("GL debug output requested but KHR_debug is unavailable");
    }

    // M9 plan D7: the window asks for a context that is lost on a driver reset (reset notification); whether it got one
    // decides if glGetGraphicsResetStatus can ever report a reset. Without it a reset shows only as a frame that doesn't
    // finish within 10 s.
    m_resetStatus = glGetGraphicsResetStatus ? glGetGraphicsResetStatus : GLAD_GL_ARB_robustness ? glGetGraphicsResetStatusARB : nullptr;
    GLint strategy = 0;
    if (m_resetStatus && (GLAD_GL_KHR_robustness || GLAD_GL_ARB_robustness)) glGetIntegerv(GL_RESET_NOTIFICATION_STRATEGY, &strategy);
    if (strategy != GL_LOSE_CONTEXT_ON_RESET) m_resetStatus = nullptr;
    FL_LOG_INFO("GL reset notification: %s", m_resetStatus ? "on (a driver reset ends the run as a device loss)"
                                                          : "off (a driver reset shows only as a frame that doesn't finish within 10 s)");

    if (!queryCaps()) return core::Error::InitFailed;
    if (m_desc.capsFloor) applyCapsFloor();
    initProgramCache();

    // GL 3.3 core can't draw without a VAO. One global VAO is re-pointed as streams change.
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS); // always on in Vulkan (ARCHITECTURE.md §9, P9)
    // Off while the default framebuffer is bound (see queryCaps); offscreen sRGB targets enable it (M1+).
    glDisable(GL_FRAMEBUFFER_SRGB);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (!createRings()) return core::Error::InitFailed;
    // One entry per per-draw block that fits the uniform ring (submit stops at the first that doesn't), so it never grows.
    m_perDrawOffsets.reserve(m_rings[uint32_t(TransientKind::Uniform)].capacityBytes() / shaderformat::kMaxPerDrawBytes);
    FL_LOG_INFO("GL device: %s | %s | backbuffer %s", m_caps.deviceName, m_caps.apiVersion,
                m_caps.srgbBackbuffer ? "sRGB" : "linear (shaders encode sRGB)");
    return {};
}

bool GlDevice::queryCaps() {
    Caps& c = m_caps;
    c.backend = Backend::GL33;
    c.clipSpace = {false, false};
    std::snprintf(c.deviceName, sizeof(c.deviceName), "%s", reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
    std::snprintf(c.apiVersion, sizeof(c.apiVersion), "%s", reinterpret_cast<const char*>(glGetString(GL_VERSION)));
    fillGlFormatSupport(c);

    GLint v = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &v);
    c.maxTextureSize = uint32_t(v);
    glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &v);
    c.uniformOffsetAlignment = uint32_t(v > 16 ? v : 16);
    glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &v);
    c.maxUniformBlockBytes = uint32_t(v);
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &v);
    c.maxFragmentTextureUnits = uint32_t(v);
    glGetIntegerv(GL_MAX_SAMPLES, &v);
    c.maxMsaaSamples = uint32_t(v > 0 ? v : 1);
    if (GLAD_GL_EXT_texture_filter_anisotropic || GLAD_GL_ARB_texture_filter_anisotropic) {
        GLfloat a = 1.0f;
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &a);
        c.maxAnisotropy = a;
    }
    // Timer queries, polygon mode and multisample textures are GL 3.2/3.3 core.
    m_timerQueries = glQueryCounter && glGetQueryObjectui64v;
    c.timestampQueries = m_timerQueries;
    c.wireframe = true;
    c.multisampleTextures = true;
    c.linearDepthCompare = true;
    c.programBinaryCache = GLAD_GL_ARB_get_program_binary != 0 && glProgramBinary && glGetProgramBinary && glProgramParameteri;
    c.debugLabels = m_khrDebug;
    c.depthFormat = Format::D24S8;

    // The GL default framebuffer is never treated as sRGB-encoding: drivers misreport it (NVIDIA answers
    // GL_LINEAR yet encodes when GL_FRAMEBUFFER_SRGB is on). Shaders encode instead (ARCHITECTURE.md §4.3, §9 P7).
    c.srgbBackbuffer = false;
    return true;
}

// --caps-floor: behave like a minimal OpenGL 3.3 core device: the GL 3.3 minimum limits, and every optional
// capability that has a fallback switched off, so the fallbacks run on this PC (ARCHITECTURE.md §2.1 A12).
// S3TC stays: every desktop driver exposes it.
void GlDevice::applyCapsFloor() {
    Caps& c = m_caps;
    c.bptc = false; // BC6H/BC7 need DX11-class hardware: exercises the RGBM and BC1/BC3 fallbacks
    for (Format f : {Format::BC6HUfloat, Format::BC7Unorm, Format::BC7Srgb}) c.formats[uint32_t(f)] = {};
    c.maxAnisotropy = 1.0f;
    c.programBinaryCache = false;
    c.linearDepthCompare = false; // exercises the 4-tap PCF fallback
    c.maxMsaaSamples = std::min(c.maxMsaaSamples, 4u);
    c.maxUniformBlockBytes = 16384;
    c.maxFragmentTextureUnits = 16;
    FL_LOG_INFO("--caps-floor: emulating a minimal OpenGL 3.3 core device");
}

bool GlDevice::createRings() {
    struct Spec { TransientKind kind; uint32_t bytes; BufferUsage usage; uint32_t align; const char* name; };
    const Spec specs[] = {
        {TransientKind::Uniform, kUniformRingBytes, BufferUsage::Uniform, m_caps.uniformOffsetAlignment, "TransientUniformRing"},
        {TransientKind::Vertex, kVertexRingBytes, BufferUsage::Vertex, 16, "TransientVertexRing"},
        {TransientKind::Index, kIndexRingBytes, BufferUsage::Index, 16, "TransientIndexRing"},
    };
    const bool fenced = m_desc.glStream == GlStreamStrategy::Fenced;
    for (const Spec& s : specs) {
        BufferDesc bd;
        bd.debugName = s.name;
        bd.size = fenced ? s.bytes * kFramesInFlight : s.bytes;
        bd.usage = s.usage;
        bd.memory = MemoryKind::Upload;
        const BufferHandle h = createBuffer(bd);
        if (!h) return false;
        GlBuffer* ring = m_buffers.get(h);
        ring->ownedByRing = true;
        m_memory.move(ring->category, GpuMemoryCategory::Buffers, ring->size); // transient, not mesh memory (M9 plan D6)
        ring->category = GpuMemoryCategory::Buffers;
        m_ringBuffers[uint32_t(s.kind)] = h;
        m_ringStaging[uint32_t(s.kind)].resize(s.bytes);
        m_rings[uint32_t(s.kind)].init(h, s.bytes, s.align);
    }
    FL_LOG_INFO("GL uniform streaming: %s", fenced ? "fenced glBufferSubData" : "orphan + glBufferSubData");
    return true;
}

void GlDevice::onDebugMessage(GLenum type, GLenum severity, const char* message) {
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
    // The screenshot readback stalls on purpose; its pixel-path performance note is expected, not a defect.
    if (m_inReadback && type == GL_DEBUG_TYPE_PERFORMANCE) return;
    // NVIDIA compiles a shader again when a draw's GL state differs from what it compiled for (P30): a compile hitch,
    // which the demo counts like a pipeline creation (M9 plan D3, D5).
    if (type == GL_DEBUG_TYPE_PERFORMANCE && std::strstr(message, "recompiled")) ++m_stats.driverRecompiles;
    if (type == GL_DEBUG_TYPE_ERROR) {
        ++m_stats.validationErrors;
        FL_LOG_ERROR("[gl] %s", message);
    } else if (severity == GL_DEBUG_SEVERITY_HIGH || severity == GL_DEBUG_SEVERITY_MEDIUM) {
        ++m_stats.validationWarnings;
        // Debug output is synchronous: during replay, the pipeline names the draw that caused it.
        FL_LOG_WARN("[gl] %s (pipeline: %s)", message, m_pipeline && m_pipeline->desc.debugName ? m_pipeline->desc.debugName : "none");
    } else {
        FL_LOG_DEBUG("[gl] %s", message);
    }
}

void GlDevice::label(GLenum identifier, GLuint name, const char* text) {
    if (!m_khrDebug || !text || !*text) return;
    glObjectLabel(identifier, name, -1, text);
}

GlDevice::~GlDevice() {
    if (!m_lost) glFinish(); // M9 plan D7: nothing waits on a lost (possibly hung) GPU
    runDeferred(true);
    for (const FboEntry& e : m_fbos) glDeleteFramebuffers(1, &e.fbo);
    for (GLsync fence : m_fences)
        if (fence) glDeleteSync(fence);
    m_pipelines.forEach([](auto, GlPipeline& p) { glDeleteProgram(p.program); });
    m_shaders.forEach([](auto, GlShader& s) { glDeleteShader(s.id); });
    m_textures.forEach([](auto, GlTexture& t) { glDeleteTextures(1, &t.id); });
    m_samplers.forEach([](auto, GlSampler& s) { glDeleteSamplers(1, &s.id); });
    m_buffers.forEach([](auto, GlBuffer& b) { glDeleteBuffers(1, &b.id); });
    m_queryPools.forEach([](auto, GlQueryPool& q) {
        for (auto& ids : q.ids) glDeleteQueries(GLsizei(ids.size()), ids.data());
    });
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
}

core::Result<bool> GlDevice::beginFrame() {
    FL_ZONE("GlDevice::beginFrame");
    m_backbuffer = {m_desc.window->drawableWidth(), m_desc.window->drawableHeight()};
    if (m_backbuffer.width == 0 || m_backbuffer.height == 0) return false; // minimized
    // M9 plan D7: a lost device stays lost; nothing below may wait on it. Without reset notification (m_resetStatus null,
    // see init) a hung GPU shows only as a fence that never signals.
    if (m_desc.simulateLossAtFrame && ++m_framesBegun >= m_desc.simulateLossAtFrame && !m_lost) {
        FL_LOG_ERROR("Simulated device loss at frame %u", m_framesBegun);
        m_lost = true;
    }
    if (!m_lost && m_resetStatus && m_resetStatus() != GL_NO_ERROR) {
        FL_LOG_ERROR("The GL context was reset: the GPU device is lost");
        m_lost = true;
    }
    if (m_lost) return core::Error::DeviceLost;
    if (m_slotUsed) {
        m_slot = (m_slot + 1) % kFramesInFlight;
        m_slotUsed = false;
    }
    if (m_fences[m_slot]) {
        // Only blocks if the GPU is two frames behind; a frame that doesn't finish in 10 s means a hung GPU.
        GLenum waited = GL_TIMEOUT_EXPIRED;
        for (uint64_t waitedNs = 0; waited == GL_TIMEOUT_EXPIRED && waitedNs < kGpuHangTimeoutNs; waitedNs += 1'000'000'000)
            waited = glClientWaitSync(m_fences[m_slot], GL_SYNC_FLUSH_COMMANDS_BIT, 1'000'000'000);
        if (waited == GL_TIMEOUT_EXPIRED || waited == GL_WAIT_FAILED) {
            FL_LOG_ERROR("The GPU device is lost: %s", waited == GL_WAIT_FAILED ? "a fence wait failed" : "a frame didn't finish within 10 s");
            m_lost = true;
            return core::Error::DeviceLost;
        }
        glDeleteSync(m_fences[m_slot]);
        m_fences[m_slot] = nullptr;
        m_completedSerial = std::max(m_completedSerial, m_frameSerial[m_slot]);
    }
    runDeferred(false);
    m_inReadback = false;

    const bool fenced = m_desc.glStream == GlStreamStrategy::Fenced;
    for (uint32_t k = 0; k < uint32_t(TransientKind::Count); ++k)
        m_rings[k].beginFrame(m_ringStaging[k].data(), fenced ? m_rings[k].capacityBytes() * m_slot : 0);
    m_commandListsUsed = 0;
    m_frameActive = true;
    return true;
}

void GlDevice::reserveCommandBytes(size_t bytes) {
    if (bytes <= m_commandBytes) return;
    m_commandBytes = bytes;
    m_stats.commandListBytes = bytes;
    for (auto& slot : m_commandLists)
        for (auto& list : slot) list->reserve(bytes);
}

CommandList* GlDevice::allocateCommandList() {
    auto& lists = m_commandLists[m_slot];
    if (m_commandListsUsed == lists.size()) lists.push_back(std::make_unique<CommandList>(m_commandBytes));
    CommandList* list = lists[m_commandListsUsed++].get();
    list->reset();
    return list;
}

void GlDevice::submit(std::span<CommandList* const> lists) {
    FL_ZONE("GlDevice::submit");
    // 1. Per-draw constants go into the uniform ring before the single upload (ARCHITECTURE.md §4.7). When the blocks
    //    don't all fit (R40), the first ones and the last kPerDrawTailBlocks keep their place and the middle is dropped,
    //    so the end of the frame (heat haze, the post chain, the debug UI) still draws; the array never holds more entries
    //    than the ring holds blocks.
    m_perDrawOffsets.clear();
    TransientRing& uniforms = m_rings[uint32_t(TransientKind::Uniform)];
    uint32_t total = 0;
    for (CommandList* list : lists) {
        cmd::Reader reader(list->data());
        for (const cmd::Header* h = reader.next(); h; h = reader.next()) total += h->op == cmd::Op::SetPerDraw ? 1u : 0u;
    }
    const uint32_t align = std::max<uint32_t>(uniforms.alignment(), 1u);
    const uint32_t blockBytes = (shaderformat::kMaxPerDrawBytes + align - 1) / align * align;
    const uint32_t start = (uniforms.usedBytes() + align - 1) / align * align; // the frame's other uniforms come first
    const uint32_t fit = start < uniforms.capacityBytes() ? (uniforms.capacityBytes() - start) / blockBytes : 0u;
    const uint32_t tail = total > fit ? std::min<uint32_t>(fit, kPerDrawTailBlocks) : 0u;
    m_perDrawHead = total > fit ? fit - tail : total;
    m_perDrawTailStart = total > fit ? total - tail : total;
    uint32_t index = 0, dropped = 0;
    for (CommandList* list : lists) {
        cmd::Reader reader(list->data());
        for (const cmd::Header* h = reader.next(); h; h = reader.next()) {
            if (h->op != cmd::Op::SetPerDraw) continue;
            const uint32_t i = index++;
            if (i >= m_perDrawHead && i < m_perDrawTailStart) { // the replay maps it to kNoPerDraw and skips its draws
                ++dropped;
                continue;
            }
            const auto* c = cmd::Reader::as<cmd::SetPerDraw>(h);
            const TransientRing::Allocation a = uniforms.allocate(shaderformat::kMaxPerDrawBytes);
            if (!a.cpu) { // only if the estimate above was off: the entry keeps the head/tail mapping aligned
                ++dropped;
                m_perDrawOffsets.push_back(kNoPerDraw);
                continue;
            }
            std::memset(a.cpu, 0, shaderformat::kMaxPerDrawBytes);
            std::memcpy(a.cpu, cmd::payload(c), c->bytes);
            m_perDrawOffsets.push_back(a.gpu.offset);
        }
    }
    m_stats.skippedDraws = dropped;
    if (dropped) FL_LOG_ERROR("Uniform ring exhausted by per-draw constants: %u draws skipped this frame", dropped);

    // 2. One upload per ring per frame (ARCHITECTURE.md R11).
    const bool fenced = m_desc.glStream == GlStreamStrategy::Fenced;
    for (uint32_t k = 0; k < uint32_t(TransientKind::Count); ++k) {
        const uint32_t used = std::min(m_rings[k].usedBytes(), m_rings[k].capacityBytes());
        if (used == 0) continue;
        const GlBuffer* b = m_buffers.get(m_ringBuffers[k]);
        glBindBuffer(GL_COPY_WRITE_BUFFER, b->id);
        if (!fenced) glBufferData(GL_COPY_WRITE_BUFFER, b->size, nullptr, GL_STREAM_DRAW); // orphan
        glBufferSubData(GL_COPY_WRITE_BUFFER, m_rings[k].gpuBaseOffset(), used, m_ringStaging[k].data());
        m_stats.uploadedBytes += used;
    }
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

    // 3. Replay.
    m_stats.drawCalls = 0;
    uint32_t perDrawIndex = 0;
    for (CommandList* list : lists) {
        replay(*list, perDrawIndex);
        m_stats.drawCalls += list->drawCount();
    }
    m_fences[m_slot] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    m_frameSerial[m_slot] = ++m_submitSerial;
    m_slotUsed = true;
    m_frameActive = false;
}

void GlDevice::present() {
    FL_ZONE("GlDevice::present");
    m_desc.window->glSwapBuffers();
}

void GlDevice::readTimestamps(QueryPoolHandle handle, std::span<double> out) {
    std::fill(out.begin(), out.end(), std::nan(""));
    GlQueryPool* q = m_queryPools.get(handle);
    if (!q || !m_timerQueries) return;
    const uint32_t n = std::min<uint32_t>(q->count, uint32_t(out.size()));
    for (uint32_t i = 0; i < n; ++i) {
        if (!q->written[m_slot][i]) continue;
        GLint available = 0;
        GLuint64 value = 0;
        glGetQueryObjectiv(q->ids[m_slot][i], GL_QUERY_RESULT_AVAILABLE, &available);
        if (available) glGetQueryObjectui64v(q->ids[m_slot][i], GL_QUERY_RESULT, &value);
        if (available) out[i] = double(value) * 1e-6;
        q->written[m_slot][i] = 0;
    }
}

core::Result<void> GlDevice::readbackBackbuffer(std::vector<uint8_t>& out, Extent2D& extent) {
    extent = m_backbuffer;
    const uint32_t w = extent.width, h = extent.height;
    std::vector<uint8_t> rows(size_t(w) * h * 4);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    m_inReadback = true; // cleared at the next beginFrame(): drivers may report the stall at the next sync point
    glReadPixels(0, 0, GLsizei(w), GLsizei(h), GL_RGBA, GL_UNSIGNED_BYTE, rows.data());
    // GL's row 0 is the bottom of the screen (ARCHITECTURE.md §4.2); the contract is top-down rows.
    out.resize(rows.size());
    const size_t pitch = size_t(w) * 4;
    for (uint32_t y = 0; y < h; ++y) std::memcpy(out.data() + y * pitch, rows.data() + (h - 1 - y) * pitch, pitch);
    return glGetError() == GL_NO_ERROR ? core::Result<void>() : core::Result<void>(core::Error::Unsupported);
}

// Same lifetime rule as Vulkan: the frame being recorded may reference the object; between frames the last
// submitted frame is the newest possible user.
void GlDevice::defer(Deferred::Kind kind, GLuint id) {
    m_deferred.push_back({kind, id, m_frameActive ? m_submitSerial + 1 : m_submitSerial});
}

void GlDevice::deleteNow(const Deferred& d) {
    using K = Deferred::Kind;
    // GL recycles names: forget cached bindings of the deleted object, or a new object reusing the name would
    // look "already bound" and never be bound.
    for (uint32_t i = 0; i < 16; ++i) {
        if (d.kind == K::Texture && m_state.texture[i] == d.id) m_state.texture[i] = 0;
        if (d.kind == K::Sampler && m_state.sampler[i] == d.id) m_state.sampler[i] = 0;
        if (d.kind == K::Buffer && m_state.ubo[i] == d.id) m_state.ubo[i] = 0;
    }
    if (d.kind == K::Program && m_state.program == d.id) m_state.program = 0;
    if (d.kind == K::Texture) { // cached framebuffers name the texture; a recycled name must not match them
        for (size_t i = 0; i < m_fbos.size();) {
            const FboKey& k = m_fbos[i].key;
            if (k.depth == d.id || std::find(k.color, k.color + k.colorCount, d.id) != k.color + k.colorCount) {
                glDeleteFramebuffers(1, &m_fbos[i].fbo);
                m_fbos[i] = m_fbos.back();
                m_fbos.pop_back();
            } else {
                ++i;
            }
        }
    }
    switch (d.kind) {
    case K::Buffer: glDeleteBuffers(1, &d.id); break;
    case K::Texture: glDeleteTextures(1, &d.id); break;
    case K::Sampler: glDeleteSamplers(1, &d.id); break;
    case K::Shader: glDeleteShader(d.id); break;
    case K::Program: glDeleteProgram(d.id); break;
    case K::Query: glDeleteQueries(1, &d.id); break;
    }
}

// GL keeps objects alive while the GPU uses them, but deferring like Vulkan keeps lifetime bugs identical
// on both backends (ARCHITECTURE.md §4.9).
void GlDevice::runDeferred(bool all) {
    size_t kept = 0;
    for (size_t i = 0; i < m_deferred.size(); ++i) {
        if (all || m_deferred[i].serial <= m_completedSerial) deleteNow(m_deferred[i]);
        else m_deferred[kept++] = m_deferred[i];
    }
    m_deferred.resize(kept);
}

} // namespace rhi::gl
