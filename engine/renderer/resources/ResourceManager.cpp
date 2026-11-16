#include "renderer/resources/ResourceManager.h"

#include "core/Log.h"
#include "core/Profile.h"
#include "platform/FileSystem.h"

#include <algorithm>
#include <cstring>

namespace renderer {

uint32_t budgetMipSkip(std::span<const uint64_t> mipBytes, uint32_t width, uint32_t height, uint32_t firstMip, uint64_t available,
                       uint32_t minTop) {
    const uint32_t mips = uint32_t(mipBytes.size());
    for (uint32_t keep = firstMip; keep < mips; ++keep) {
        if (keep > firstMip && std::max(width >> keep, height >> keep) < minTop) break;
        uint64_t bytes = 0;
        for (uint32_t m = keep; m < mips; ++m) bytes += mipBytes[m];
        if (bytes <= available) return keep;
    }
    return UINT32_MAX;
}

namespace {

size_t align16(size_t v) {
    return (v + 15) & ~size_t(15);
}

rhi::Format toRhiFormat(assets::PixelFormat f) {
    switch (f) {
    case assets::PixelFormat::RGBA8Unorm: return rhi::Format::RGBA8Unorm;
    case assets::PixelFormat::RGBA8Srgb: return rhi::Format::RGBA8Srgb;
    case assets::PixelFormat::BC1Srgb: return rhi::Format::BC1Srgb;
    case assets::PixelFormat::BC3Srgb: return rhi::Format::BC3Srgb;
    case assets::PixelFormat::BC3Unorm: return rhi::Format::BC3Unorm;
    case assets::PixelFormat::BC6H: return rhi::Format::BC6HUfloat;
    case assets::PixelFormat::BC4: return rhi::Format::BC4;
    case assets::PixelFormat::BC5: return rhi::Format::BC5;
    case assets::PixelFormat::BC7Unorm: return rhi::Format::BC7Unorm;
    case assets::PixelFormat::BC7Srgb: return rhi::Format::BC7Srgb;
    case assets::PixelFormat::Unknown: break;
    }
    return rhi::Format::Undefined;
}

rhi::TextureHandle makeSolidTexture(rhi::Device& device, const char* name, rhi::Format format, uint8_t r, uint8_t g, uint8_t b,
                                    uint8_t a) {
    rhi::TextureDesc td;
    td.debugName = name;
    td.format = format;
    const rhi::TextureHandle t = device.createTexture(td);
    const uint8_t pixel[4] = {r, g, b, a};
    rhi::TextureUpload u;
    u.width = u.height = 1;
    u.data = std::as_bytes(std::span(pixel));
    if (t && !device.uploadTexture(t, u).ok()) {
        device.destroy(t);
        return {};
    }
    return t;
}

template <typename H>
H fromBits(uint32_t bits) {
    H h;
    h.bits = bits;
    return h;
}

} // namespace

ResourceManager::~ResourceManager() {
    shutdown();
}

core::Result<void> ResourceManager::init(const Desc& desc) {
    m_device = desc.device;
    m_dataDir = desc.dataDir;
    m_targets.bc = m_device->caps().bc1to5;
    m_targets.bptc = m_device->caps().bptc;
    m_grey = makeSolidTexture(*m_device, "PlaceholderGrey", rhi::Format::RGBA8Srgb, 128, 128, 128, 255);
    m_white = makeSolidTexture(*m_device, "White", rhi::Format::RGBA8Srgb, 255, 255, 255, 255);
    m_black = makeSolidTexture(*m_device, "Black", rhi::Format::RGBA8Srgb, 0, 0, 0, 0);
    m_flatNormal = makeSolidTexture(*m_device, "FlatNormal", rhi::Format::RGBA8Unorm, 128, 128, 255, 255);
    if (!m_grey || !m_white || !m_black || !m_flatNormal) return core::Error::InitFailed;
    m_stop = false;
    for (uint32_t i = 0; i < std::max(1u, desc.loaderThreads); ++i) m_threads.emplace_back([this] { loaderLoop(); });
    FL_LOG_INFO("Resource manager: %u loader threads, textures transcode to %s", uint32_t(m_threads.size()),
                m_targets.bptc ? "BC1/BC7" : m_targets.bc ? "BC1/BC3" : "RGBA8");
    return {};
}

void ResourceManager::shutdown() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_wake.notify_all();
    for (std::thread& t : m_threads) t.join();
    m_threads.clear();
    if (!m_device) return;
    // Materials are torn down by the material system itself; only GPU objects owned here are released.
    m_models.forEach([&](auto, ModelAsset& m) {
        if (m.buffer) m_device->destroy(m.buffer);
    });
    m_textures.forEach([&](auto, TextureRecord& t) {
        if (t.texture) m_device->destroy(t.texture);
    });
    m_device->destroy(m_grey);
    m_device->destroy(m_white);
    m_device->destroy(m_black);
    m_device->destroy(m_flatNormal);
    m_requests.clear();
    m_loaded.clear();
    m_device = nullptr;
}

void ResourceManager::enqueue(Kind kind, uint32_t handle, const std::string& path, uint32_t serial) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_requests.push_back({kind, handle, path, serial});
    }
    m_wake.notify_one();
    ++m_inFlight;
}

ModelHandle ResourceManager::loadModel(const std::string& path) {
    if (auto it = m_modelByPath.find(path); it != m_modelByPath.end()) {
        const ModelHandle h = fromBits<ModelHandle>(it->second);
        if (ModelAsset* m = m_models.get(h)) {
            ++m->refs;
            return h;
        }
    }
    ModelAsset asset;
    asset.path = path;
    asset.refs = 1;
    const ModelHandle h = m_models.create(std::move(asset));
    if (!h) {
        FL_LOG_ERROR("Model pool exhausted; %s not loaded", path.c_str());
        return {};
    }
    m_modelByPath[path] = h.bits;
    enqueue(Kind::Model, h.bits, path);
    return h;
}

TextureAsset ResourceManager::loadTexture(const std::string& path) {
    if (auto it = m_textureByPath.find(path); it != m_textureByPath.end()) {
        const TextureAsset h = fromBits<TextureAsset>(it->second);
        if (TextureRecord* t = m_textures.get(h)) {
            ++t->refs;
            return h;
        }
    }
    TextureRecord record;
    record.path = path;
    record.refs = 1;
    const TextureAsset h = m_textures.create(std::move(record));
    if (!h) {
        FL_LOG_ERROR("Texture pool exhausted; %s not loaded", path.c_str());
        return {};
    }
    m_textureByPath[path] = h.bits;
    enqueue(Kind::Texture, h.bits, path);
    return h;
}

bool ResourceManager::reloadTexture(const std::string& path) {
    const auto it = m_textureByPath.find(path);
    TextureRecord* t = it == m_textureByPath.end() ? nullptr : m_textures.get(fromBits<TextureAsset>(it->second));
    if (!t) return false;
    // Two loader threads may finish loads of one file out of order (two quick saves): each reload is numbered and finalize
    // keeps only the latest one's.
    enqueue(Kind::Texture, it->second, path, ++t->reloads); // finalize sees a Ready record and replaces its texture
    return true;
}

AnimHandle ResourceManager::loadAnimation(const std::string& path) {
    if (auto it = m_animByPath.find(path); it != m_animByPath.end()) {
        const AnimHandle h = fromBits<AnimHandle>(it->second);
        if (AnimAsset* a = m_anims.get(h)) {
            ++a->refs;
            return h;
        }
    }
    AnimAsset asset;
    asset.path = path;
    asset.refs = 1;
    const AnimHandle h = m_anims.create(std::move(asset));
    if (!h) {
        FL_LOG_ERROR("Animation pool exhausted; %s not loaded", path.c_str());
        return {};
    }
    m_animByPath[path] = h.bits;
    enqueue(Kind::Anim, h.bits, path);
    return h;
}

void ResourceManager::release(AnimHandle handle) {
    AnimAsset* a = m_anims.get(handle);
    if (!a || --a->refs > 0) return;
    if (auto it = m_animByPath.find(a->path); it != m_animByPath.end() && it->second == handle.bits) m_animByPath.erase(it);
    m_anims.destroy(handle);
}

PresetHandle ResourceManager::loadPreset(const std::string& path) {
    if (auto it = m_presetByPath.find(path); it != m_presetByPath.end()) {
        const PresetHandle h = fromBits<PresetHandle>(it->second);
        if (PresetAsset* p = m_presets.get(h)) {
            ++p->refs;
            return h;
        }
    }
    PresetAsset asset;
    asset.path = path;
    asset.refs = 1;
    const PresetHandle h = m_presets.create(std::move(asset));
    if (!h) {
        FL_LOG_ERROR("Particle preset pool exhausted; %s not loaded", path.c_str());
        return {};
    }
    m_presetByPath[path] = h.bits;
    enqueue(Kind::Preset, h.bits, path);
    return h;
}

PresetHandle ResourceManager::addPreset(const std::string& name, assets::ParticleEffectFile file) {
    std::string error;
    if (!assets::validateParticleEffect(file, &error)) { // a file's preset was validated by the parser; this one wasn't
        FL_LOG_ERROR("Particle preset %s: %s; not added", name.c_str(), error.c_str());
        return {};
    }
    PresetAsset asset;
    asset.path = name;
    asset.refs = 1;
    asset.state = LoadState::Ready;
    for (const assets::ParticleEmitterDesc& e : file.emitters)
        asset.textures.push_back(e.texture.empty() ? TextureAsset{} : loadTexture(e.texture));
    asset.file = std::move(file);
    const PresetHandle h = m_presets.create(std::move(asset));
    if (!h) FL_LOG_ERROR("Particle preset pool exhausted; %s not added", name.c_str());
    return h; // not entered by path: a later loadPreset of the same name loads the file
}

void ResourceManager::addRef(PresetHandle handle) {
    if (PresetAsset* p = m_presets.get(handle)) ++p->refs;
}

void ResourceManager::release(PresetHandle handle) {
    PresetAsset* p = m_presets.get(handle);
    if (!p || --p->refs > 0) return;
    for (TextureAsset t : p->textures)
        if (t) release(t);
    if (auto it = m_presetByPath.find(p->path); it != m_presetByPath.end() && it->second == handle.bits) m_presetByPath.erase(it);
    m_presets.destroy(handle);
}

LoadState ResourceManager::state(PresetHandle handle) const {
    const PresetAsset* p = m_presets.get(handle);
    return p ? p->state : LoadState::Failed;
}

const PresetAsset* ResourceManager::preset(PresetHandle handle) const {
    return m_presets.get(handle);
}

LoadState ResourceManager::state(AnimHandle handle) const {
    const AnimAsset* a = m_anims.get(handle);
    return a ? a->state : LoadState::Failed;
}

const AnimAsset* ResourceManager::animation(AnimHandle handle) const {
    return m_anims.get(handle);
}

void ResourceManager::addRef(ModelHandle handle) {
    if (ModelAsset* m = m_models.get(handle)) ++m->refs;
}

void ResourceManager::addRef(TextureAsset handle) {
    if (TextureRecord* t = m_textures.get(handle)) ++t->refs;
}

void ResourceManager::release(ModelHandle handle) {
    ModelAsset* m = m_models.get(handle);
    if (!m || --m->refs > 0) return;
    if (m->buffer) m_device->destroy(m->buffer);
    if (m_materialCallbacks.release)
        for (MaterialHandle material : m->materials) m_materialCallbacks.release(m_materialCallbacks.user, material);
    m_usage[uint32_t(MemoryCategory::Meshes)] -= m->gpuBytes;
    if (auto it = m_modelByPath.find(m->path); it != m_modelByPath.end() && it->second == handle.bits) m_modelByPath.erase(it);
    m_models.destroy(handle); // a load still in flight finds the handle stale and is discarded
}

void ResourceManager::release(TextureAsset handle) {
    TextureRecord* t = m_textures.get(handle);
    if (!t || --t->refs > 0) return;
    if (t->texture) m_device->destroy(t->texture);
    m_usage[uint32_t(MemoryCategory::Textures)] -= t->gpuBytes;
    if (auto it = m_textureByPath.find(t->path); it != m_textureByPath.end() && it->second == handle.bits)
        m_textureByPath.erase(it);
    m_textures.destroy(handle);
}

LoadState ResourceManager::state(ModelHandle handle) const {
    const ModelAsset* m = m_models.get(handle);
    return m ? m->state : LoadState::Failed;
}

LoadState ResourceManager::state(TextureAsset handle) const {
    const TextureRecord* t = m_textures.get(handle);
    return t ? t->state : LoadState::Failed;
}

const ModelAsset* ResourceManager::model(ModelHandle handle) const {
    return m_models.get(handle);
}

rhi::TextureHandle ResourceManager::texture(TextureAsset handle) const {
    const TextureRecord* t = m_textures.get(handle);
    return t && t->state == LoadState::Ready ? t->texture : rhi::TextureHandle{};
}

uint32_t ResourceManager::textureMips(TextureAsset handle) const {
    const TextureRecord* t = m_textures.get(handle);
    return t && t->state == LoadState::Ready ? t->mips : 0;
}

void ResourceManager::loaderLoop() {
    while (true) {
        Request request;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_wake.wait(lock, [this] { return m_stop || !m_requests.empty(); });
            if (m_stop) return;
            request = std::move(m_requests.front());
            m_requests.pop_front();
        }
        Loaded out;
        out.kind = request.kind;
        out.handle = request.handle;
        out.serial = request.serial;
        // Engine code doesn't throw, but containers and third-party decoders can (std::bad_alloc on a huge file).
        // An exception escaping this thread would terminate the process; it fails this one load instead.
        try {
            if (request.kind == Kind::Model) loadModelFile(request, out);
            else if (request.kind == Kind::Anim) loadAnimFile(request, out);
            else if (request.kind == Kind::Preset) loadPresetFile(request, out);
            else loadTextureFile(request, out);
        } catch (const std::exception& e) {
            out = Loaded{};
            out.kind = request.kind;
            out.handle = request.handle;
            out.serial = request.serial;
            out.error = std::string("loading threw: ") + e.what();
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        m_loaded.push_back(std::move(out));
    }
}

void ResourceManager::loadModelFile(const Request& request, Loaded& out) const {
    FL_ZONE("Load model");
    auto file = platform::readFile(m_dataDir + request.path);
    if (!file) {
        out.error = "cannot read the file";
        return;
    }
    assets::ModelView view;
    std::string error;
    if (!assets::openModel(file.value(), view, &error)) {
        out.error = error;
        return;
    }
    out.nodes.assign(view.nodes.begin(), view.nodes.end());
    // The hierarchy as the anim library reads it (M6 plan D1).
    for (const assets::ModelNode& n : view.nodes) {
        out.parents.push_back(n.parent);
        anim::Transform t;
        t.translation = glm::vec3(n.translation[0], n.translation[1], n.translation[2]);
        t.rotation = glm::quat(n.rotation[3], n.rotation[0], n.rotation[1], n.rotation[2]);
        t.scale = glm::vec3(n.scale[0], n.scale[1], n.scale[2]);
        out.rest.push_back(t);
        out.nameHashes.push_back(n.nameHash);
    }
    out.joints.assign(view.joints.begin(), view.joints.end());
    for (const assets::ModelSubmesh& s : view.submeshes) {
        GpuSubmesh g;
        g.materialSlot = s.materialSlot;
        g.firstIndex = s.lods[0].firstIndex;
        g.indexCount = s.lods[0].indexCount;
        for (uint32_t l = 0; l < assets::kMaxLods; ++l) {
            const assets::ModelLod& lod = s.lods[std::min(l, s.lodCount - 1)]; // openModel checked 1 ≤ lodCount ≤ 4
            g.lodFirst[l] = lod.firstIndex;
            g.lodCount[l] = lod.indexCount;
        }
        out.submeshes.push_back(g);
    }

    assets::DecodedMesh decoded;
    auto place = [&](const std::vector<std::byte>& bytes) {
        const size_t offset = align16(out.meshData.size());
        out.meshData.resize(offset + bytes.size());
        std::memcpy(out.meshData.data() + offset, bytes.data(), bytes.size());
        return rhi::BufferSlice{{}, uint32_t(offset), uint32_t(bytes.size())};
    };
    for (uint32_t i = 0; i < view.meshes.size(); ++i) {
        if (!assets::decodeMesh(view, i, decoded, &error)) {
            out.error = "mesh " + std::to_string(i) + ": " + error;
            return;
        }
        const assets::ModelMesh& mm = view.meshes[i];
        GpuMesh g;
        g.positions = place(decoded.positions);
        g.surface = place(decoded.surface);
        g.indices = place(decoded.indices);
        if (mm.jointCount > 0) {
            g.skin = place(decoded.skin);
            g.firstJoint = mm.firstJoint;
            g.jointCount = mm.jointCount;
        }
        g.indexType = mm.indexBytes == 2 ? rhi::IndexType::U16 : rhi::IndexType::U32;
        g.firstSubmesh = mm.firstSubmesh;
        g.submeshCount = mm.submeshCount;
        g.dequant.step = mm.dequantStep;
        g.dequant.offset = glm::vec3(mm.dequantOffset[0], mm.dequantOffset[1], mm.dequantOffset[2]);
        g.sphere = glm::vec4(mm.sphere[0], mm.sphere[1], mm.sphere[2], mm.sphere[3]);
        g.boundsMin = glm::vec3(mm.boundsMin[0], mm.boundsMin[1], mm.boundsMin[2]);
        g.boundsMax = glm::vec3(mm.boundsMax[0], mm.boundsMax[1], mm.boundsMax[2]);
        const assets::ModelSubmesh& first = view.submeshes[mm.firstSubmesh]; // every submesh has the same levels
        g.lodCount = first.lodCount;
        for (uint32_t l = 1; l < first.lodCount; ++l) g.lodScreenSize[l] = first.lods[l].screenSize;
        out.meshes.push_back(g);
    }
    // Material files are tiny; reading them here keeps file I/O off the main thread.
    for (const assets::ModelMaterialSlot& slot : view.materials) {
        out.materialPaths.emplace_back(slot.path);
        assets::MaterialFile material;
        auto bytes = platform::readFile(m_dataDir + slot.path);
        if (!bytes || !assets::parseMaterial(bytes.value(), material, &error)) {
            FL_LOG_ERROR("Material %s (model %s): %s; using the error material", slot.path, request.path.c_str(),
                         bytes ? error.c_str() : "cannot read the file");
            material = assets::MaterialFile{};
            material.templateName.clear(); // the material system renders an empty template as the error material
        }
        out.materialFiles.push_back(material);
    }
    out.ok = true;
}

void ResourceManager::loadAnimFile(const Request& request, Loaded& out) const {
    FL_ZONE("Load animation");
    auto file = platform::readFile(m_dataDir + request.path);
    if (!file) {
        out.error = "cannot read the file";
        return;
    }
    out.animBytes = std::move(file.value());
    std::string error;
    if (!assets::openAnim(out.animBytes, out.animView, &error)) { // the view points into animBytes, which moves intact
        out.error = error;
        return;
    }
    out.ok = true;
}

void ResourceManager::loadPresetFile(const Request& request, Loaded& out) const {
    FL_ZONE("Load particle preset");
    auto file = platform::readFile(m_dataDir + request.path);
    if (!file) {
        out.error = "cannot read the file";
        return;
    }
    if (!assets::parseParticleEffect(file.value(), out.preset, &out.error)) return;
    out.ok = true;
}

void ResourceManager::loadTextureFile(const Request& request, Loaded& out) const {
    FL_ZONE("Load texture");
    auto file = platform::readFile(m_dataDir + request.path);
    if (!file) {
        out.error = "cannot read the file";
        return;
    }
    std::string error;
    if (!assets::decodeKtx2(file.value(), m_targets, out.texture, &error)) {
        out.error = error;
        return;
    }
    out.ok = true;
}

void ResourceManager::finalizeUploads(uint64_t byteBudget) {
    FL_ZONE("ResourceManager::finalizeUploads");
    uint64_t used = 0;
    for (bool first = true; first || used < byteBudget; first = false) {
        // Check before constructing: even an empty Loaded allocates in debug-STL builds (container proxies), and
        // this runs every frame.
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_loaded.empty()) break;
        Loaded item = std::move(m_loaded.front());
        m_loaded.pop_front();
        lock.unlock();
        --m_inFlight;
        used += finalize(item);
    }
}

uint64_t ResourceManager::finalize(Loaded& item) {
    if (item.kind == Kind::Anim) {
        AnimAsset* a = m_anims.get(fromBits<AnimHandle>(item.handle));
        if (!a) return 0; // released while loading
        if (!item.ok) {
            a->state = LoadState::Failed;
            FL_LOG_ERROR("Animation %s: %s", a->path.c_str(), item.error.c_str());
            return 0;
        }
        a->bytes = std::move(item.animBytes); // the heap block moves, so the view's spans stay valid
        a->view = item.animView;
        a->state = LoadState::Ready;
        return 0;
    }
    if (item.kind == Kind::Preset) {
        PresetAsset* p = m_presets.get(fromBits<PresetHandle>(item.handle));
        if (!p) return 0; // released while loading
        if (!item.ok) {
            p->state = LoadState::Failed;
            FL_LOG_ERROR("Particle preset %s: %s", p->path.c_str(), item.error.c_str());
            return 0;
        }
        p->file = std::move(item.preset);
        p->textures.clear();
        for (const assets::ParticleEmitterDesc& e : p->file.emitters) // loads are de-duplicated by path
            p->textures.push_back(e.texture.empty() ? TextureAsset{} : loadTexture(e.texture));
        p->state = LoadState::Ready;
        return 0;
    }
    if (item.kind == Kind::Model) {
        ModelAsset* m = m_models.get(fromBits<ModelHandle>(item.handle));
        if (!m) return 0; // released while loading
        const uint64_t bytes = item.meshData.size();
        const uint64_t meshBudget = m_budget[uint32_t(MemoryCategory::Meshes)]; // 0 = no budget (MemoryBudgets)
        if (item.ok && meshBudget && m_usage[uint32_t(MemoryCategory::Meshes)] + bytes > meshBudget)
            item.ok = false, item.error = "the mesh memory budget is exhausted";
        if (item.ok) {
            rhi::BufferDesc bd;
            bd.debugName = m->path.c_str();
            bd.size = uint32_t(bytes);
            bd.usage = rhi::BufferUsage::Vertex | rhi::BufferUsage::Index;
            bd.memory = rhi::MemoryKind::GpuOnly;
            bd.initialData = item.meshData.data();
            m->buffer = m_device->createBuffer(bd);
            if (!m->buffer) item.ok = false, item.error = "GPU buffer creation failed";
        }
        if (!item.ok) {
            m->state = LoadState::Failed;
            FL_LOG_ERROR("Model %s: %s", m->path.c_str(), item.error.c_str());
            return 0;
        }
        for (GpuMesh& mesh : item.meshes) {
            mesh.positions.buffer = mesh.surface.buffer = mesh.indices.buffer = m->buffer;
            if (mesh.jointCount) mesh.skin.buffer = m->buffer;
            m->skinned = m->skinned || mesh.jointCount > 0;
        }
        m->parents = std::move(item.parents);
        m->rest = std::move(item.rest);
        m->nameHashes = std::move(item.nameHashes);
        m->joints = std::move(item.joints);
        m->nodes = std::move(item.nodes);
        m->meshes = std::move(item.meshes);
        m->submeshes = std::move(item.submeshes);
        m->materials.clear();
        for (size_t i = 0; i < item.materialPaths.size(); ++i)
            m->materials.push_back(m_materialCallbacks.acquire
                                       ? m_materialCallbacks.acquire(m_materialCallbacks.user, item.materialPaths[i], item.materialFiles[i])
                                       : MaterialHandle{});
        m->gpuBytes = bytes;
        m_usage[uint32_t(MemoryCategory::Meshes)] += bytes;
        m->state = LoadState::Ready;
        return bytes;
    }

    TextureRecord* t = m_textures.get(fromBits<TextureAsset>(item.handle));
    if (!t) return 0;
    if (item.serial != t->reloads) return 0; // a later reload of this file is on its way: this load is older (dropped)
    const assets::DecodedTexture& tex = item.texture;
    const rhi::Format format = item.ok ? toRhiFormat(tex.format) : rhi::Format::Undefined;
    if (item.ok && format == rhi::Format::Undefined) item.ok = false, item.error = "unsupported pixel format";
    // Mips larger than the device allows are skipped rather than failing the texture.
    uint32_t firstMip = 0;
    while (item.ok && firstMip + 1 < tex.mips && std::max(tex.width >> firstMip, tex.height >> firstMip) > m_device->caps().maxTextureSize)
        ++firstMip;
    // M9 plan D6: a texture that doesn't fit the texture budget skips its largest mips until it does.
    uint64_t mipBytes[16] = {};
    const uint32_t mipCount = std::min<uint32_t>(tex.mips, 16u);
    for (const assets::DecodedTexture::Level& l : tex.levels)
        if (l.mip < mipCount) mipBytes[l.mip] += l.size;
    // A hot reload (M9 plan D9) replaces a Ready texture: its old bytes count as free for the new one, and the old texture
    // stays until the new one is made.
    const bool reload = t->state == LoadState::Ready && t->texture;
    if (item.ok) {
        const uint64_t used = m_usage[uint32_t(MemoryCategory::Textures)] - (reload ? t->gpuBytes : 0);
        const uint64_t budget = m_budget[uint32_t(MemoryCategory::Textures)];
        const uint64_t available = budget == 0 ? UINT64_MAX : budget > used ? budget - used : 0; // 0 = no budget
        const uint32_t keep = budgetMipSkip(std::span<const uint64_t>(mipBytes, mipCount), tex.width, tex.height, firstMip, available);
        if (keep == UINT32_MAX) {
            item.ok = false, item.error = "the texture memory budget is exhausted";
        } else if (keep > firstMip) {
            FL_LOG_WARN("Texture %s: %ux%u instead of %ux%u, to fit the texture budget", t->path.c_str(), std::max(1u, tex.width >> keep),
                        std::max(1u, tex.height >> keep), std::max(1u, tex.width >> firstMip), std::max(1u, tex.height >> firstMip));
            firstMip = keep;
            ++m_reducedTextures;
        }
    }
    uint64_t bytes = 0;
    for (const assets::DecodedTexture::Level& l : tex.levels)
        if (l.mip >= firstMip) bytes += l.size;
    rhi::TextureHandle created;
    if (item.ok) {
        rhi::TextureDesc td;
        td.debugName = t->path.c_str();
        td.type = tex.shape == assets::TextureShape::Cube    ? rhi::TextureType::Cube
                  : tex.shape == assets::TextureShape::Array ? rhi::TextureType::Tex2DArray
                                                             : rhi::TextureType::Tex2D;
        td.format = format;
        td.width = std::max(1u, tex.width >> firstMip);
        td.height = std::max(1u, tex.height >> firstMip);
        td.layers = uint16_t(tex.layers);
        td.mips = uint16_t(tex.mips - firstMip);
        td.usage = rhi::TextureUsage::Sampled;
        created = m_device->createTexture(td);
        if (!created) item.ok = false, item.error = "GPU texture creation failed";
        for (const assets::DecodedTexture::Level& l : tex.levels) {
            if (!item.ok) break;
            if (l.mip < firstMip) continue;
            rhi::TextureUpload u;
            u.mip = l.mip - firstMip;
            u.layer = l.layer;
            u.width = l.width;
            u.height = l.height;
            u.data = std::span(tex.data.data() + l.offset, l.size);
            if (!m_device->uploadTexture(created, u).ok()) item.ok = false, item.error = "upload failed";
        }
    }
    if (!item.ok) {
        if (created) m_device->destroy(created);
        if (reload) { // the old texture stays in use
            FL_LOG_ERROR("Hot reload: texture %s: %s; the old one stays", t->path.c_str(), item.error.c_str());
            return 0;
        }
        t->texture = {};
        t->state = LoadState::Failed;
        FL_LOG_ERROR("Texture %s: %s", t->path.c_str(), item.error.c_str());
        return 0;
    }
    if (reload) {
        m_device->destroy(t->texture); // deferred: frames in flight may still sample it
        m_usage[uint32_t(MemoryCategory::Textures)] -= t->gpuBytes;
    }
    // Any finished hot reload moves the revision on, also one that fixed a texture whose first load had failed: whoever
    // holds sets with it, or a placeholder for it, looks again.
    if (item.serial) ++m_textureRevision;
    t->texture = created;
    t->gpuBytes = bytes;
    t->mips = tex.mips - firstMip;
    m_usage[uint32_t(MemoryCategory::Textures)] += bytes;
    t->state = LoadState::Ready;
    return bytes;
}

} // namespace renderer
