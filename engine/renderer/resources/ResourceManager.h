#pragma once

#include "anim/Pose.h"
#include "assets/AnimFile.h"
#include "assets/Material.h"
#include "assets/Model.h"
#include "assets/ParticleFormat.h"
#include "assets/Quantize.h"
#include "assets/Texture.h"
#include "core/Handle.h"
#include "core/Result.h"
#include "rhi/Device.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace renderer {

using ModelHandle = core::Handle<struct ModelTag>;
using TextureAsset = core::Handle<struct TextureAssetTag>;
using MaterialHandle = core::Handle<struct MaterialTag>;
using AnimHandle = core::Handle<struct AnimTag>; // a .fanim clip library (M6 plan D3)
using PresetHandle = core::Handle<struct PresetTag>; // a .fpart particle effect preset (M7 plan D2)

enum class LoadState : uint8_t { Loading, Ready, Failed };
enum class MemoryCategory : uint8_t { Textures, Meshes, Count };

// The texture-budget fallback (M9 plan D6): the first mip to keep so the texture fits `available` bytes, starting at
// `firstMip`. `mipBytes` holds each level's bytes over all layers. Skipping a level drops about three quarters of the
// size; a texture is never cut below a `minTop`-pixel top level (one smaller than that to begin with isn't cut at all).
// UINT32_MAX when even the smallest allowed cut doesn't fit.
uint32_t budgetMipSkip(std::span<const uint64_t> mipBytes, uint32_t width, uint32_t height, uint32_t firstMip, uint64_t available,
                       uint32_t minTop = 64);

struct GpuSubmesh {
    uint32_t materialSlot = 0;
    uint32_t firstIndex = 0; // level 0
    uint32_t indexCount = 0;
    // Index ranges of every LOD (M5 plan D5) over the mesh's shared streams; [0] repeats firstIndex/indexCount. Every
    // submesh of a mesh has GpuMesh::lodCount levels.
    uint32_t lodFirst[assets::kMaxLods] = {};
    uint32_t lodCount[assets::kMaxLods] = {};
};

struct GpuMesh {
    rhi::BufferSlice positions; // 8 B/vertex
    rhi::BufferSlice surface;   // 20 B/vertex
    rhi::BufferSlice indices;
    rhi::BufferSlice skin;      // 8 B/vertex, skinned meshes only (M6 plan D2)
    rhi::IndexType indexType = rhi::IndexType::U16;
    uint32_t firstSubmesh = 0;  // into ModelAsset::submeshes
    uint32_t submeshCount = 0;
    assets::Dequantization dequant;
    glm::vec4 sphere{0.0f};     // object space: center, radius
    glm::vec3 boundsMin{0.0f};  // object space box (blob shadows, M4 plan D9)
    glm::vec3 boundsMax{0.0f};
    uint32_t lodCount = 1;      // M5 plan D5: levels of every submesh
    float lodScreenSize[assets::kMaxLods] = {}; // [l ≥ 1]: largest screen size the level serves (renderer/visibility/Lod.h)
    uint32_t firstJoint = 0;    // its skin: ModelAsset::joints[firstJoint, firstJoint + jointCount); 0 joints = rigid
    uint32_t jointCount = 0;
};

// A loaded model: one GPU buffer holds every mesh's streams and indices (M1 plan D1).
struct ModelAsset {
    std::string path;
    LoadState state = LoadState::Loading;
    uint32_t refs = 0;
    rhi::BufferHandle buffer;
    uint64_t gpuBytes = 0;
    std::vector<assets::ModelNode> nodes;
    std::vector<GpuMesh> meshes;
    std::vector<GpuSubmesh> submeshes;
    std::vector<MaterialHandle> materials; // one per material slot, resolved when the model becomes Ready
    // Animation (M6 plan D1): the node hierarchy as the anim library reads it, and every skin's joints.
    std::vector<int32_t> parents;
    std::vector<anim::Transform> rest;
    std::vector<uint32_t> nameHashes;
    std::vector<assets::ModelJoint> joints;
    bool skinned = false; // some mesh has a skin
};

// A loaded clip library: the file and its validated view (spans into `bytes`).
struct AnimAsset {
    std::string path;
    LoadState state = LoadState::Loading;
    uint32_t refs = 0;
    std::vector<std::byte> bytes;
    assets::AnimView view;
};

// A loaded particle preset (M7 plan D2): the validated file and one atlas per emitter (loaded when the preset became
// Ready, released with it; empty for emitters without a texture).
struct PresetAsset {
    std::string path;
    LoadState state = LoadState::Loading;
    uint32_t refs = 0;
    assets::ParticleEffectFile file;
    std::vector<TextureAsset> textures; // one per emitter
};

// The material system resolves a model's slots when it becomes Ready (and releases them with it). Plain function
// pointers keep ResourceManager free of a MaterialSystem dependency.
struct MaterialCallbacks {
    MaterialHandle (*acquire)(void* user, const std::string& path, const assets::MaterialFile& file) = nullptr;
    void (*release)(void* user, MaterialHandle material) = nullptr;
    void* user = nullptr;
};

// Handle-based, ref-counted asset cache (ARCHITECTURE.md §5.4, M1 plan D4). load*() returns immediately; loader
// threads read, validate and decode (meshopt streams, KTX2 transcoding), and finalizeUploads() turns finished loads
// into GPU resources on the main thread under a byte budget. Loads are de-duplicated by path. Budgets are enforced
// with computed byte sizes on both backends; a load that would exceed one fails with a logged error.
class ResourceManager {
public:
    struct Desc {
        rhi::Device* device = nullptr;
        std::string dataDir; // with a trailing separator
        uint32_t loaderThreads = 2;
    };

    ResourceManager() = default;
    ~ResourceManager();
    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;

    core::Result<void> init(const Desc& desc);
    void shutdown();
    void setMaterialCallbacks(const MaterialCallbacks& callbacks) { m_materialCallbacks = callbacks; }

    ModelHandle loadModel(const std::string& path); // data-relative; each call adds a reference
    TextureAsset loadTexture(const std::string& path);
    AnimHandle loadAnimation(const std::string& path); // a .fanim clip library (M6 plan D3); no GPU work
    PresetHandle loadPreset(const std::string& path);  // a .fpart particle preset (M7 plan D2); its atlases load once it is Ready
    // A preset built in code (tests, tools), Ready at once under `name` (its emitters' atlases load as with loadPreset).
    PresetHandle addPreset(const std::string& name, assets::ParticleEffectFile file);
    void addRef(ModelHandle handle);
    void addRef(TextureAsset handle);
    void addRef(PresetHandle handle);
    void release(ModelHandle handle);
    void release(TextureAsset handle);
    void release(AnimHandle handle);
    void release(PresetHandle handle);

    LoadState state(ModelHandle handle) const;      // Failed for stale handles
    LoadState state(TextureAsset handle) const;
    LoadState state(AnimHandle handle) const;
    LoadState state(PresetHandle handle) const;
    const ModelAsset* model(ModelHandle handle) const; // nullptr for stale handles
    const AnimAsset* animation(AnimHandle handle) const; // nullptr for stale handles; check state() before using the view
    const PresetAsset* preset(PresetHandle handle) const; // nullptr for stale handles; check state() before using the file
    rhi::TextureHandle texture(TextureAsset handle) const; // invalid until Ready
    uint32_t textureMips(TextureAsset handle) const;       // mip count once Ready, else 0
    rhi::TextureHandle greyTexture() const { return m_grey; }   // placeholder while a texture loads
    rhi::TextureHandle whiteTexture() const { return m_white; } // materials without a texture
    rhi::TextureHandle blackTexture() const { return m_black; } // RGBA 0: no specular or emission while a map loads
    rhi::TextureHandle flatNormalTexture() const { return m_flatNormal; } // linear (0.5, 0.5, 1): an unperturbed normal

    // Main thread, right after Device::beginFrame(). At least one finished load is finalized per call, so an item
    // larger than the budget still makes progress.
    void finalizeUploads(uint64_t byteBudget);

    uint32_t pendingCount() const { return m_inFlight; } // requested, not yet Ready or Failed
    const std::string& dataDir() const { return m_dataDir; }
    uint64_t usage(MemoryCategory c) const { return m_usage[uint32_t(c)]; }
    uint64_t budget(MemoryCategory c) const { return m_budget[uint32_t(c)]; }
    void setBudget(MemoryCategory c, uint64_t bytes) { m_budget[uint32_t(c)] = bytes; } // 0 = no budget
    uint32_t reducedTextures() const { return m_reducedTextures; } // loaded with mips skipped to fit the budget (M9 plan D6)

    // Hot reload (dev; M9 plan D9): loads a texture's file again into the same asset. Once it's in, the old GPU texture
    // is released (deferred) and texture() returns the new one, so whoever built resource sets with it must look again:
    // textureRevision() moves on with every finished reload (the material system, the environment and the particle and
    // sprite atlas caches follow it). A reload that fails keeps the old texture; of several reloads in flight only the
    // latest is kept. False if the path isn't loaded.
    bool reloadTexture(const std::string& path);
    uint32_t textureRevision() const { return m_textureRevision; }

private:
    struct TextureRecord {
        std::string path;
        LoadState state = LoadState::Loading;
        uint32_t refs = 0;
        rhi::TextureHandle texture;
        uint64_t gpuBytes = 0;
        uint32_t mips = 0;
        uint32_t reloads = 0; // hot reloads asked for: only the load of the latest one is kept (M9 review)
    };

    enum class Kind : uint8_t { Model, Texture, Anim, Preset };
    struct Request {
        Kind kind;
        uint32_t handle;
        std::string path;
        uint32_t serial = 0; // textures: which reload this load is for (0 = the first load)
    };
    struct Loaded {
        Kind kind = Kind::Model;
        uint32_t handle = 0;
        uint32_t serial = 0; // Request::serial
        bool ok = false;
        std::string error;
        // model
        std::vector<std::byte> meshData; // laid out exactly like the GPU buffer
        std::vector<assets::ModelNode> nodes;
        std::vector<GpuMesh> meshes;     // slices are offsets into meshData; the buffer is set at finalize
        std::vector<GpuSubmesh> submeshes;
        std::vector<std::string> materialPaths;
        std::vector<assets::MaterialFile> materialFiles;
        std::vector<int32_t> parents;
        std::vector<anim::Transform> rest;
        std::vector<uint32_t> nameHashes;
        std::vector<assets::ModelJoint> joints;
        // clip library
        std::vector<std::byte> animBytes;
        assets::AnimView animView;
        // particle preset
        assets::ParticleEffectFile preset;
        // texture
        assets::DecodedTexture texture;
    };

    void loaderLoop();
    void loadModelFile(const Request& request, Loaded& out) const;
    void loadTextureFile(const Request& request, Loaded& out) const;
    void loadAnimFile(const Request& request, Loaded& out) const;
    void loadPresetFile(const Request& request, Loaded& out) const;
    uint64_t finalize(Loaded& item);
    void enqueue(Kind kind, uint32_t handle, const std::string& path, uint32_t serial = 0);

    rhi::Device* m_device = nullptr;
    std::string m_dataDir;
    assets::TranscodeTargets m_targets;
    MaterialCallbacks m_materialCallbacks;
    rhi::TextureHandle m_grey, m_white, m_black, m_flatNormal;

    core::HandlePool<ModelAsset, ModelTag> m_models{4096};
    core::HandlePool<TextureRecord, TextureAssetTag> m_textures{8192};
    core::HandlePool<AnimAsset, AnimTag> m_anims{1024};
    core::HandlePool<PresetAsset, PresetTag> m_presets{256};
    std::unordered_map<std::string, uint32_t> m_modelByPath;
    std::unordered_map<std::string, uint32_t> m_textureByPath;
    std::unordered_map<std::string, uint32_t> m_animByPath;
    std::unordered_map<std::string, uint32_t> m_presetByPath;
    uint32_t m_inFlight = 0;
    uint64_t m_usage[uint32_t(MemoryCategory::Count)] = {};
    uint64_t m_budget[uint32_t(MemoryCategory::Count)] = {512ull << 20, 256ull << 20}; // Medium: VRAM ≤ 1 GB in total
    uint32_t m_reducedTextures = 0;
    uint32_t m_textureRevision = 0;

    std::vector<std::thread> m_threads;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    bool m_stop = false;
    std::deque<Request> m_requests;
    std::deque<Loaded> m_loaded;
};

} // namespace renderer
