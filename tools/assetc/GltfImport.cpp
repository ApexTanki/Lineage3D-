#include "GltfImport.h"

#include "ImageCook.h"
#include "MeshBuilder.h"
#include "assets/Material.h"
#include "assets/NameHash.h"

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <tuple>

namespace tools {
namespace fs = std::filesystem;

namespace {

// glTF URIs and names are UTF-8.
fs::path utf8Path(const std::string& s) {
    return fs::path(std::u8string(s.begin(), s.end()));
}
std::string utf8String(const fs::path& p) {
    const std::u8string s = p.u8string();
    return std::string(s.begin(), s.end());
}

template <size_t N>
void copyName(char (&dst)[N], const std::string& src) {
    std::memset(dst, 0, N);
    std::memcpy(dst, src.data(), std::min(src.size(), N - 1));
}

std::string sanitize(const std::string& s) {
    std::string out;
    for (char c : s) out += (std::isalnum(static_cast<unsigned char>(c)) || c == '-') ? c : '_';
    return out.empty() ? std::string("unnamed") : out;
}

std::string uriDecode(const std::string& uri) {
    std::string out;
    for (size_t i = 0; i < uri.size(); ++i) {
        if (uri[i] == '%' && i + 2 < uri.size()) {
            out += char(std::strtol(uri.substr(i + 1, 2).c_str(), nullptr, 16));
            i += 2;
        } else {
            out += uri[i];
        }
    }
    return out;
}

// Decomposes a TRS matrix (glTF matrices never shear).
void decompose(const float m[16], float t[3], float r[4], float s[3]) {
    const glm::mat4 mat = glm::make_mat4(m);
    glm::vec3 axis[3] = {glm::vec3(mat[0]), glm::vec3(mat[1]), glm::vec3(mat[2])};
    glm::vec3 scale(glm::length(axis[0]), glm::length(axis[1]), glm::length(axis[2]));
    if (glm::determinant(glm::mat3(mat)) < 0.0f) scale.x = -scale.x;
    for (int i = 0; i < 3; ++i) axis[i] /= scale[i] != 0.0f ? scale[i] : 1.0f;
    const glm::quat q = glm::quat_cast(glm::mat3(axis[0], axis[1], axis[2]));
    t[0] = mat[3][0], t[1] = mat[3][1], t[2] = mat[3][2];
    r[0] = q.x, r[1] = q.y, r[2] = q.z, r[3] = q.w;
    s[0] = scale.x, s[1] = scale.y, s[2] = scale.z;
}

bool imageBytes(const cgltf_image* image, const fs::path& baseDir, std::vector<std::byte>& out, std::string& error) {
    if (image->buffer_view) {
        const cgltf_buffer_view* v = image->buffer_view;
        const auto* p = static_cast<const std::byte*>(v->buffer->data) + v->offset;
        out.assign(p, p + v->size);
        return true;
    }
    if (!image->uri) {
        error = "image has neither a URI nor a buffer view";
        return false;
    }
    const std::string uri = image->uri;
    if (uri.rfind("data:", 0) == 0) {
        const size_t comma = uri.find("base64,");
        if (comma == std::string::npos) {
            error = "unsupported data URI";
            return false;
        }
        const char* b64 = uri.c_str() + comma + 7;
        const size_t len = std::strlen(b64);
        size_t size = len / 4 * 3;
        if (len >= 1 && b64[len - 1] == '=') --size;
        if (len >= 2 && b64[len - 2] == '=') --size;
        void* decoded = nullptr;
        cgltf_options options{};
        if (cgltf_load_buffer_base64(&options, size, b64, &decoded) != cgltf_result_success) {
            error = "bad base64 image";
            return false;
        }
        out.assign(static_cast<std::byte*>(decoded), static_cast<std::byte*>(decoded) + size);
        std::free(decoded);
        return true;
    }
    std::ifstream file(baseDir / utf8Path(uriDecode(uri)), std::ios::binary);
    if (!file) {
        error = "cannot open image " + uri;
        return false;
    }
    const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    out.assign(reinterpret_cast<const std::byte*>(bytes.data()), reinterpret_cast<const std::byte*>(bytes.data() + bytes.size()));
    return true;
}

bool readFloats(const cgltf_accessor* a, size_t vertex, float* out, cgltf_size n) {
    return cgltf_accessor_read_float(a, vertex, out, n) != 0;
}

// Perceptual roughness → gloss: the Blinn-Phong exponent that matches a GGX lobe of roughness α = r² is about
// n = 2/α² − 2, and the engine maps gloss g to n = 2^(1+10g) (ARCHITECTURE.md §4.4).
float glossFromRoughness(float roughness) {
    const float r = std::clamp(roughness, 0.02f, 1.0f);
    const float exponent = std::max(2.0f / (r * r * r * r) - 2.0f, 2.0f);
    return std::clamp((std::log2(exponent) - 1.0f) / 10.0f, 0.0f, 1.0f);
}

// One animation channel sampled at kAnimSampleRate over [0, duration] (M6 plan D4). STEP holds a key, LINEAR lerps
// (slerps rotations, as glTF specifies), CUBICSPLINE evaluates the Hermite spline with the file's tangents; before the
// first key and after the last the end values hold. Rotations come out normalized and hemisphere-continuous, so the
// runtime's nlerp never takes the long way; a channel whose keys are all equal collapses to one key.
bool resampleChannel(const cgltf_animation_sampler& sampler, assets::AnimChannel channel, float duration, std::vector<float>& keys,
                     std::string& error) {
    const cgltf_accessor* input = sampler.input;
    const cgltf_accessor* output = sampler.output;
    const uint32_t n = assets::animChannelFloats(channel);
    if (!input || !output || input->count == 0) {
        error = "a channel has no keys";
        return false;
    }
    const bool cubic = sampler.interpolation == cgltf_interpolation_type_cubic_spline;
    const size_t keyCount = input->count;
    if (output->count != keyCount * (cubic ? 3 : 1)) {
        error = "a channel's output count doesn't match its input";
        return false;
    }
    std::vector<float> times(keyCount);
    for (size_t k = 0; k < keyCount; ++k) cgltf_accessor_read_float(input, k, &times[k], 1);
    auto value = [&](size_t index, float* out) { cgltf_accessor_read_float(output, index, out, n); };
    // Keys span exactly [0, duration] (animEffectiveRate), so a loop's last key is its end value.
    const float rate = assets::animEffectiveRate(duration);
    const uint32_t frames = assets::animFrameCount(duration, rate);
    keys.assign(size_t(frames) * n, 0.0f);
    for (uint32_t f = 0; f < frames; ++f) {
        const float t = std::min(float(f) / rate, duration);
        float* out = keys.data() + size_t(f) * n;
        const size_t k1 = size_t(std::upper_bound(times.begin(), times.end(), t) - times.begin()); // first key after t
        if (k1 == 0) {
            value(cubic ? 1 : 0, out);
            continue;
        }
        if (k1 >= keyCount) {
            value(cubic ? (keyCount - 1) * 3 + 1 : keyCount - 1, out);
            continue;
        }
        const size_t k0 = k1 - 1;
        const float dt = times[k1] - times[k0];
        const float s = dt > 0.0f ? (t - times[k0]) / dt : 0.0f;
        if (sampler.interpolation == cgltf_interpolation_type_step) {
            value(cubic ? k0 * 3 + 1 : k0, out);
            continue;
        }
        if (cubic) {
            float p0[4] = {}, m0[4] = {}, p1[4] = {}, m1[4] = {};
            value(k0 * 3 + 1, p0);
            value(k0 * 3 + 2, m0); // out-tangent of k0
            value(k1 * 3 + 1, p1);
            value(k1 * 3 + 0, m1); // in-tangent of k1
            const float s2 = s * s, s3 = s2 * s;
            const float h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
            for (uint32_t i = 0; i < n; ++i) out[i] = h00 * p0[i] + h10 * dt * m0[i] + h01 * p1[i] + h11 * dt * m1[i];
            continue;
        }
        float a[4] = {}, b[4] = {};
        value(k0, a);
        value(k1, b);
        if (channel == assets::AnimChannel::Rotation) {
            const glm::quat q = glm::slerp(glm::normalize(glm::quat(a[3], a[0], a[1], a[2])), glm::normalize(glm::quat(b[3], b[0], b[1], b[2])), s);
            out[0] = q.x, out[1] = q.y, out[2] = q.z, out[3] = q.w;
        } else {
            for (uint32_t i = 0; i < n; ++i) out[i] = a[i] + (b[i] - a[i]) * s;
        }
    }
    if (channel == assets::AnimChannel::Rotation)
        for (uint32_t f = 0; f < frames; ++f) {
            float* q = keys.data() + size_t(f) * 4;
            const float len = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (len > 1e-12f) {
                for (int i = 0; i < 4; ++i) q[i] /= len;
            } else {
                q[0] = q[1] = q[2] = 0.0f, q[3] = 1.0f;
            }
            if (f == 0) continue;
            const float* prev = q - 4;
            if (prev[0] * q[0] + prev[1] * q[1] + prev[2] * q[2] + prev[3] * q[3] < 0.0f)
                for (int i = 0; i < 4; ++i) q[i] = -q[i];
        }
    bool constant = true;
    for (size_t i = n; i < keys.size() && constant; ++i)
        constant = std::fabs(keys[i] - keys[i % n]) <= 1e-6f * std::max(1.0f, std::fabs(keys[i % n]));
    if (constant) keys.resize(n);
    return true;
}

} // namespace

bool writeFileAtomic(const std::string& path, const void* data, size_t size, std::string& error) {
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    const std::string temp = path + ".tmp";
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        if (!file || !file.write(static_cast<const char*>(data), std::streamsize(size))) {
            error = "cannot write " + temp;
            return false;
        }
    }
    fs::rename(temp, path, ec);
    if (ec) {
        error = "cannot replace " + path + ": " + ec.message();
        return false;
    }
    return true;
}

bool importGltf(const std::string& gltfPath, const std::string& name, const std::string& dataDir, assets::ModelData& out,
                std::string& error, bool generateLods, assets::AnimData* anims) {
    cgltf_options options{};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, gltfPath.c_str(), &data) != cgltf_result_success) {
        error = "cannot parse " + gltfPath;
        return false;
    }
    struct Guard {
        cgltf_data* d;
        ~Guard() { cgltf_free(d); }
    } guard{data};
    if (cgltf_load_buffers(&options, data, gltfPath.c_str()) != cgltf_result_success) {
        error = "cannot load the buffers of " + gltfPath;
        return false;
    }
    if (cgltf_validate(data) != cgltf_result_success) {
        error = gltfPath + " fails glTF validation";
        return false;
    }
    const fs::path baseDir = fs::path(gltfPath).parent_path();
    out = assets::ModelData{};

    // Different glTF names can sanitize to the same file name ("Mat A" / "Mat_A", two images with one stem, a material
    // called "default"); every output gets its own path instead of silently overwriting another.
    std::set<std::string> usedPaths;
    auto uniquePath = [&](const std::string& stem, const char* extension) {
        std::string path = stem + extension;
        for (int n = 2; !usedPaths.insert(path).second; ++n) path = stem + "_" + std::to_string(n) + extension;
        return path;
    };

    // Images are decoded once; textures are cooked once per (image, role, cutoff).
    std::map<size_t, Image> decoded;
    auto decodeImage = [&](const cgltf_image* image, const Image*& outImage) -> bool {
        const size_t index = size_t(image - data->images);
        if (auto it = decoded.find(index); it != decoded.end()) {
            outImage = &it->second;
            return true;
        }
        std::vector<std::byte> bytes;
        Image pixels;
        if (!imageBytes(image, baseDir, bytes, error) || !loadImageMemory(bytes, pixels, error)) return false;
        outImage = &decoded.emplace(index, std::move(pixels)).first->second;
        return true;
    };
    auto imageStem = [&](const cgltf_image* image) {
        const size_t index = size_t(image - data->images);
        std::string imageName = image->name ? image->name : "";
        if (imageName.empty() && image->uri && std::strncmp(image->uri, "data:", 5) != 0)
            imageName = utf8String(utf8Path(uriDecode(image->uri)).stem());
        return sanitize(imageName.empty() ? "image" + std::to_string(index) : imageName);
    };
    auto cookPixels = [&](const Image& pixels, const std::string& stem, TextureRole role, float cutoff, std::string& outPath) -> bool {
        const std::string rel = uniquePath("textures/" + name + "/" + stem, ".ktx2");
        if (!cookTexture(pixels, role, (fs::path(dataDir) / rel).string(), error, 2048, cutoff)) return false;
        std::printf("  texture %s (%ux%u, %s)\n", rel.c_str(), pixels.width, pixels.height, textureRoleName(role));
        outPath = rel;
        return true;
    };
    std::map<std::tuple<size_t, int, float>, std::string> cooked;
    auto cookImage = [&](const cgltf_image* image, TextureRole role, float cutoff, std::string& outPath) -> bool {
        const auto key = std::make_tuple(size_t(image - data->images), int(role), role == TextureRole::AlbedoCutout ? cutoff : 0.0f);
        if (auto it = cooked.find(key); it != cooked.end()) {
            outPath = it->second;
            return true;
        }
        const Image* pixels = nullptr;
        const std::string suffix = role == TextureRole::Albedo ? "" : std::string("_") + textureRoleName(role);
        if (!decodeImage(image, pixels) || !cookPixels(*pixels, imageStem(image) + suffix, role, cutoff, outPath)) return false;
        cooked[key] = outPath;
        return true;
    };
    auto usable = [&](const cgltf_texture_view& view, const char* what, const std::string& material) {
        if (!view.texture || !view.texture->image) return false;
        if (view.texcoord != 0)
            std::printf("  note: %s: the %s map uses UV set %d; the engine samples UV set 0\n", material.c_str(), what, view.texcoord);
        if (view.has_transform)
            std::printf("  note: %s: KHR_texture_transform on the %s map is ignored\n", material.c_str(), what);
        return true;
    };

    // Material slots: one per glTF material, plus "default" for primitives without one.
    std::vector<int> slotOfMaterial(data->materials_count, -1);
    int defaultSlot = -1;
    auto addSlot = [&](const std::string& slotName, const assets::MaterialFile& mf) -> bool {
        assets::ModelMaterialSlot slot{};
        const std::string rel = uniquePath("materials/" + name + "/" + slotName, ".fmat");
        copyName(slot.name, slotName);
        copyName(slot.path, rel);
        const std::string text = assets::writeMaterial(mf);
        if (!writeFileAtomic((fs::path(dataDir) / rel).string(), text.data(), text.size(), error)) return false;
        out.materials.push_back(slot);
        return true;
    };
    bool notedOcclusion = false;
    for (size_t i = 0; i < data->materials_count; ++i) {
        const cgltf_material& m = data->materials[i];
        const std::string slotName = sanitize(m.name ? m.name : "material" + std::to_string(i));
        assets::MaterialFile mf;
        mf.blend = m.alpha_mode == cgltf_alpha_mode_mask    ? assets::BlendMode::AlphaTest
                   : m.alpha_mode == cgltf_alpha_mode_blend ? assets::BlendMode::AlphaBlend
                                                            : assets::BlendMode::Opaque;
        mf.alphaCutoff = m.alpha_cutoff;
        mf.doubleSided = m.double_sided != 0;
        const TextureRole albedoRole = mf.blend == assets::BlendMode::Opaque      ? TextureRole::Albedo
                                       : mf.blend == assets::BlendMode::AlphaTest ? TextureRole::AlbedoCutout
                                                                                  : TextureRole::AlbedoAlpha;
        // Alpha test compares map alpha × base-color alpha with the cutoff, so a cutout map keeps its coverage at
        // cutoff / factor alpha when the factor alpha stays in the material.
        auto mapCutoff = [&](float factorAlpha) { return std::clamp(mf.alphaCutoff / std::max(factorAlpha, 1e-3f), 0.0f, 1.0f); };

        if (m.has_pbr_specular_glossiness) {
            // The engine's own workflow: taken as is.
            const cgltf_pbr_specular_glossiness& sg = m.pbr_specular_glossiness;
            mf.baseColor = glm::vec4(sg.diffuse_factor[0], sg.diffuse_factor[1], sg.diffuse_factor[2], sg.diffuse_factor[3]);
            if (usable(sg.diffuse_texture, "diffuse", slotName) &&
                !cookImage(sg.diffuse_texture.texture->image, albedoRole, mapCutoff(sg.diffuse_factor[3]), mf.albedo))
                return false;
            mf.specular = glm::vec3(sg.specular_factor[0], sg.specular_factor[1], sg.specular_factor[2]);
            mf.gloss = std::clamp(sg.glossiness_factor, 0.0f, 1.0f);
            if (usable(sg.specular_glossiness_texture, "specular-glossiness", slotName)) {
                if (!cookImage(sg.specular_glossiness_texture.texture->image, TextureRole::SpecGloss, 0.5f, mf.specGloss)) return false;
                mf.features |= assets::kFeatureSpecGlossMap;
            }
        } else {
            // Metallic-roughness (also glTF's default) converted to specular/gloss (M2 plan D9):
            //   diffuse = base × (1 − metallic), specular = lerp(0.04, base, metallic), gloss from roughness.
            const cgltf_pbr_metallic_roughness& mr = m.pbr_metallic_roughness;
            const glm::vec4 factor(mr.base_color_factor[0], mr.base_color_factor[1], mr.base_color_factor[2], mr.base_color_factor[3]);
            const bool baseMap = usable(mr.base_color_texture, "base color", slotName);
            const bool mrMap = usable(mr.metallic_roughness_texture, "metallic-roughness", slotName);
            const float metallic = std::clamp(mr.metallic_factor, 0.0f, 1.0f), roughness = std::clamp(mr.roughness_factor, 0.0f, 1.0f);
            if (!mrMap && (metallic == 0.0f || !baseMap)) {
                // Uniform metalness: the conversion is pure parameters.
                const glm::vec3 base(factor);
                mf.baseColor = glm::vec4(base * (1.0f - metallic), factor.a);
                mf.specular = glm::mix(glm::vec3(0.04f), base, metallic);
                mf.gloss = glossFromRoughness(roughness);
                if (baseMap && !cookImage(mr.base_color_texture.texture->image, albedoRole, mapCutoff(factor.a), mf.albedo)) return false;
            } else {
                // Per-texel metalness or roughness: bake a new albedo and a spec/gloss map at the larger source size.
                const Image *base = nullptr, *metalRough = nullptr;
                if (baseMap && !decodeImage(mr.base_color_texture.texture->image, base)) return false;
                if (mrMap && !decodeImage(mr.metallic_roughness_texture.texture->image, metalRough)) return false;
                const uint32_t w = std::max(base ? base->width : 1u, metalRough ? metalRough->width : 1u);
                const uint32_t h = std::max(base ? base->height : 1u, metalRough ? metalRough->height : 1u);
                Image baseSized, mrSized;
                if (base) baseSized = base->width == w && base->height == h ? *base : resizeImage(*base, w, h, true);
                if (metalRough) mrSized = metalRough->width == w && metalRough->height == h ? *metalRough : resizeImage(*metalRough, w, h, false);
                Image albedo, specGloss;
                albedo.width = specGloss.width = w;
                albedo.height = specGloss.height = h;
                albedo.rgba.resize(size_t(w) * h * 4);
                specGloss.rgba.resize(size_t(w) * h * 4);
                for (size_t p = 0, n = size_t(w) * h; p < n; ++p) {
                    glm::vec3 color(factor);
                    float alpha = factor.a;
                    if (base) {
                        const uint8_t* b = &baseSized.rgba[p * 4];
                        color *= glm::vec3(srgbToLinear(b[0]), srgbToLinear(b[1]), srgbToLinear(b[2]));
                        alpha *= float(b[3]) / 255.0f;
                    }
                    float texelMetal = metallic, texelRough = roughness;
                    if (metalRough) { // glTF: G = roughness, B = metalness
                        texelRough *= float(mrSized.rgba[p * 4 + 1]) / 255.0f;
                        texelMetal *= float(mrSized.rgba[p * 4 + 2]) / 255.0f;
                    }
                    const glm::vec3 diffuse = color * (1.0f - texelMetal);
                    const glm::vec3 specular = glm::mix(glm::vec3(0.04f), color, texelMetal);
                    uint8_t* a = &albedo.rgba[p * 4];
                    uint8_t* s = &specGloss.rgba[p * 4];
                    for (int c = 0; c < 3; ++c) {
                        a[c] = linearToSrgb(diffuse[c]);
                        s[c] = linearToSrgb(specular[c]);
                    }
                    a[3] = uint8_t(std::clamp(std::round(alpha * 255.0f), 0.0f, 255.0f));
                    s[3] = uint8_t(std::clamp(std::round(glossFromRoughness(texelRough) * 255.0f), 0.0f, 255.0f));
                }
                if (!cookPixels(albedo, slotName + "_albedo", albedoRole, mf.alphaCutoff, mf.albedo) ||
                    !cookPixels(specGloss, slotName + "_specgloss", TextureRole::SpecGloss, 0.5f, mf.specGloss))
                    return false;
                mf.features |= assets::kFeatureSpecGlossMap;
                mf.baseColor = glm::vec4(1.0f); // the factors are baked into the maps
                mf.specular = glm::vec3(1.0f);
                mf.gloss = 1.0f;
            }
        }
        if (usable(m.normal_texture, "normal", slotName)) {
            if (!cookImage(m.normal_texture.texture->image, TextureRole::Normal, 0.5f, mf.normal)) return false;
            mf.features |= assets::kFeatureNormalMap;
            mf.normalStrength = std::clamp(m.normal_texture.scale, 0.0f, 10.0f);
        }
        const glm::vec3 emissive(m.emissive_factor[0], m.emissive_factor[1], m.emissive_factor[2]);
        if (emissive != glm::vec3(0.0f)) { // glTF: emission = factor × map, so a zero factor means none
            mf.emissionColor = emissive;
            mf.emissionIntensity = m.has_emissive_strength ? std::max(0.0f, m.emissive_strength.emissive_strength) : 1.0f;
            if (usable(m.emissive_texture, "emissive", slotName)) {
                if (!cookImage(m.emissive_texture.texture->image, TextureRole::Emission, 0.5f, mf.emission)) return false;
                mf.features |= assets::kFeatureEmissionMap;
            }
        }
        if (m.occlusion_texture.texture && !notedOcclusion) {
            std::printf("  note: occlusion maps are ignored (baked light comes from a map's lightmaps)\n");
            notedOcclusion = true;
        }
        slotOfMaterial[i] = int(out.materials.size());
        if (!addSlot(slotName, mf)) return false;
    }

    // Nodes of the default scene first, parents before children, so that skins can name their joints by engine node index.
    // Clips bind to nodes by name hash (M6 plan D3), so every node gets a unique, non-empty name (truncated names
    // included).
    std::map<const cgltf_node*, int> nodeIndex;
    std::set<std::string> usedNames;
    auto uniqueName = [&](std::string base) {
        if (base.size() > assets::kModelNameLength - 8) base.resize(assets::kModelNameLength - 8);
        std::string candidate = base;
        for (int k = 2; !usedNames.insert(candidate).second; ++k) candidate = base + "_" + std::to_string(k);
        return candidate;
    };
    struct MeshNode {
        int node;
        const cgltf_mesh* mesh;
        const cgltf_skin* skin;
    };
    std::vector<MeshNode> meshNodes;
    auto visit = [&](auto& self, const cgltf_node* node, int parent) -> void {
        assets::ModelNode n{};
        const size_t gltfIndex = size_t(node - data->nodes);
        copyName(n.name, uniqueName(node->name && *node->name ? std::string(node->name) : "node" + std::to_string(gltfIndex)));
        n.parent = parent;
        n.mesh = -1; // set below, once its mesh is built
        if (node->has_matrix) {
            decompose(node->matrix, n.translation, n.rotation, n.scale);
        } else {
            const float t[3] = {0, 0, 0}, r[4] = {0, 0, 0, 1}, s[3] = {1, 1, 1};
            std::memcpy(n.translation, node->has_translation ? node->translation : t, sizeof(t));
            std::memcpy(n.rotation, node->has_rotation ? node->rotation : r, sizeof(r));
            std::memcpy(n.scale, node->has_scale ? node->scale : s, sizeof(s));
        }
        const int index = int(out.nodes.size());
        nodeIndex[node] = index;
        out.nodes.push_back(n);
        if (node->mesh) meshNodes.push_back({index, node->mesh, node->skin});
        for (size_t c = 0; c < node->children_count; ++c) self(self, node->children[c], index);
    };
    const cgltf_scene* scene = data->scene ? data->scene : (data->scenes_count ? &data->scenes[0] : nullptr);
    if (scene) {
        for (size_t i = 0; i < scene->nodes_count; ++i) visit(visit, scene->nodes[i], -1);
    } else {
        for (size_t i = 0; i < data->nodes_count; ++i)
            if (!data->nodes[i].parent) visit(visit, &data->nodes[i], -1);
    }

    // Meshes: one engine mesh per (glTF mesh, skin) pair; its triangle primitives are the submeshes. A skin with more than
    // assets::kMaxSkinJoints joints in use splits the mesh into parts (M6 plan D4): the first part stays on the node,
    // the others become sibling nodes "<node>_part<k>".
    bool notedInfluences = false, notedUnweighted = false, notedMorph = false;
    std::map<std::pair<const cgltf_mesh*, const cgltf_skin*>, std::vector<int>> built;
    for (const MeshNode& mn : meshNodes) {
        const auto key = std::make_pair(mn.mesh, mn.skin);
        auto found = built.find(key);
        if (found == built.end()) {
            const cgltf_mesh& gm = *mn.mesh;
            const size_t mi = size_t(mn.mesh - data->meshes);
            RawMesh raw;
            raw.hasNormals = raw.hasTangents = raw.hasUv0 = raw.hasUv1 = true;
            // The skin: joint nodes as engine indices, and inverse binds (identity when the file has none).
            bool skinned = mn.skin != nullptr;
            if (skinned) {
                for (size_t j = 0; j < mn.skin->joints_count && skinned; ++j) {
                    const auto it = nodeIndex.find(mn.skin->joints[j]);
                    if (it == nodeIndex.end()) {
                        std::printf("  note: mesh %zu: a joint of its skin is outside the scene; drawn rigid at its bind pose\n", mi);
                        skinned = false;
                        break;
                    }
                    raw.skinJoints.push_back(uint32_t(it->second));
                    glm::mat4 inverseBind(1.0f);
                    if (mn.skin->inverse_bind_matrices) {
                        float m[16];
                        cgltf_accessor_read_float(mn.skin->inverse_bind_matrices, j, m, 16);
                        inverseBind = glm::make_mat4(m);
                    }
                    raw.inverseBinds.push_back(inverseBind);
                }
                if (!skinned || raw.skinJoints.empty()) {
                    raw.skinJoints.clear();
                    raw.inverseBinds.clear();
                    skinned = false;
                }
            }
            for (size_t pi = 0; pi < gm.primitives_count; ++pi) {
                const cgltf_primitive& prim = gm.primitives[pi];
                if (prim.type != cgltf_primitive_type_triangles) {
                    std::printf("  note: mesh %zu primitive %zu isn't a triangle list; skipped\n", mi, pi);
                    continue;
                }
                if (prim.targets_count && !notedMorph) {
                    std::printf("  note: morph targets are ignored\n");
                    notedMorph = true;
                }
                const cgltf_accessor *position = nullptr, *normal = nullptr, *tangent = nullptr, *uv0 = nullptr, *uv1 = nullptr,
                                     *joints = nullptr, *weights = nullptr;
                for (size_t a = 0; a < prim.attributes_count; ++a) {
                    const cgltf_attribute& attr = prim.attributes[a];
                    if (attr.type == cgltf_attribute_type_position) position = attr.data;
                    else if (attr.type == cgltf_attribute_type_normal) normal = attr.data;
                    else if (attr.type == cgltf_attribute_type_tangent) tangent = attr.data;
                    else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) uv0 = attr.data;
                    else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 1) uv1 = attr.data;
                    else if (attr.type == cgltf_attribute_type_joints && attr.index == 0) joints = attr.data;
                    else if (attr.type == cgltf_attribute_type_weights && attr.index == 0) weights = attr.data;
                    else if ((attr.type == cgltf_attribute_type_joints || attr.type == cgltf_attribute_type_weights) && attr.index > 0 &&
                             !notedInfluences) {
                        std::printf("  note: only the first four joint influences of a vertex are used\n");
                        notedInfluences = true;
                    }
                }
                if (!position) {
                    std::printf("  note: mesh %zu primitive %zu has no positions; skipped\n", mi, pi);
                    continue;
                }
                raw.hasNormals &= normal != nullptr;
                raw.hasTangents &= tangent != nullptr;
                raw.hasUv0 &= uv0 != nullptr;
                raw.hasUv1 &= uv1 != nullptr; // TEXCOORD_1 is the lightmap set; without it the unwrapper makes one
                const auto base = uint32_t(raw.vertices.size());
                for (size_t v = 0; v < position->count; ++v) {
                    RawVertex vtx;
                    readFloats(position, v, &vtx.position.x, 3);
                    if (normal) readFloats(normal, v, &vtx.normal.x, 3);
                    if (tangent) readFloats(tangent, v, &vtx.tangent.x, 4);
                    if (uv0) readFloats(uv0, v, &vtx.uv0.x, 2);
                    if (uv1) readFloats(uv1, v, &vtx.uv1.x, 2);
                    if (skinned) {
                        cgltf_uint j[4] = {0, 0, 0, 0};
                        float w[4] = {0, 0, 0, 0};
                        if (joints && weights) {
                            cgltf_accessor_read_uint(joints, v, j, 4);
                            readFloats(weights, v, w, 4); // normalized integer weights come back in [0, 1]
                        }
                        float total = 0.0f;
                        for (int k = 0; k < 4; ++k) {
                            if (j[k] >= raw.skinJoints.size()) w[k] = 0.0f; // a joint past the skin can't be honored
                            total += std::max(w[k], 0.0f);
                        }
                        if (!(total > 0.0f)) { // no usable influence: hang the vertex on the skin's first joint
                            j[0] = 0;
                            w[0] = 1.0f;
                            w[1] = w[2] = w[3] = 0.0f;
                            if (!notedUnweighted) {
                                std::printf("  note: mesh %zu has vertices without joint weights; they follow the first joint\n", mi);
                                notedUnweighted = true;
                            }
                        }
                        vtx.joints = glm::u16vec4(uint16_t(std::min<cgltf_uint>(j[0], 0xFFFF)), uint16_t(std::min<cgltf_uint>(j[1], 0xFFFF)),
                                                  uint16_t(std::min<cgltf_uint>(j[2], 0xFFFF)), uint16_t(std::min<cgltf_uint>(j[3], 0xFFFF)));
                        vtx.weights = glm::vec4(w[0], w[1], w[2], w[3]);
                    }
                    raw.vertices.push_back(vtx);
                }
                RawSubmesh sub;
                sub.firstIndex = uint32_t(raw.indices.size());
                if (prim.indices) {
                    for (size_t k = 0; k < prim.indices->count; ++k)
                        raw.indices.push_back(base + uint32_t(cgltf_accessor_read_index(prim.indices, k)));
                } else {
                    for (uint32_t k = 0; k < uint32_t(position->count); ++k) raw.indices.push_back(base + k);
                }
                sub.indexCount = uint32_t(raw.indices.size()) - sub.firstIndex;
                if (prim.material) {
                    sub.materialSlot = uint32_t(slotOfMaterial[size_t(prim.material - data->materials)]);
                } else {
                    if (defaultSlot < 0) {
                        defaultSlot = int(out.materials.size());
                        if (!addSlot("default", assets::MaterialFile{})) return false;
                    }
                    sub.materialSlot = uint32_t(defaultSlot);
                }
                if (sub.indexCount >= 3) raw.submeshes.push_back(sub);
            }
            std::vector<int> parts;
            if (!raw.submeshes.empty()) {
                raw.generateLods = generateLods;
                std::vector<RawMesh> pieces;
                if (!splitSkin(raw, pieces, error)) {
                    error = "mesh " + std::to_string(mi) + ": " + error;
                    return false;
                }
                if (pieces.size() > 1)
                    std::printf("  mesh %zu: skin split into %zu parts of at most %u joints\n", mi, pieces.size(), assets::kMaxSkinJoints);
                for (RawMesh& piece : pieces) {
                    assets::ModelData::Mesh mesh;
                    if (!buildMesh(std::move(piece), mesh, out.submeshes, error)) {
                        error = "mesh " + std::to_string(mi) + ": " + error;
                        return false;
                    }
                    parts.push_back(int(out.meshes.size()));
                    out.meshes.push_back(std::move(mesh));
                }
            }
            found = built.emplace(key, std::move(parts)).first;
        }
        const std::vector<int>& parts = found->second;
        if (parts.empty()) continue;
        out.nodes[size_t(mn.node)].mesh = parts[0];
        for (size_t p = 1; p < parts.size(); ++p) {
            assets::ModelNode extra = out.nodes[size_t(mn.node)]; // same parent (which precedes it) and transform
            copyName(extra.name, uniqueName(std::string(out.nodes[size_t(mn.node)].name) + "_part" + std::to_string(p)));
            extra.mesh = parts[p];
            out.nodes.push_back(extra);
        }
    }
    if (out.meshes.empty() || out.nodes.empty()) {
        error = gltfPath + " has no drawable meshes";
        return false;
    }

    // Animations (M6 plan D3, D4): every channel on a scene node, resampled at kAnimSampleRate from 0 to the clip's last
    // key. Morph-target weights are ignored.
    if (anims) {
        anims->clips.clear();
        std::set<std::string> clipNames;
        for (size_t ai = 0; ai < data->animations_count; ++ai) {
            const cgltf_animation& a = data->animations[ai];
            assets::AnimData::Clip clip;
            std::string clipName = sanitize(a.name && *a.name ? a.name : "animation" + std::to_string(ai));
            if (clipName.size() > assets::kModelNameLength - 8) clipName.resize(assets::kModelNameLength - 8);
            for (int k = 2; !clipNames.insert(clipName).second; ++k) clipName = clipName + "_" + std::to_string(k);
            clip.name = clipName;
            for (size_t c = 0; c < a.channels_count; ++c) {
                const cgltf_accessor* input = a.channels[c].sampler ? a.channels[c].sampler->input : nullptr;
                if (input && input->count) {
                    float last = 0.0f;
                    cgltf_accessor_read_float(input, input->count - 1, &last, 1);
                    clip.duration = std::max(clip.duration, last);
                }
            }
            for (size_t c = 0; c < a.channels_count; ++c) {
                const cgltf_animation_channel& ch = a.channels[c];
                if (ch.target_path == cgltf_animation_path_type_weights) {
                    if (!notedMorph) std::printf("  note: morph targets are ignored\n");
                    notedMorph = true;
                    continue;
                }
                const auto node = nodeIndex.find(ch.target_node);
                if (node == nodeIndex.end() || !ch.sampler) continue;
                assets::AnimData::Track track;
                track.nodeHash = assets::nameHash(out.nodes[size_t(node->second)].name);
                track.channel = ch.target_path == cgltf_animation_path_type_rotation ? assets::AnimChannel::Rotation
                                : ch.target_path == cgltf_animation_path_type_scale  ? assets::AnimChannel::Scale
                                                                                     : assets::AnimChannel::Translation;
                if (ch.target_path != cgltf_animation_path_type_translation && ch.target_path != cgltf_animation_path_type_rotation &&
                    ch.target_path != cgltf_animation_path_type_scale)
                    continue;
                if (!resampleChannel(*ch.sampler, track.channel, clip.duration, track.keys, error)) {
                    error = "animation '" + clipName + "': " + error;
                    return false;
                }
                clip.tracks.push_back(std::move(track));
            }
            clip.sampleRate = assets::animEffectiveRate(clip.duration); // what resampleChannel used
            if (!clip.tracks.empty()) anims->clips.push_back(std::move(clip));
        }
        if (!anims->clips.empty()) std::printf("  %zu animation clip(s)\n", anims->clips.size());
    }
    return true;
}

} // namespace tools
