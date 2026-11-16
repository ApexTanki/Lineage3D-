#include "renderer/core/ShaderLibrary.h"

#include "core/Log.h"
#include "platform/FileSystem.h"

namespace renderer {
namespace {

shaderformat::Target targetFor(rhi::Backend backend) {
    switch (backend) {
    case rhi::Backend::Vulkan: return shaderformat::Target::SpirV;
    case rhi::Backend::GL33: return shaderformat::Target::Glsl330;
    }
    return shaderformat::Target::SpirV;
}

} // namespace

ShaderLibrary::ShaderLibrary(rhi::Device& device) : m_device(device) {}

ShaderLibrary::~ShaderLibrary() {
    for (auto& package : m_packages)
        for (Variant& v : package->variants) {
            m_device.destroy(v.program.vertex);
            m_device.destroy(v.program.fragment);
        }
}

core::Result<void> ShaderLibrary::load(const std::string& name, const std::string& path) {
    auto file = platform::readFile(path);
    if (!file) {
        FL_LOG_ERROR("Shader package %s not found at %s", name.c_str(), path.c_str());
        return file.error();
    }
    auto package = std::make_unique<Package>();
    package->name = name;
    package->data = std::move(file.value());
    std::string error;
    if (!package->reader.open(package->data, &error)) {
        FL_LOG_ERROR("Shader package %s is invalid: %s", path.c_str(), error.c_str());
        return core::Error::InvalidData;
    }
    m_packages.push_back(std::move(package));
    return {};
}

core::Result<ShaderLibrary::Program> ShaderLibrary::program(const std::string& name, uint64_t key) {
    Package* package = nullptr;
    for (auto& p : m_packages)
        if (p->name == name) package = p.get();
    if (!package) {
        FL_LOG_ERROR("Shader package %s isn't loaded", name.c_str());
        return core::Error::NotFound;
    }
    for (const Variant& v : package->variants)
        if (v.key == key) return v.program;

    std::optional<shaderformat::VariantView> found = package->reader.find(key);
    if (!found) {
        // Never a silent fallback: a missing permutation is a content/build error (ARCHITECTURE.md §7.4).
        FL_LOG_ERROR("Shader %s has no permutation with key 0x%llx", name.c_str(), (unsigned long long)key);
        return core::Error::NotFound;
    }
    Variant variant{key, std::make_unique<shaderformat::VariantView>(*found), {}};
    variant.program.reflection = &variant.view->reflection;
    const uint32_t target = uint32_t(targetFor(m_device.caps().backend));
    const std::string vsName = name + ".vert", fsName = name + ".frag";

    rhi::ShaderDesc vs;
    vs.debugName = vsName.c_str();
    vs.stage = shaderformat::Stage::Vertex;
    vs.code = variant.view->code[target][uint32_t(shaderformat::Stage::Vertex)];
    vs.reflection = &variant.view->reflection;
    rhi::ShaderDesc fs = vs;
    fs.debugName = fsName.c_str();
    fs.stage = shaderformat::Stage::Fragment;
    fs.code = variant.view->code[target][uint32_t(shaderformat::Stage::Fragment)];

    variant.program.vertex = m_device.createShader(vs);
    variant.program.fragment = m_device.createShader(fs);
    if (!variant.program.vertex || !variant.program.fragment) {
        m_device.destroy(variant.program.vertex);
        m_device.destroy(variant.program.fragment);
        return core::Error::CompileFailed;
    }
    const Program result = variant.program;
    package->variants.push_back(std::move(variant));
    return result;
}

core::Result<ShaderLibrary::ReloadResult> ShaderLibrary::reload(const std::string& name, const std::string& path) {
    Package* package = nullptr;
    for (auto& p : m_packages)
        if (p->name == name) package = p.get();
    if (!package) return core::Error::NotFound;
    auto file = platform::readFile(path);
    if (!file) return file.error();
    std::vector<std::byte> data = std::move(file.value());
    shaderformat::PackageReader reader;
    std::string error;
    if (!reader.open(data, &error)) {
        FL_LOG_ERROR("Hot reload: shader package %s is invalid: %s", path.c_str(), error.c_str());
        return core::Error::InvalidData;
    }
    const uint32_t target = uint32_t(targetFor(m_device.caps().backend));
    const std::string vsName = name + ".vert", fsName = name + ".frag";
    ReloadResult result{uint32_t(package->variants.size()), 0};
    for (Variant& v : package->variants) {
        std::optional<shaderformat::VariantView> found = reader.find(v.key);
        if (!found) {
            FL_LOG_WARN("Hot reload: %s has no variant 0x%llx any more; it keeps its old code", path.c_str(), (unsigned long long)v.key);
            continue;
        }
        // replaceShaders makes the old pipelines again from their descriptions (set layouts, per-draw size, vertex
        // layout), which code with other bindings or inputs wouldn't match; graphics APIs don't check that themselves.
        if (!shaderformat::sameInterface(v.view->reflection, found->reflection)) {
            FL_LOG_WARN("Hot reload: %s variant 0x%llx changed its bindings, per-draw block or vertex inputs: it keeps its old code "
                        "(restart to use the new)",
                        path.c_str(), (unsigned long long)v.key);
            continue;
        }
        auto view = std::make_unique<shaderformat::VariantView>(*found);
        rhi::ShaderDesc vs;
        vs.debugName = vsName.c_str();
        vs.stage = shaderformat::Stage::Vertex;
        vs.code = view->code[target][uint32_t(shaderformat::Stage::Vertex)];
        vs.reflection = &view->reflection;
        rhi::ShaderDesc fs = vs;
        fs.debugName = fsName.c_str();
        fs.stage = shaderformat::Stage::Fragment;
        fs.code = view->code[target][uint32_t(shaderformat::Stage::Fragment)];
        if (!m_device.replaceShaders(v.program.vertex, vs, v.program.fragment, fs)) {
            FL_LOG_WARN("Hot reload: %s variant 0x%llx doesn't build with the new code; it keeps the old", path.c_str(),
                        (unsigned long long)v.key);
            continue;
        }
        m_retiredViews.push_back(std::move(v.view));
        v.view = std::move(view);
        v.program.reflection = &v.view->reflection;
        ++result.replaced;
    }
    m_retiredData.push_back(std::move(package->data));
    package->data = std::move(data); // a moved vector keeps its buffer: the reader's and the views' spans stay valid
    package->reader = std::move(reader);
    return result;
}

core::Result<shaderformat::Axis> ShaderLibrary::axis(const std::string& name, const char* axisName) const {
    for (const auto& p : m_packages) {
        if (p->name != name) continue;
        for (const shaderformat::Axis& a : p->reader.axes())
            if (shaderformat::nameOf(a.name) == axisName) return a;
        FL_LOG_ERROR("Shader package %s has no permutation axis %s", name.c_str(), axisName);
        return core::Error::NotFound;
    }
    FL_LOG_ERROR("Shader package %s isn't loaded", name.c_str());
    return core::Error::NotFound;
}

} // namespace renderer
